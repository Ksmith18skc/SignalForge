"""BallparkPal job orchestration.

The API spawns ``scripts/update_ballparkpal_cache.py`` as a **subprocess**
and a daemon watcher thread streams its output into the
``ballparkpal_jobs`` row. The API process itself never imports
Playwright; only the spawned child does.

Concurrency rule: only one job (refresh or login) may be active at a
time. Callers see ``BallparkPalBusyError`` rather than racing two
browsers against the same persistent profile.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

import app.db as _db_module
from app.models import BallparkPalJob

logger = logging.getLogger(__name__)

# Defaults. These can be overridden per-call but the wall-clock cap exists
# so a hung browser never sits forever holding the concurrency lock.
DEFAULT_REFRESH_TIMEOUT_S = 600     # 10 minutes for a full scrape
DEFAULT_LOGIN_TIMEOUT_S = 600       # 10 minutes for the operator to finish
LOG_TAIL_CAP_BYTES = 16_000          # rolling tail kept in the DB row

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_SCRIPT = REPO_ROOT / "scripts" / "update_ballparkpal_cache.py"
PROFILE_DIR = REPO_ROOT / ".cache" / "ballparkpal_profile"

ACTIVE_STATUSES = {"queued", "running"}
TERMINAL_STATUSES = {"success", "failed"}


class BallparkPalBusyError(RuntimeError):
    """Raised when a refresh/login is already in-flight."""


@dataclass
class JobSpec:
    mode: str  # refresh | login
    pages: list[str]
    slate_date: str | None
    headless: bool
    timeout_seconds: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def active_job(db: Session) -> BallparkPalJob | None:
    """Return whichever job is currently queued/running, if any.

    Used both by the concurrency guard and by the dashboard's "is something
    running right now?" polling state.
    """
    query = (
        select(BallparkPalJob)
        .where(BallparkPalJob.status.in_(list(ACTIVE_STATUSES)))
        .order_by(BallparkPalJob.started_at.desc())
        .limit(1)
    )
    return db.scalar(query)


def get_job(db: Session, job_id: str) -> BallparkPalJob | None:
    return db.scalar(select(BallparkPalJob).where(BallparkPalJob.job_id == job_id))


def list_recent(db: Session, *, limit: int = 20) -> list[BallparkPalJob]:
    query = (
        select(BallparkPalJob)
        .order_by(BallparkPalJob.started_at.desc())
        .limit(limit)
    )
    return list(db.scalars(query))


def has_persistent_profile() -> bool:
    """Heuristic: did a login flow ever populate the profile directory?

    A fresh profile directory created by Playwright is at least populated
    with a ``Default`` subfolder, so this lets the dashboard show "Run
    login initialization first" before the user wastes a refresh on an
    unauthenticated session.
    """
    if not PROFILE_DIR.exists():
        return False
    try:
        return any(PROFILE_DIR.iterdir())
    except OSError:
        return False


def start_refresh(
    db: Session,
    *,
    pages: list[str] | None = None,
    slate_date: str | None = None,
    headless: bool = True,
    timeout_seconds: int = DEFAULT_REFRESH_TIMEOUT_S,
) -> BallparkPalJob:
    """Spawn a refresh subprocess. Returns the new job row (status=queued)."""
    spec = JobSpec(
        mode="refresh",
        pages=pages or [],
        slate_date=slate_date,
        headless=headless,
        timeout_seconds=int(timeout_seconds or DEFAULT_REFRESH_TIMEOUT_S),
    )
    return _start(db, spec)


def start_login(
    db: Session,
    *,
    timeout_seconds: int = DEFAULT_LOGIN_TIMEOUT_S,
) -> BallparkPalJob:
    """Spawn the login subprocess (headed Playwright)."""
    spec = JobSpec(
        mode="login",
        pages=[],
        slate_date=None,
        headless=False,
        timeout_seconds=int(timeout_seconds or DEFAULT_LOGIN_TIMEOUT_S),
    )
    return _start(db, spec)


def signal_finish(db: Session, job_id: str) -> bool:
    """Tell a running login job "I'm done logging in" by closing its stdin.

    The CLI's login_flow blocks on ``input()`` — closing stdin sends EOF
    which the CLI handles by closing the browser. Returns True if we
    signaled, False if there was nothing to signal.
    """
    job = get_job(db, job_id)
    if job is None or job.status not in ACTIVE_STATUSES:
        return False
    handle = _ACTIVE_HANDLES.get(job_id)
    if handle is None or handle.popen is None:
        return False
    stdin = handle.popen.stdin
    if stdin is None or stdin.closed:
        return False
    try:
        stdin.write(b"\n")
        stdin.flush()
    except (OSError, ValueError):
        pass
    try:
        stdin.close()
    except (OSError, ValueError):
        pass
    return True


# ---------------------------------------------------------------------------
# Internal: subprocess + watcher
# ---------------------------------------------------------------------------


@dataclass
class _Handle:
    """Process + state we keep in-memory to support signal/cancel.

    The DB row is the source of truth for status; this handle is just
    so the API can poke the running process (close stdin, etc.).
    """
    popen: subprocess.Popen | None


_ACTIVE_HANDLES: dict[str, _Handle] = {}
_START_LOCK = threading.Lock()


def _start(db: Session, spec: JobSpec) -> BallparkPalJob:
    """Hold a short critical section so two simultaneous POSTs can't both
    create a job. The actual subprocess is launched outside the lock so
    we don't block the API thread on browser startup.
    """
    with _START_LOCK:
        busy = active_job(db)
        if busy is not None:
            raise BallparkPalBusyError(
                f"BallparkPal job {busy.job_id} is already {busy.status}."
            )
        job = BallparkPalJob(
            job_id=uuid.uuid4().hex[:12],
            mode=spec.mode,
            status="queued",
            pages=list(spec.pages),
            slate_date=spec.slate_date,
            headless=spec.headless,
            logs="",
        )
        db.add(job)
        db.commit()
        # Refresh so the SQLAlchemy default-stamped fields are populated
        # before the API hands the row back to the caller.
        db.refresh(job)
        job_id = job.job_id
        spec_snapshot = JobSpec(
            mode=spec.mode,
            pages=list(spec.pages),
            slate_date=spec.slate_date,
            headless=spec.headless,
            timeout_seconds=spec.timeout_seconds,
        )

    popen, error = _spawn_process(job_id, spec_snapshot)
    if popen is None:
        # Failed at spawn time. Mark failed immediately so the dashboard
        # surfaces the error rather than waiting for a watcher that won't
        # run.
        _finalize(job_id, status="failed", return_code=None, error_message=error)
        # Re-read so the caller gets the failed status.
        return _refetch(db, job_id) or job
    _ACTIVE_HANDLES[job_id] = _Handle(popen=popen)
    _mark_running(job_id, pid=popen.pid)
    watcher = threading.Thread(
        target=_watch,
        name=f"bpp-watch-{job_id}",
        args=(job_id, popen, spec_snapshot.timeout_seconds),
        daemon=True,
    )
    watcher.start()
    return _refetch(db, job_id) or job


def _spawn_process(job_id: str, spec: JobSpec) -> tuple[subprocess.Popen | None, str | None]:
    """Launch the CLI script. Returns (popen, error_message)."""
    if not CLI_SCRIPT.exists():
        return None, f"CLI script missing: {CLI_SCRIPT}"
    cmd: list[str] = [sys.executable, "-u", str(CLI_SCRIPT)]
    if spec.mode == "login":
        cmd.append("--login")
    if spec.pages:
        cmd.extend(["--pages", ",".join(spec.pages)])
    if spec.slate_date:
        cmd.extend(["--date", spec.slate_date])
    cmd.extend(["--headless", "true" if spec.headless else "false"])
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        popen = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=1,  # line-buffered
        )
    except OSError as exc:
        return None, f"Failed to spawn CLI: {exc}"
    return popen, None


def _watch(job_id: str, popen: subprocess.Popen, timeout_seconds: int) -> None:
    """Read child stdout into the job row, enforce the timeout, finalize.

    Runs in a daemon thread with its own DB session — the request that
    created the job has already returned.
    """
    started = time.time()
    log_buffer = bytearray()
    killed_due_to_timeout = False
    try:
        assert popen.stdout is not None
        while True:
            line = popen.stdout.readline()
            if not line:
                if popen.poll() is not None:
                    break
                # No output and not yet exited — sleep briefly so we don't
                # hot-loop on an idle child.
                if (time.time() - started) > timeout_seconds:
                    killed_due_to_timeout = True
                    _kill_tree(popen)
                    break
                time.sleep(0.2)
                continue
            log_buffer.extend(line)
            # Cap to a tail so an over-chatty run can't grow without bound.
            if len(log_buffer) > LOG_TAIL_CAP_BYTES:
                del log_buffer[: len(log_buffer) - LOG_TAIL_CAP_BYTES]
            # Periodically flush logs to the DB so the dashboard can poll
            # an in-progress run. Cheaper than per-line writes.
            if int(time.time() - started) % 2 == 0:
                _append_logs(job_id, log_buffer.decode("utf-8", errors="replace"))
            if (time.time() - started) > timeout_seconds:
                killed_due_to_timeout = True
                _kill_tree(popen)
                break
        try:
            popen.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _kill_tree(popen)
        return_code = popen.returncode
        final_log = log_buffer.decode("utf-8", errors="replace")
        error_msg: str | None = None
        status = "success"
        if killed_due_to_timeout:
            status = "failed"
            error_msg = f"Timed out after {timeout_seconds}s."
        elif return_code is None or return_code != 0:
            status = "failed"
            error_msg = f"CLI exited with code {return_code}."
        # Heuristic session-expired detection. The CLI prints
        # ``login_required`` per page when redirected to the sign-in page;
        # surface it as a dedicated error so the dashboard can show the
        # right CTA instead of a generic failure.
        if "login_required" in final_log.lower():
            status = "failed"
            error_msg = (
                "BallparkPal session expired. Run 'Launch Login Browser' "
                "to re-authenticate."
            )
        _finalize(
            job_id,
            status=status,
            return_code=return_code,
            logs=final_log,
            error_message=error_msg,
        )
    except Exception as exc:  # noqa: BLE001 — watcher must never crash silently
        logger.exception("BallparkPal watcher crashed for %s", job_id)
        _finalize(
            job_id,
            status="failed",
            return_code=None,
            logs=log_buffer.decode("utf-8", errors="replace"),
            error_message=f"Watcher crashed: {exc}",
        )
    finally:
        _ACTIVE_HANDLES.pop(job_id, None)


def _kill_tree(popen: subprocess.Popen) -> None:
    """Best-effort kill. On Windows we ``taskkill /T`` so Playwright's
    chromium child also dies; otherwise the browser would orphan.
    """
    if popen.poll() is not None:
        return
    if sys.platform.startswith("win"):
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(popen.pid)],
                check=False, capture_output=True, timeout=10,
            )
            return
        except Exception:  # noqa: BLE001
            pass
    try:
        popen.kill()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# DB write helpers — each opens its own session so the watcher thread
# never reuses the request-scoped session that spawned the job.
# ---------------------------------------------------------------------------


def _mark_running(job_id: str, *, pid: int) -> None:
    with _session_scope() as db:
        job = db.scalar(select(BallparkPalJob).where(BallparkPalJob.job_id == job_id))
        if job is None:
            return
        job.status = "running"
        job.pid = pid


def _append_logs(job_id: str, tail: str) -> None:
    with _session_scope() as db:
        job = db.scalar(select(BallparkPalJob).where(BallparkPalJob.job_id == job_id))
        if job is None:
            return
        job.logs = tail[-LOG_TAIL_CAP_BYTES:]


def _finalize(
    job_id: str,
    *,
    status: str,
    return_code: int | None,
    logs: str | None = None,
    error_message: str | None = None,
) -> None:
    with _session_scope() as db:
        job = db.scalar(select(BallparkPalJob).where(BallparkPalJob.job_id == job_id))
        if job is None:
            return
        job.status = status
        job.return_code = return_code
        job.finished_at = datetime.utcnow()
        if logs is not None:
            job.logs = logs[-LOG_TAIL_CAP_BYTES:]
        if error_message is not None:
            job.error_message = error_message


def _refetch(db: Session, job_id: str) -> BallparkPalJob | None:
    db.expire_all()
    return db.scalar(select(BallparkPalJob).where(BallparkPalJob.job_id == job_id))


class _session_scope:
    """Context manager that yields a fresh session and commits on exit."""

    def __enter__(self) -> Session:
        # Dereference via the module attribute so the test conftest's
        # SessionLocal swap takes effect — a top-level import would have
        # captured the production binding at import time.
        self.db = _db_module.SessionLocal()
        return self.db

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is None:
                self.db.commit()
            else:
                self.db.rollback()
        finally:
            self.db.close()


# ---------------------------------------------------------------------------
# Serialization for the API
# ---------------------------------------------------------------------------


def job_payload(job: BallparkPalJob | None) -> dict[str, Any]:
    if job is None:
        return {"job_id": None, "status": "missing"}
    return {
        "job_id": job.job_id,
        "mode": job.mode,
        "status": job.status,
        "pages": list(job.pages or []),
        "slate_date": job.slate_date,
        "headless": job.headless,
        "pid": job.pid,
        "return_code": job.return_code,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "duration_seconds": _duration(job),
        "logs": job.logs or "",
        "error_message": job.error_message,
        "active": job.status in ACTIVE_STATUSES,
    }


def _duration(job: BallparkPalJob) -> float | None:
    if job.started_at is None:
        return None
    end = job.finished_at or datetime.utcnow()
    start = job.started_at.replace(tzinfo=None) if job.started_at.tzinfo else job.started_at
    end = end.replace(tzinfo=None) if end.tzinfo else end
    return round((end - start).total_seconds(), 2)
