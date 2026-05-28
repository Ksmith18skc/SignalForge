"""Read/write helpers around the ``ballparkpal_snapshots`` table.

The dashboard and API import from here only — they must never touch the
provider/Playwright layer. Each writer upserts on (page, slate_date) so a
nightly re-run replaces the previous day's snapshot in place instead of
piling up history rows.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import BallparkPalSnapshot


PAGE_LABELS = {
    "positive_ev": "Positive EV",
    "strikeouts": "Strikeout Center",
    "hr_zone": "Home Run Zone",
    "hits": "Hits",
    "game_sims": "Game Simulations",
}


def upsert_snapshot(
    db: Session,
    *,
    page: str,
    slate_date: str | None,
    source_url: str,
    parsed: dict[str, Any] | None,
    raw_html_path: str | None,
    last_updated_text: str | None,
    status: str,
    error_message: str | None = None,
) -> BallparkPalSnapshot:
    """Insert or replace the snapshot for (page, slate_date).

    ``parsed`` is stored verbatim; ``row_count`` is derived from
    ``parsed["rows"]`` if it's a list. Both are tolerated to be missing
    so error-path snapshots (status != ok) can still be persisted with
    enough context to diagnose later.
    """
    rows = (parsed or {}).get("rows") if isinstance(parsed, dict) else None
    row_count = len(rows) if isinstance(rows, list) else 0
    existing = db.scalar(
        select(BallparkPalSnapshot)
        .where(BallparkPalSnapshot.page == page)
        .where(BallparkPalSnapshot.slate_date == slate_date)
    )
    if existing is None:
        existing = BallparkPalSnapshot(page=page, slate_date=slate_date, source_url=source_url)
        db.add(existing)
    existing.source_url = source_url
    existing.parsed_json = parsed or {}
    existing.raw_html_path = raw_html_path
    existing.last_updated_text = last_updated_text
    existing.status = status
    existing.error_message = error_message
    existing.row_count = row_count
    existing.fetched_at = datetime.utcnow()
    db.flush()
    return existing


def latest_snapshot(
    db: Session,
    *,
    page: str,
    slate_date: str | None = None,
) -> BallparkPalSnapshot | None:
    """Return the most recent snapshot for ``page``. If ``slate_date`` is
    provided we require an exact match; otherwise we return whichever row
    has the newest ``fetched_at``.
    """
    query = select(BallparkPalSnapshot).where(BallparkPalSnapshot.page == page)
    if slate_date is not None:
        query = query.where(BallparkPalSnapshot.slate_date == slate_date)
    query = query.order_by(BallparkPalSnapshot.fetched_at.desc()).limit(1)
    return db.scalar(query)


def all_latest_snapshots(db: Session, *, slate_date: str | None = None) -> list[BallparkPalSnapshot]:
    """One row per page — the newest snapshot for each. Useful for the
    overview tab where we want to show "what's cached right now."
    """
    out: list[BallparkPalSnapshot] = []
    for page in PAGE_LABELS:
        snap = latest_snapshot(db, page=page, slate_date=slate_date)
        if snap is not None:
            out.append(snap)
    return out


def snapshot_payload(snap: BallparkPalSnapshot | None) -> dict[str, Any]:
    """Serialize a snapshot for API/dashboard consumption. Missing snapshot
    → an empty payload with ``status="missing"`` so the dashboard renders
    an honest "no data yet" state instead of failing.
    """
    if snap is None:
        return {
            "page": None,
            "status": "missing",
            "rows": [],
            "row_count": 0,
            "fetched_at": None,
            "slate_date": None,
            "last_updated_text": None,
            "source_url": None,
            "error_message": None,
            "stale": True,
            "source": None,
            "filename": None,
            "uploaded_at": None,
            "meta": {},
            "warnings": [],
        }
    parsed = snap.parsed_json or {}
    rows = parsed.get("rows") if isinstance(parsed, dict) else parsed
    meta = parsed.get("meta") if isinstance(parsed, dict) else {}
    warnings = parsed.get("warnings") if isinstance(parsed, dict) else []
    meta = meta or {}
    # `source` is set by the manual-CSV upload path (ballparkpal_csv.py).
    # Anything else is assumed to be the Playwright scraper. Surface it as
    # a top-level field so the dashboard cache table can show it without
    # having to dig through nested meta.
    source = meta.get("source") or "playwright"
    return {
        "page": snap.page,
        "status": snap.status,
        "rows": rows or [],
        "row_count": snap.row_count or (len(rows) if isinstance(rows, list) else 0),
        "fetched_at": snap.fetched_at.isoformat() if snap.fetched_at else None,
        "slate_date": snap.slate_date,
        "last_updated_text": snap.last_updated_text,
        "source_url": snap.source_url,
        "raw_html_path": snap.raw_html_path,
        "error_message": snap.error_message,
        "stale": _is_stale(snap.fetched_at),
        "source": source,
        "filename": meta.get("filename"),
        "uploaded_at": meta.get("uploaded_at"),
        "meta": meta,
        "warnings": warnings or [],
    }


def _is_stale(fetched_at: datetime | None, *, max_age_hours: int = 24) -> bool:
    """A snapshot is stale if it was fetched more than ``max_age_hours``
    ago. The dashboard surfaces this as a warning banner.
    """
    if fetched_at is None:
        return True
    delta = datetime.utcnow() - (
        fetched_at.replace(tzinfo=None) if fetched_at.tzinfo else fetched_at
    )
    return delta > timedelta(hours=max_age_hours)
