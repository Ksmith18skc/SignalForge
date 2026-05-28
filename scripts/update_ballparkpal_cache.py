"""BallparkPal ingestion CLI.

Scheduled or manual. Loads each requested page with Playwright, parses
it into structured rows, and writes a snapshot row per page to the
``ballparkpal_snapshots`` table. A single page failure does **not** abort
the whole run — every page reports its own status.

Examples:

    # First-time auth: open the browser headed and sign in.
    python scripts/update_ballparkpal_cache.py --login

    # Nightly ingestion: headless, all pages.
    python scripts/update_ballparkpal_cache.py \\
        --pages positive_ev,strikeouts,hr_zone,hits,game_sims \\
        --date today

    # Manual refresh of a single page in headed mode for debugging.
    python scripts/update_ballparkpal_cache.py \\
        --pages game_sims --headless false --force-refresh
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

from app.db import SessionLocal, init_db
from app.providers.ballparkpal import (
    PAGES,
    BallparkPalFetcher,
    DEFAULT_HTML_DIR,
    DEFAULT_PROFILE_DIR,
    parse_page,
    extract_last_updated,
)
from app.services.ballparkpal_cache import upsert_snapshot
from app.services.card_date import TZ_ARIZONA

logger = logging.getLogger("ballparkpal.update")


def _arizona_today() -> str:
    return datetime.now(TZ_ARIZONA).date().isoformat()


def _parse_date(value: str | None) -> str | None:
    """Accept ``today``, ``yesterday``, or an ISO YYYY-MM-DD date."""
    if not value:
        return None
    lowered = value.strip().lower()
    if lowered == "today":
        return _arizona_today()
    if lowered == "yesterday":
        from datetime import timedelta

        today = datetime.now(TZ_ARIZONA).date()
        return (today - timedelta(days=1)).isoformat()
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise SystemExit(f"--date must be YYYY-MM-DD, 'today', or 'yesterday' (got {value!r})") from exc
    return value


def _parse_pages(value: str | None) -> list[str]:
    if not value:
        return list(PAGES.keys())
    requested = [p.strip() for p in value.split(",") if p.strip()]
    unknown = [p for p in requested if p not in PAGES]
    if unknown:
        raise SystemExit(f"Unknown pages: {unknown}. Available: {list(PAGES)}")
    return requested


def _str_to_bool(value: str | None) -> bool:
    if value is None:
        return True
    return value.strip().lower() in {"1", "true", "t", "yes", "y"}


def run(
    *,
    pages: Iterable[str],
    slate_date: str | None,
    headless: bool,
    profile_dir: Path,
    html_dir: Path | None,
    login: bool,
    force_refresh: bool,
) -> dict[str, dict[str, object]]:
    """Drive the fetcher across ``pages``. Returns per-page status dict.

    ``force_refresh`` is currently a passthrough flag for symmetry with
    the CLI — we always re-fetch, but the flag is plumbed through so a
    future "skip if fetched < N hours ago" optimization has a place to
    land without a CLI change.
    """
    init_db()
    results: dict[str, dict[str, object]] = {}
    db = SessionLocal()
    try:
        with BallparkPalFetcher(
            profile_dir=profile_dir,
            headless=headless,
            html_dir=html_dir,
        ) as fetcher:
            if login:
                fetcher.login_flow()
            for page in pages:
                spec = PAGES.get(page)
                if spec is None:
                    results[page] = {"status": "error", "error": "Unknown page."}
                    continue
                fetch = fetcher.fetch(
                    page_name=page,
                    url=spec["url"],
                    slate_date=slate_date,
                    wait_selector=spec.get("wait_selector"),
                )
                parsed = parse_page(page, fetch.html) if fetch.html else {
                    "rows": [], "meta": {}, "warnings": [fetch.error or "no html"],
                }
                last_updated = (parsed.get("meta") or {}).get("last_updated") or extract_last_updated(fetch.html)
                snap = upsert_snapshot(
                    db,
                    page=page,
                    slate_date=slate_date,
                    source_url=fetch.url or spec["url"],
                    parsed=parsed,
                    raw_html_path=fetch.raw_html_path,
                    last_updated_text=last_updated,
                    status=fetch.status,
                    error_message=fetch.error,
                )
                db.commit()
                results[page] = {
                    "status": fetch.status,
                    "rows": snap.row_count,
                    "error": fetch.error,
                    "raw_html_path": fetch.raw_html_path,
                    "last_updated_text": last_updated,
                }
                logger.info(
                    "ballparkpal %s -> %s (rows=%d, err=%s)",
                    page, fetch.status, snap.row_count, fetch.error,
                )
    finally:
        db.close()
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Update the BallparkPal cache.")
    parser.add_argument("--login", action="store_true", help="Open the browser headed to sign in.")
    parser.add_argument("--date", default=None, help="Slate date (YYYY-MM-DD | today | yesterday)")
    parser.add_argument(
        "--pages",
        default=None,
        help=(
            "Comma-separated page names. Defaults to all. "
            f"Available: {','.join(PAGES)}"
        ),
    )
    parser.add_argument(
        "--headless",
        default="true",
        help="true/false. Defaults to true (set false to watch the run).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Reserved; today every run is a refresh.",
    )
    parser.add_argument(
        "--profile-dir",
        default=str(DEFAULT_PROFILE_DIR),
        help=f"Persistent profile dir. Default: {DEFAULT_PROFILE_DIR}",
    )
    parser.add_argument(
        "--html-dir",
        default=str(DEFAULT_HTML_DIR),
        help=f"Raw-HTML dump dir for debugging. Default: {DEFAULT_HTML_DIR}",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    slate_date = _parse_date(args.date)
    pages = _parse_pages(args.pages)
    headless = _str_to_bool(args.headless)
    if args.login:
        # Headless + login doesn't make sense — the user needs to see the form.
        headless = False
    html_dir = Path(args.html_dir) if args.html_dir else None
    profile_dir = Path(args.profile_dir)

    try:
        results = run(
            pages=pages,
            slate_date=slate_date,
            headless=headless,
            profile_dir=profile_dir,
            html_dir=html_dir,
            login=args.login,
            force_refresh=args.force_refresh,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("BallparkPal ingestion failed: %s", exc)
        return 1

    # Summary line for the operator. Any login_required / error status
    # makes the run "imperfect" but still exits 0 — partial data is more
    # useful than no data, and the caller can grep this output.
    for page, info in results.items():
        print(f"{page}: status={info.get('status')}, rows={info.get('rows', 0)}, "
              f"err={info.get('error') or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
