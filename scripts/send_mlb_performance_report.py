"""Send or print an MLB performance report."""

from __future__ import annotations

import argparse
import logging

import httpx

from app.config import get_settings
from app.db import SessionLocal, init_db
from app.services.mlb_performance import (
    clv_report,
    performance_by_market,
    performance_by_score_band,
    performance_summary,
    top_factors_by_performance,
)

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate MLB performance report")
    parser.add_argument("--discord", action="store_true", help="Send report to configured Discord webhook")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level)

    init_db()
    db = SessionLocal()
    try:
        message = build_report(db)
    finally:
        db.close()

    print(message)
    if args.discord:
        ok, err = send_discord(message)
        if ok:
            logger.info("Discord MLB performance report sent")
        else:
            logger.warning("Discord MLB performance report skipped/failed: %s", err)


def build_report(db) -> str:  # noqa: ANN001
    summary = performance_summary(db)
    by_market = performance_by_market(db)
    by_band = performance_by_score_band(db)
    clv = clv_report(db)
    factors = top_factors_by_performance(db)[:5]
    lines = [
        "MLB Performance Report",
        "",
        f"Graded edges: {summary.get('graded_edges', 0)}",
        f"W-L-P: {summary.get('wins', 0)}-{summary.get('losses', 0)}-{summary.get('pushes', 0)}",
        f"Win rate: {_pct(summary.get('win_rate'))}",
        f"ROI units: {summary.get('roi_units', 0):.2f}",
        f"Average CLV: {_pct(clv.get('average_clv_percent'))}",
        "",
        "By edge type:",
        *[
            f"- {row['edge_type']}: ROI {row.get('roi_units', 0):.2f}, win rate {_pct(row.get('win_rate'))}, avg CLV {_pct(row.get('average_clv_percent'))}"
            for row in by_market
        ],
        "",
        "By score band:",
        *[
            f"- {row['score_band']}: ROI {row.get('roi_units', 0):.2f}, win rate {_pct(row.get('win_rate'))}, sample {row.get('graded_edges', 0)}"
            for row in by_band
        ],
        "",
        "Best/worst factors:",
        *[
            f"- {row['factor']}: performance {row.get('performance_score', 0):.3f}, sample {row.get('sample', 0)}"
            for row in factors
        ],
    ]
    return "\n".join(lines)[:1900]


def send_discord(message: str) -> tuple[bool, str | None]:
    webhook = get_settings().discord_webhook_url
    if not webhook:
        return False, "SIGNALFORGE_DISCORD_WEBHOOK_URL is not configured"
    try:
        response = httpx.post(webhook, json={"content": message}, timeout=10)
        if response.status_code >= 400:
            return False, f"Discord HTTP {response.status_code}: {response.text[:200]}"
        return True, None
    except httpx.HTTPError as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _pct(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "n/a"


if __name__ == "__main__":
    main()
