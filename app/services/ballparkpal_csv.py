"""Parse a manually-exported BallparkPal CSV into a snapshot payload.

The CSV path is the **primary** ingestion route. Playwright is optional
and runs only when an operator explicitly opts in from the Advanced
expander — too brittle to be the default on Render (the headed browser
never even appears in a hosted container, which is why the legacy
"Launch Login Browser" flow stuck in RUNNING_LOGIN for hours).

Design notes:

* We do **not** rewrite the existing per-page parsers. Each parser
  already knows how to extract the canonical row dict from any table
  with appropriate header names. The cheapest way to reuse that work
  is to build a synthetic ``<table>`` from the CSV and call
  ``parse_page`` on it. Same column-aliasing, same team-name
  normalization, same warnings shape — zero duplication.

* CSV columns BallparkPal exports do not always match the page's
  on-screen labels exactly (e.g. the Strikeout Center export has an
  ``IP`` column that the HTML calls ``Inn``). The parser's ``_pick``
  fallback list already covers these aliases, so a synthetic table
  built from the CSV resolves cleanly.

* Validation happens **after** parsing. We compare parsed row keys
  against a small expected-column set and report missing fields as
  warnings so the dashboard can show "required columns missing" without
  refusing the upload outright. Partial data is more useful than
  rejection.
"""

from __future__ import annotations

import csv
import html
import io
import logging
from datetime import datetime, timezone
from typing import Any

from app.providers.ballparkpal import parse_page
from app.services.ballparkpal_cache import PAGE_LABELS

logger = logging.getLogger(__name__)

# Per-page "did the parsed output look right?" check. Each entry is a
# set of keys at least one parsed row should populate; missing all of
# them gets surfaced as a validation warning.
EXPECTED_PARSED_KEYS: dict[str, set[str]] = {
    "positive_ev": {"player", "market", "line", "odds"},
    "strikeouts": {"pitcher", "projected_k", "over_line"},
    "hr_zone": {"park", "total_projected_hrs", "player", "hr_probability"},
    "hits": {"player", "projected_hits", "line"},
    "game_sims": {"away_team", "home_team", "projected_total", "projected_runs_away"},
}


class CsvParseError(ValueError):
    """Raised for unrecoverable CSV problems (empty file, no header row).

    Distinct from "parsed but looks wrong" — those become warnings on
    the snapshot rather than aborting the upload.
    """


def _decode_bytes(blob: bytes) -> str:
    """Decode a CSV file as text. BallparkPal serves UTF-8 with an
    occasional BOM; some operator browsers re-encode to Latin-1 when
    saving. Try UTF-8 with BOM-stripping first, then fall back."""
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return blob.decode(encoding)
        except UnicodeDecodeError:
            continue
    return blob.decode("utf-8", errors="replace")


def parse_csv_text(csv_text: str) -> tuple[list[str], list[dict[str, str]]]:
    """Parse raw CSV text into (header, rows).

    Sniffs the dialect so tab-separated paste-from-spreadsheet exports
    work the same as comma-separated downloads. We strip cells but
    keep empty strings — downstream ``_to_float`` already treats empty
    as None, and dropping them here would prevent the parser from
    knowing which column was empty.
    """
    sample = csv_text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
    except csv.Error:
        dialect = csv.excel
    reader = csv.reader(io.StringIO(csv_text), dialect=dialect)
    rows = [r for r in reader if any((c or "").strip() for c in r)]
    if not rows:
        raise CsvParseError("CSV file is empty or has no non-blank rows.")
    header = [(c or "").strip() for c in rows[0]]
    if not any(header):
        raise CsvParseError("CSV file has no header row.")
    data_rows: list[dict[str, str]] = []
    for raw_row in rows[1:]:
        record: dict[str, str] = {}
        for idx, cell in enumerate(raw_row):
            key = header[idx] if idx < len(header) and header[idx] else f"col_{idx}"
            record[key] = (cell or "").strip()
        data_rows.append(record)
    return header, data_rows


def _build_synthetic_html(header: list[str], rows: list[dict[str, str]]) -> str:
    """Wrap the CSV rows in a minimal ``<table>`` so ``parse_page`` can
    chew on it. Cheaper than re-implementing every parser in CSV form.
    """
    parts: list[str] = ["<html><body><table>", "<tr>"]
    for col in header:
        parts.append(f"<th>{html.escape(col)}</th>")
    parts.append("</tr>")
    for row in rows:
        parts.append("<tr>")
        for col in header:
            value = row.get(col, "")
            parts.append(f"<td>{html.escape(str(value) if value is not None else '')}</td>")
        parts.append("</tr>")
    parts.append("</table></body></html>")
    return "".join(parts)


def _validate_parsed_rows(page: str, parsed_rows: list[dict[str, Any]]) -> list[str]:
    """Return a list of human-readable warnings about parsed output.

    Empty list = nothing to flag. Non-empty = surface in the dashboard
    so the operator can see "did the column mapping survive my CSV?"
    """
    warnings: list[str] = []
    expected = EXPECTED_PARSED_KEYS.get(page) or set()
    if not parsed_rows:
        warnings.append(
            "No data rows parsed from CSV. Check the page mapping — the file "
            "may have been exported from a different BallparkPal page."
        )
        return warnings
    if not expected:
        return warnings
    populated_keys: set[str] = set()
    for row in parsed_rows:
        for key, value in row.items():
            if value not in (None, ""):
                populated_keys.add(key)
    missing = expected - populated_keys
    if len(missing) == len(expected):
        warnings.append(
            f"None of the expected columns for '{page}' were detected "
            f"({sorted(expected)}). The CSV is probably from a different page."
        )
    elif missing:
        warnings.append(
            f"Some expected columns for '{page}' were empty or missing: "
            f"{sorted(missing)}. The snapshot was still saved."
        )
    return warnings


def parse_uploaded_csv(
    *,
    page: str,
    csv_bytes: bytes,
    filename: str,
    slate_date: str | None,
) -> dict[str, Any]:
    """Parse an uploaded CSV into the same ``parsed_json`` shape the
    HTML scraper produces, with manual-upload provenance baked into
    ``meta``.

    Returns ``{"page", "slate_date", "header", "raw_row_count",
    "parsed", "warnings"}``. The caller persists ``parsed`` via
    ``upsert_snapshot`` and reads ``warnings`` to surface them on the
    upload preview.
    """
    if page not in PAGE_LABELS:
        raise CsvParseError(
            f"Unknown page '{page}'. Expected one of: {sorted(PAGE_LABELS)}"
        )
    csv_text = _decode_bytes(csv_bytes)
    header, rows = parse_csv_text(csv_text)
    raw_row_count = len(rows)
    synthetic = _build_synthetic_html(header, rows)
    parsed = parse_page(page, synthetic)
    parsed_rows = parsed.get("rows") or []
    validation_warnings = _validate_parsed_rows(page, parsed_rows)
    existing_warnings = parsed.get("warnings") or []
    parsed_warnings = [*existing_warnings, *validation_warnings]
    # Bake upload provenance into meta so the dashboard's cache overview
    # can show "source=manual_csv · file.csv" without a schema change.
    meta = dict(parsed.get("meta") or {})
    meta.update(
        {
            "source": "manual_csv",
            "filename": filename,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "raw_row_count": raw_row_count,
            "parsed_row_count": len(parsed_rows),
            "csv_header": header,
        }
    )
    parsed = {
        **parsed,
        "meta": meta,
        "warnings": parsed_warnings,
    }
    logger.info(
        "ballparkpal_csv parsed page=%s filename=%s raw_rows=%d parsed_rows=%d warnings=%d",
        page, filename, raw_row_count, len(parsed_rows), len(parsed_warnings),
    )
    return {
        "page": page,
        "slate_date": slate_date,
        "header": header,
        "raw_row_count": raw_row_count,
        "parsed_row_count": len(parsed_rows),
        "parsed": parsed,
        "warnings": parsed_warnings,
        "rows_preview": parsed_rows[:10],
    }
