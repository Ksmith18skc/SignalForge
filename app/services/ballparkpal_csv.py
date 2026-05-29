"""Parse a manually-exported BallparkPal CSV into a snapshot payload.

The CSV path is the **primary** ingestion route. Playwright is optional
and runs only when an operator explicitly opts in from the Advanced
expander — too brittle to be the default on Render (the headed browser
never even appears in a hosted container, which is why the legacy
"Launch Login Browser" flow stuck in RUNNING_LOGIN for hours).

Why this parser is the way it is
--------------------------------

Real exported CSVs are messy:

* The table-scraper browser extensions some operators use prepend a
  junk row of column ids (``"tablescraper-selected-row"``,
  ``"tablescraper-selected-row 2"``, …) so the *real* header row is
  row index 1, not 0. We detect the header row by scoring each
  candidate against a set of known BPP column tokens — the row with
  the most matches wins.

* BallparkPal column names differ between the on-screen labels, the
  CSV export, and a couple of scraper tools. We pre-rename CSV
  headers through a per-page alias map so the existing
  ``parse_strikeout_center`` / ``parse_positive_ev`` / etc. functions
  can keep their narrow expected-header sets.

* Even with the right headers, a row might still be rejected by the
  strict parser (e.g. a "pitcher" column that's empty). We compute
  per-row rejection reasons so the dashboard can show "row 7 skipped:
  missing pitcher/player name" instead of a silent 0-row result.

* If the strict parser still yields 0 rows after all that, we fall
  back to a **generic** parser that just keeps every non-blank row
  as a dict keyed by the detected headers. The MLB edge scan won't
  know what to do with generic rows, but at least the dashboard's
  cache overview shows non-zero rows and the operator can see what
  they uploaded.
"""

from __future__ import annotations

import csv
import html
import io
import logging
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any

from app.providers.ballparkpal import parse_page
from app.services.ballparkpal_cache import PAGE_LABELS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-page hints
# ---------------------------------------------------------------------------
#
# Each set is the union of every column token we'd expect to see on the
# corresponding BPP page. Used by ``_detect_header_row`` to score
# candidate rows. Aliases (e.g. "ip" and "innings") are included so a
# scraped CSV using either label still scores. Keep these as
# already-normalized tokens (lowercase, separator-stripped) — same shape
# ``_normalize_header_token`` produces.
RAW_HEADER_HINTS: dict[str, set[str]] = {
    "strikeouts": {
        "team", "pitcher", "player", "name", "k", "ks", "projected_k",
        "strikeouts", "opp", "opponent", "inn", "ip", "innings", "bf",
        "batters_faced", "over", "line", "k_line", "bp", "ballparkpal",
        "ballparkpal_odds", "ka", "kalshi", "kalshi_odds", "k_advantage",
    },
    "positive_ev": {
        "tm", "team", "player", "name", "bk", "book", "sportsbook",
        "market", "prop", "o_u", "ou", "over_under", "line", "total",
        "odds", "price", "cs", "consensus", "consensus_odds", "delta",
        "bp", "ballparkpal_odds",
    },
    "hr_zone": {
        "park", "total", "team", "tm", "opp", "opponent", "away",
        "home", "player", "name", "hr", "hrs", "hr_probability",
        "probability", "p_hr", "fair_odds", "projected_hrs",
        "total_projected_hrs",
    },
    "hits": {
        "player", "name", "team", "tm", "opp", "opponent", "line",
        "hits", "projected_hits", "p_hits", "over", "under", "odds",
        "over_odds", "under_odds", "probability", "p_over", "fair_odds",
    },
    "game_sims": {
        "away", "home", "team", "away_team", "home_team", "runs",
        "away_runs", "home_runs", "runs_away", "runs_home", "total",
        "projected_total", "wp_away", "wp_home", "away_wp", "home_wp",
        "win_probability_away", "win_probability_home", "spread",
        "runline", "ml_away", "ml_home", "moneyline_away",
        "moneyline_home",
    },
}


# ---------------------------------------------------------------------------
# Per-page CSV header alias map
# ---------------------------------------------------------------------------
#
# The HTML parsers in ``app/providers/ballparkpal.py`` look up cells via
# narrow ``_pick(raw, "k", "projected_k", "ks")`` lists. CSV exports use
# extra label variants ("Projected K", "Strikeouts", "Kalshi") that the
# HTML parser doesn't know about. Rather than touch the HTML parser
# (which is exercised by snapshot tests), we rewrite CSV headers into
# the canonical form before building the synthetic table. Keys here are
# normalized tokens; values are the canonical token the HTML parser
# expects.
CSV_HEADER_ALIASES: dict[str, dict[str, str]] = {
    "strikeouts": {
        "projected_k": "k",
        "projectedk": "k",
        "proj_k": "k",
        "strikeouts": "k",
        "k_proj": "k",
        "ks": "k",
        "opponent": "opp",
        "innings": "inn",
        "ip": "inn",
        "batters_faced": "bf",
        "batters": "bf",
        "k_line": "over",
        "line": "over",
        "ballparkpal": "bp",
        "ballparkpal_odds": "bp",
        "kalshi": "ka",
        "kalshi_odds": "ka",
        "k_advantage": "ka",
    },
    "positive_ev": {
        "team": "tm",
        "name": "player",
        "book": "bk",
        "sportsbook": "bk",
        "prop": "market",
        "over_under": "o_u",
        "ou": "o_u",
        "total": "line",
        "price": "odds",
        "consensus": "cs",
        "consensus_odds": "cs",
        "ballparkpal_odds": "bp",
    },
    "hr_zone": {
        "name": "player",
        "opponent": "opp",
        "hrs": "projected_hrs",
        "p_hr": "hr_probability",
        "probability": "hr_probability",
        # BPP HR Zone "Hitters" CSV uses short column codes — keep the
        # short forms as-is so the parser's branch detector (which
        # looks for ``prob`` + ``bp``) still triggers, but expose them
        # in the parsed rows under canonical names. The aliasing here
        # is intentionally light: the operator-facing semantics of the
        # scraper-tool export don't perfectly line up with the BPP UI
        # column order, so we keep raw codes available for inspection.
        "ballparkpal_odds": "bp",
        "kalshi_odds": "ka",
    },
    "hits": {
        "name": "player",
        "tm": "team",
        "opponent": "opp",
        "hits": "projected_hits",
        "p_hits": "projected_hits",
        "p_over": "probability_over",
        "probability": "probability_over",
    },
    "game_sims": {
        "team_away": "away",
        "team_home": "home",
        "away_team": "away",
        "home_team": "home",
        "away_runs": "runs_away",
        "home_runs": "runs_home",
        "projected_total": "total",
        "away_wp": "wp_away",
        "home_wp": "wp_home",
        "win_probability_away": "wp_away",
        "win_probability_home": "wp_home",
        "runline": "spread",
        "moneyline_away": "ml_away",
        "moneyline_home": "ml_home",
    },
}


# ---------------------------------------------------------------------------
# Per-page minimum-required check (used both for diagnostics and to
# decide whether a row should have been parsed)
# ---------------------------------------------------------------------------
#
# Each entry is a list of (label, set-of-acceptable-canonical-keys)
# tuples. A row passes only if every label has at least one non-empty
# value among its acceptable keys. The first failing label becomes the
# row's rejection reason.
MIN_REQUIRED: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "strikeouts": [
        ("pitcher/player name", ("pitcher", "player", "name")),
        ("projected strikeouts OR line", ("k", "over")),
    ],
    "positive_ev": [
        ("player name", ("player", "name")),
        ("market/prop", ("market", "prop")),
    ],
    "hr_zone": [
        ("park or player or matchup", ("park", "player", "name", "team", "tm", "away", "home")),
    ],
    "hits": [
        ("player name", ("player", "name")),
        ("projected hits OR line", ("projected_hits", "hits", "p_hits", "line")),
    ],
    "game_sims": [
        ("away/home team or runs/total", ("away", "home", "total", "runs_away", "runs_home")),
    ],
}

# Per-page expected-key check used after strict parsing — kept for
# backwards-compatibility with the older validator.
EXPECTED_PARSED_KEYS: dict[str, set[str]] = {
    "strikeouts": {"pitcher", "projected_k", "over_line"},
    "positive_ev": {"player", "market", "line", "odds"},
    "hr_zone": {"park", "total_projected_hrs", "player", "hr_probability"},
    "hits": {"player", "projected_hits", "line"},
    "game_sims": {"away_team", "home_team", "projected_total", "projected_runs_away"},
}


class CsvParseError(ValueError):
    """Raised for unrecoverable CSV problems (empty file, no header row).

    Distinct from "parsed but looks wrong" — those become warnings on
    the snapshot rather than aborting the upload.
    """


# ---------------------------------------------------------------------------
# Cell + header cleanup
# ---------------------------------------------------------------------------

# Characters BPP and scraper tools leave behind that confuse downstream
# parsing: NBSP, zero-width spaces, BOM. We never want those treated as
# significant content.
_HIDDEN_UNICODE_RE = re.compile(r"[ ​‌‍﻿]")


def _strip_cell(value: Any) -> str:
    """Trim a cell of whitespace, hidden unicode, and stray quoting.

    We don't strip ``%``, ``+``, or commas from values here — the
    downstream ``_to_float`` does that, and stripping at this layer
    would also clobber legitimate header tokens that happen to contain
    those characters.
    """
    if value is None:
        return ""
    text = str(value)
    text = unicodedata.normalize("NFKC", text)
    text = _HIDDEN_UNICODE_RE.sub("", text)
    return text.strip()


def _normalize_header_token(value: Any) -> str:
    """Match ``_normalize_header`` in the HTML parser so detection scores
    align with what the parser will eventually see."""
    text = _strip_cell(value).lower()
    # The HTML parser maps "%" → "pct" and "Δ" → "delta"; we keep those
    # in sync so a CSV header of "Δ%" normalizes the same way the HTML
    # parser would normalize it.
    text = text.replace("%", "pct").replace("Δ", "delta").replace("δ", "delta")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


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


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _read_all_rows(csv_text: str) -> list[list[str]]:
    """Read every non-blank row from the CSV text.

    Sniffs the dialect so tab-separated paste-from-spreadsheet exports
    work the same as comma-separated downloads. Returns rows in their
    original order (header detection happens later) with cells already
    stripped of hidden unicode + whitespace.
    """
    sample = csv_text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
    except csv.Error:
        dialect = csv.excel
    reader = csv.reader(io.StringIO(csv_text), dialect=dialect)
    out: list[list[str]] = []
    for raw in reader:
        cells = [_strip_cell(c) for c in raw]
        if any(cells):
            out.append(cells)
    if not out:
        raise CsvParseError("CSV file is empty or has only blank rows.")
    return out


# ---------------------------------------------------------------------------
# Header row detection
# ---------------------------------------------------------------------------


def _detect_header_row(rows: list[list[str]], page: str) -> tuple[int, int]:
    """Return ``(header_row_index, score)``.

    Scans up to the first 12 non-blank rows looking for the row with
    the most overlap against ``RAW_HEADER_HINTS[page]``. Falls back to
    row 0 if no row scores at least 2 hints (we don't want to "detect"
    a header on a data row that happens to contain a stray "team"
    cell).

    The header tokens used for scoring are the **post-normalization**
    forms (``_normalize_header_token``) so they match the keys stored
    in ``RAW_HEADER_HINTS``.
    """
    hints = RAW_HEADER_HINTS.get(page) or set()
    if not hints:
        return 0, 0
    best_idx = 0
    best_score = 0
    for idx in range(min(len(rows), 12)):
        tokens = {_normalize_header_token(c) for c in rows[idx] if c}
        tokens.discard("")
        score = len(tokens & hints)
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_score < 2:
        # Not confident — keep the first row as headers so we don't
        # accidentally promote a data row to header.
        return 0, best_score
    return best_idx, best_score


def _apply_header_aliases(headers: list[str], page: str) -> list[str]:
    """Rewrite each header into its canonical normalized form using the
    per-page alias map. ``_pick`` in the HTML parser already covers the
    canonical keys, so once we've aliased we can stop worrying about
    label drift.
    """
    aliases = CSV_HEADER_ALIASES.get(page) or {}
    out: list[str] = []
    seen: dict[str, int] = {}
    for raw in headers:
        token = _normalize_header_token(raw)
        canonical = aliases.get(token, token)
        if not canonical:
            out.append("")
            continue
        # De-duplicate collisions — same logic the HTML parser uses for
        # repeated "Δ" columns.
        seen[canonical] = seen.get(canonical, 0) + 1
        out.append(canonical if seen[canonical] == 1 else f"{canonical}_{seen[canonical]}")
    return out


# ---------------------------------------------------------------------------
# Synthetic HTML builder + diagnostics
# ---------------------------------------------------------------------------


def _build_synthetic_html(header: list[str], data_rows: list[list[str]]) -> str:
    """Wrap the CSV rows in a minimal ``<table>`` so ``parse_page`` can
    chew on it. Cheaper than re-implementing every parser in CSV form.
    """
    parts: list[str] = ["<html><body><table>", "<tr>"]
    for col in header:
        parts.append(f"<th>{html.escape(col)}</th>")
    parts.append("</tr>")
    for row in data_rows:
        parts.append("<tr>")
        for idx in range(len(header)):
            value = row[idx] if idx < len(row) else ""
            parts.append(f"<td>{html.escape(value)}</td>")
        parts.append("</tr>")
    parts.append("</table></body></html>")
    return "".join(parts)


def _row_to_dict(header: list[str], row: list[str]) -> dict[str, str]:
    """Pair a data row up against the canonical headers for diagnostics."""
    out: dict[str, str] = {}
    for idx, h in enumerate(header):
        if not h:
            continue
        value = row[idx] if idx < len(row) else ""
        out[h] = value
    return out


def _diagnose_rejected_rows(
    page: str,
    canonical_headers: list[str],
    data_rows: list[list[str]],
    parsed_count: int,
) -> list[dict[str, Any]]:
    """For each data row that doesn't satisfy the per-page minimum
    requirements, return a structured reason.

    We don't try to mirror the HTML parser's internal logic exactly;
    instead we report against ``MIN_REQUIRED`` which captures the
    actual "row was dropped" conditions the parsers enforce
    (e.g. ``if not pitcher: continue`` in ``parse_strikeout_center``).
    Rows that *would* satisfy the minimum but still ended up parsed
    away are surfaced separately as a count delta.
    """
    rules = MIN_REQUIRED.get(page) or []
    if not rules:
        return []
    reasons: list[dict[str, Any]] = []
    for row_idx, raw_row in enumerate(data_rows):
        row_dict = _row_to_dict(canonical_headers, raw_row)
        failing: list[str] = []
        for label, candidate_keys in rules:
            if not any(row_dict.get(k) for k in candidate_keys):
                failing.append(label)
        if failing:
            sample = {k: v for k, v in row_dict.items() if v}
            # Trim the sample so giant rows don't bloat the response.
            sample_short = dict(list(sample.items())[:6])
            reasons.append(
                {
                    "row_index": row_idx,
                    "missing": failing,
                    "reason": "Missing required field(s): " + ", ".join(failing),
                    "sample": sample_short,
                }
            )
    return reasons


def _generic_fallback_rows(
    canonical_headers: list[str],
    data_rows: list[list[str]],
) -> list[dict[str, Any]]:
    """Build a list-of-dicts using canonical headers as keys.

    Used when the strict parser returns zero rows — better to surface
    *something* than to show an empty snapshot. Empty cells are dropped
    so the dashboard table doesn't show a wall of blank columns; the
    raw row count is preserved on the snapshot meta.
    """
    out: list[dict[str, Any]] = []
    for row in data_rows:
        record: dict[str, Any] = {}
        for idx, h in enumerate(canonical_headers):
            if not h:
                continue
            value = row[idx] if idx < len(row) else ""
            if value:
                record[h] = value
        if record:
            out.append(record)
    return out


def _validate_parsed_rows(page: str, parsed_rows: list[dict[str, Any]]) -> list[str]:
    """Surface "did the column mapping survive?" warnings."""
    warnings: list[str] = []
    expected = EXPECTED_PARSED_KEYS.get(page) or set()
    if not parsed_rows or not expected:
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
            f"({sorted(expected)}). The CSV may be from a different page."
        )
    elif missing:
        warnings.append(
            f"Some expected columns for '{page}' were empty or missing: "
            f"{sorted(missing)}. The snapshot was still saved."
        )
    return warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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

    Returns a dict shaped like ::

        {
            "page": ...,
            "slate_date": ...,
            "raw_headers": [...],          # detected real header row
            "raw_rows_preview": [[...]],    # first 20 rows incl. junk
            "detected_header_row_index": int,
            "canonical_headers": [...],
            "raw_row_count": int,
            "parsed_row_count": int,
            "parsed": {...},                # what gets persisted
            "warnings": [...],
            "rows_preview": [...],          # first 10 parsed rows
            "rejection_reasons": [...],     # per-row why-skipped
            "used_generic_fallback": bool,
            "header_detection_score": int,
        }

    The caller persists ``parsed`` via ``upsert_snapshot`` and surfaces
    the other fields on the upload preview.
    """
    if page not in PAGE_LABELS:
        raise CsvParseError(
            f"Unknown page '{page}'. Expected one of: {sorted(PAGE_LABELS)}"
        )
    csv_text = _decode_bytes(csv_bytes)
    rows = _read_all_rows(csv_text)
    raw_rows_preview = [list(r) for r in rows[:20]]
    header_idx, score = _detect_header_row(rows, page)
    raw_headers = list(rows[header_idx])
    data_rows = rows[header_idx + 1 :]
    canonical_headers = _apply_header_aliases(raw_headers, page)

    synthetic = _build_synthetic_html(canonical_headers, data_rows)
    parsed = parse_page(page, synthetic)
    parsed_rows = parsed.get("rows") or []

    rejection_reasons = _diagnose_rejected_rows(
        page, canonical_headers, data_rows, parsed_count=len(parsed_rows),
    )

    used_generic_fallback = False
    fallback_rows: list[dict[str, Any]] = []
    if not parsed_rows and data_rows:
        fallback_rows = _generic_fallback_rows(canonical_headers, data_rows)
        used_generic_fallback = bool(fallback_rows)

    warnings: list[str] = list(parsed.get("warnings") or [])
    warnings.extend(_validate_parsed_rows(page, parsed_rows))
    if header_idx > 0:
        warnings.append(
            f"Detected real header row at CSV index {header_idx} "
            f"(score={score}). Junk rows above it were skipped."
        )
    if used_generic_fallback:
        warnings.append(
            "Strict per-page parser returned 0 rows. Falling back to a "
            "generic row dump so the upload is not silently empty. "
            "Check the rejection reasons below and the column mapping."
        )

    # Build the final parsed payload. When we used the generic
    # fallback, the rows come from the fallback; otherwise from the
    # strict parser.
    if used_generic_fallback:
        out_rows = fallback_rows
        out_meta = dict(parsed.get("meta") or {})
        out_meta["fallback"] = "generic_csv"
    else:
        out_rows = parsed_rows
        out_meta = dict(parsed.get("meta") or {})

    # Page-specific subtable mirrors. Some dashboard tabs read from
    # ``meta.<subkey>`` rather than ``rows`` (Home Run Zone in
    # particular splits into totals / by_game / by_team / hitters). If
    # the strict parser didn't populate the subkey the page tab needs,
    # mirror our rows into it so the upload is actually visible. This
    # was the missing piece for the BPP HR Zone CSV export: header
    # detection + generic fallback both succeeded, but the HR Zone tab
    # only reads ``meta.hitters`` and there were no hitters.
    if page == "hr_zone" and not out_meta.get("hitters"):
        out_meta["hitters"] = out_rows

    out_meta.update(
        {
            "source": "manual_csv",
            "filename": filename,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "raw_row_count": len(data_rows),
            "parsed_row_count": len(out_rows),
            "csv_header": raw_headers,
            "canonical_header": canonical_headers,
            "detected_header_row_index": header_idx,
            "header_detection_score": score,
            "used_generic_fallback": used_generic_fallback,
        }
    )
    persisted = {
        "rows": out_rows,
        "meta": out_meta,
        "warnings": warnings,
    }

    logger.info(
        "ballparkpal_csv page=%s filename=%s header_idx=%d score=%d "
        "data_rows=%d strict_rows=%d generic_rows=%d rejections=%d",
        page, filename, header_idx, score, len(data_rows),
        len(parsed_rows), len(fallback_rows), len(rejection_reasons),
    )
    return {
        "page": page,
        "slate_date": slate_date,
        "raw_headers": raw_headers,
        "raw_rows_preview": raw_rows_preview,
        "detected_header_row_index": header_idx,
        "header_detection_score": score,
        "canonical_headers": canonical_headers,
        "raw_row_count": len(data_rows),
        "parsed_row_count": len(out_rows),
        "parsed": persisted,
        "warnings": warnings,
        "rows_preview": out_rows[:10],
        "rejection_reasons": rejection_reasons,
        "used_generic_fallback": used_generic_fallback,
    }
