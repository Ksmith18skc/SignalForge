"""Tests for the manual-CSV upload parser.

These exercise the real exported CSV the dashboard receives — including
the case where a table-scraper browser extension prepends a junk row of
``tablescraper-selected-row`` ids ahead of the real header.
"""

from __future__ import annotations

from pathlib import Path

import pytest

bs4 = pytest.importorskip("bs4", reason="beautifulsoup4 required for CSV parser tests")

from app.services.ballparkpal_csv import (  # noqa: E402
    CSV_HEADER_ALIASES,
    CsvParseError,
    parse_uploaded_csv,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"
STRIKEOUTS_FIXTURE = FIXTURE_DIR / "ballparkpal_strikeouts_real.csv"


def _load_fixture(name: str) -> bytes:
    """Load a binary fixture from ``tests/fixtures/`` — we use bytes
    rather than text so the parser exercises its own decoding path
    (utf-8-sig / utf-8 / latin-1)."""
    return (FIXTURE_DIR / name).read_bytes()


# ---------------------------------------------------------------------------
# Real Strikeout Center export (with table-scraper junk first row)
# ---------------------------------------------------------------------------


def test_parses_real_strikeouts_csv_with_junk_first_row() -> None:
    """The fixture's first row is the table-scraper extension's
    ``tablescraper-selected-row`` ids — the real header is on row 1.
    Header detection must find row 1 and parse every pitcher row.
    """
    blob = _load_fixture("ballparkpal_strikeouts_real.csv")

    result = parse_uploaded_csv(
        page="strikeouts",
        csv_bytes=blob,
        filename="ballparkpal_strikeouts.csv",
        slate_date="2026-05-28",
    )

    # Header row detected at index 1, not 0.
    assert result["detected_header_row_index"] == 1, (
        f"expected header on row 1, got {result['detected_header_row_index']}"
    )
    # Score should be well above the "not confident" threshold of 2.
    assert result["header_detection_score"] >= 6

    # Real BPP column tokens made it into raw_headers.
    raw = result["raw_headers"]
    for token in ("Team", "Pitcher", "K", "Opp", "Inn", "BF", "Over", "BP", "KA"):
        assert token in raw, f"expected token {token!r} in raw_headers, got {raw}"

    # All 12 pitcher rows must parse — no silent zero-row result.
    assert result["raw_row_count"] == 12
    assert result["parsed_row_count"] == 12, (
        "expected 12 parsed pitcher rows; rejection_reasons="
        f"{result['rejection_reasons']}"
    )
    assert not result["used_generic_fallback"]
    assert not result["rejection_reasons"]


def test_real_strikeouts_csv_first_row_has_expected_fields() -> None:
    """First parsed row is Davis Martin (CHW) — projected_k 6.46, BF
    6.5. Confirms the synthetic-HTML → parse_strikeout_center path is
    actually populating canonical fields, not just emitting empty
    dicts.
    """
    blob = _load_fixture("ballparkpal_strikeouts_real.csv")
    result = parse_uploaded_csv(
        page="strikeouts", csv_bytes=blob,
        filename="x.csv", slate_date=None,
    )

    first = result["rows_preview"][0]
    assert first["team"] == "CHW"
    assert first["pitcher"] == "Davis Martin"
    assert first["projected_k"] == pytest.approx(6.46)
    assert first["batters_faced"] == pytest.approx(6.5)


def test_real_strikeouts_csv_emits_header_detection_warning() -> None:
    """When the real header row isn't index 0 we must surface a
    warning so the operator knows junk rows were skipped — silent
    success on a malformed file would be worse than a hard failure.
    """
    blob = _load_fixture("ballparkpal_strikeouts_real.csv")
    result = parse_uploaded_csv(
        page="strikeouts", csv_bytes=blob,
        filename="x.csv", slate_date=None,
    )
    assert any(
        "Detected real header row" in w for w in result["warnings"]
    ), f"expected header-detection warning, got {result['warnings']}"


def test_real_strikeouts_csv_meta_source_and_provenance() -> None:
    """Manual-upload provenance must land in meta so the snapshot
    overview can distinguish CSV uploads from scraped data.
    """
    blob = _load_fixture("ballparkpal_strikeouts_real.csv")
    result = parse_uploaded_csv(
        page="strikeouts", csv_bytes=blob,
        filename="strikeout_center.csv", slate_date="2026-05-28",
    )

    meta = result["parsed"]["meta"]
    assert meta["source"] == "manual_csv"
    assert meta["filename"] == "strikeout_center.csv"
    assert meta["uploaded_at"]  # ISO timestamp from datetime.now
    assert meta["detected_header_row_index"] == 1
    assert meta["used_generic_fallback"] is False


# ---------------------------------------------------------------------------
# Negative / fallback paths
# ---------------------------------------------------------------------------


def test_empty_csv_raises() -> None:
    with pytest.raises(CsvParseError):
        parse_uploaded_csv(
            page="strikeouts", csv_bytes=b"",
            filename="empty.csv", slate_date=None,
        )


def test_blank_only_csv_raises() -> None:
    with pytest.raises(CsvParseError):
        parse_uploaded_csv(
            page="strikeouts",
            csv_bytes=b",,\n\n,,,\n",
            filename="blanks.csv", slate_date=None,
        )


def test_unknown_page_rejected() -> None:
    with pytest.raises(CsvParseError):
        parse_uploaded_csv(
            page="not_a_real_page",
            csv_bytes=b"Team,Pitcher\nCHW,Foo\n",
            filename="x.csv", slate_date=None,
        )


def test_generic_fallback_when_strict_parser_drops_everything() -> None:
    """If no row passes the strict ``if not pitcher: continue`` gate,
    the parser must fall back to a generic row dump so the upload
    is not silently empty. The rows come back as raw dicts keyed by
    the canonical headers.
    """
    csv = (
        b"Team,SomeRandomCol\n"
        b"CHW,foo\n"
        b"DET,bar\n"
    )
    result = parse_uploaded_csv(
        page="strikeouts", csv_bytes=csv,
        filename="weird.csv", slate_date=None,
    )

    # Strict parser rejects every row (no pitcher column), so we land
    # in the generic-fallback branch.
    assert result["used_generic_fallback"] is True
    assert result["parsed_row_count"] == 2
    assert result["rows_preview"]  # not empty

    # Each fallback row preserves whatever cells were populated.
    first = result["rows_preview"][0]
    assert first.get("team") == "CHW"
    assert any("Strict per-page parser returned 0 rows" in w for w in result["warnings"])
    # Per-row rejection reasons must spell out exactly why each row
    # was dropped by the strict parser.
    assert len(result["rejection_reasons"]) == 2
    assert all(
        "pitcher/player name" in (r.get("reason") or "")
        for r in result["rejection_reasons"]
    )


def test_real_hr_zone_csv_populates_meta_hitters() -> None:
    """The exact failure mode the operator reported: a real BPP
    Home Run Zone export uploaded as ``hr_zone`` left the dashboard's
    HR Zone tab empty because the strict parser had no branch for the
    ``prob`` + ``bp`` + sportsbook-odds column pattern, and the
    generic fallback only populated ``rows`` while the HR Zone tab
    reads from ``meta.hitters``.

    Both halves of the fix are checked here:
      * The strict parser now matches the ``prob``+``bp`` pattern, so
        we end up with ``used_generic_fallback=False``.
      * ``meta.hitters`` is populated, so the HR Zone tab renders.
    """
    blob = _load_fixture("ballparkpal_hr_zone_real.csv")

    result = parse_uploaded_csv(
        page="hr_zone",
        csv_bytes=blob,
        filename="hr-zone.csv",
        slate_date="2026-05-28",
    )

    # Real BPP header on row 2 (row 0 is the table-scraper noise, row 1
    # is the on-page "Actual / Expected / Meatballs" sub-banner).
    assert result["detected_header_row_index"] == 2
    assert result["header_detection_score"] >= 2

    # 14 hitter rows in the fixture, all parsed.
    assert result["raw_row_count"] == 14
    assert result["parsed_row_count"] == 14
    # Strict parser matched the new prob+bp branch — no fallback.
    assert result["used_generic_fallback"] is False

    # The HR Zone dashboard tab reads meta.hitters; this is where the
    # data actually has to land for the tab to render.
    hitters = (result["parsed"]["meta"] or {}).get("hitters") or []
    assert len(hitters) == 14

    # Raw cells are preserved verbatim. We intentionally don't
    # synthesize player/hr_probability from a fixed column because
    # scraper tools shift cells differently — operators read the
    # truth from the raw columns.
    first = hitters[0]
    assert first.get("batter") == "CHW"
    assert first.get("park") == "C. Montgomery"
    assert first.get("pitcher") == "22.0%"
    assert first.get("prob") == "+355"
    assert first.get("bp") == "+350"


def test_alias_map_rewrites_csv_headers_into_canonical_form() -> None:
    """A CSV that uses long labels ("Projected K", "IP", "Kalshi")
    must end up with canonical keys (k, inn, ka) after aliasing — the
    HTML parser only knows those short forms.
    """
    aliases = CSV_HEADER_ALIASES["strikeouts"]
    assert aliases.get("projected_k") == "k"
    assert aliases.get("ip") == "inn"
    assert aliases.get("kalshi") == "ka"

    csv = (
        b"Team,Pitcher,Projected K,IP,Batters Faced,Line,BallparkPal,Kalshi\n"
        b"CHW,Davis Martin,6.46,5.0,25.1,6.5,+175,+146\n"
    )
    result = parse_uploaded_csv(
        page="strikeouts", csv_bytes=csv,
        filename="aliased.csv", slate_date=None,
    )

    assert result["parsed_row_count"] == 1
    assert not result["used_generic_fallback"]
    first = result["rows_preview"][0]
    assert first["pitcher"] == "Davis Martin"
    assert first["projected_k"] == pytest.approx(6.46)
    assert first["batters_faced"] == pytest.approx(25.1)
    # "Line" alias → "over"; the BPP parser stores it as over_line.
    assert first["over_line"] == pytest.approx(6.5)
