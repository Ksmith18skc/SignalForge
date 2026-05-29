"""BallparkPal provider.

Two layers:

1. **Fetcher** — wraps Playwright with a persistent context so the user can
   log in once (headed) and reuse the session headlessly thereafter. The
   web process must never import this layer; only the ingestion script in
   ``scripts/update_ballparkpal_cache.py`` calls into it.

2. **Parsers** — pure HTML → list[dict] functions. They take a raw HTML
   string and return structured rows. They tolerate missing columns,
   reorderings, and renamed headers: each row is built from a header-name
   lookup, not column index.

The parsers are import-safe (no Playwright import at module top). That
keeps unit tests and the API/dashboard process from pulling in heavy
browser deps.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Page registry. Adding a new page = add an entry here + a parser.
# ---------------------------------------------------------------------------

PAGES: dict[str, dict[str, str]] = {
    "positive_ev": {
        "url": "https://www.ballparkpal.com/Positive-EV.php",
        "wait_selector": "table",
    },
    "strikeouts": {
        "url": "https://www.ballparkpal.com/Strikeout-Center.php",
        "wait_selector": "table",
    },
    "hr_zone": {
        "url": "https://www.ballparkpal.com/Home-Run-Zone.php",
        "wait_selector": "table",
    },
    "hits": {
        "url": "https://www.ballparkpal.com/Hits.php",
        "wait_selector": "table",
    },
    "game_sims": {
        "url": "https://www.ballparkpal.com/Game-Simulations.php",
        "wait_selector": "table",
    },
}


# ---------------------------------------------------------------------------
# Login detection. BallparkPal redirects unauthenticated visitors to a
# sign-in page; we check both the URL and the presence of a sign-in form
# in the HTML so we catch both 302-style and SPA-style redirects.
# ---------------------------------------------------------------------------

LOGIN_PATH_TOKENS = ("Login", "Sign-In", "signin", "login")
LOGIN_HTML_HINTS = (
    'type="password"',
    "Sign In",
    "Forgot password",
)


def looks_like_login_page(*, url: str, html: str) -> bool:
    """Return True when the response is the login page rather than the
    requested data page. False positives are acceptable here — we'd
    rather flag the snapshot as ``login_required`` than silently store
    an empty parsed payload.
    """
    lowered_url = (url or "").lower()
    if any(token.lower() in lowered_url for token in LOGIN_PATH_TOKENS):
        return True
    if not html:
        return False
    snippet = html[:8000].lower()
    return all(hint.lower() in snippet for hint in ('type="password"',)) and any(
        hint.lower() in snippet for hint in ("sign in", "log in", "forgot password")
    )


# ---------------------------------------------------------------------------
# Team-abbreviation normalization. BallparkPal mixes "CHW" / "WSX" / "WHI"
# and team-logo alt text. We keep this conservative — anything we don't
# recognize passes through untouched so the parser never silently corrupts
# a row.
# ---------------------------------------------------------------------------

_TEAM_ALIASES = {
    "WSX": "CHW",
    "CWS": "CHW",
    "WHI": "CHW",
    "WAS": "WSH",
    "WSN": "WSH",
    "TBR": "TB",
    "TAM": "TB",
    "KCR": "KC",
    "SDP": "SD",
    "SFG": "SF",
    "AZ": "ARI",
    "CHI": "CHC",
    "NY": "NYY",
    "LA": "LAD",
}


def normalize_team_abbr(value: str | None) -> str | None:
    """Map a team token to a canonical 2-4 char abbreviation, or None."""
    if not value:
        return None
    token = re.sub(r"[^A-Za-z]", "", value).upper()
    if not token:
        return None
    return _TEAM_ALIASES.get(token, token)


# ---------------------------------------------------------------------------
# Pure HTML parsers
# ---------------------------------------------------------------------------

@dataclass
class ParseResult:
    """What every parser returns. ``rows`` is the canonical structured
    output the dashboard renders; ``meta`` carries cross-row context
    (slate date, last-updated banner, narrative blurb)."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"rows": self.rows, "meta": self.meta, "warnings": self.warnings}


def _soup(html: str):
    """BeautifulSoup factory with a defensive parser fallback chain.

    Imports inside the function so unit tests can stub the dependency
    without paying its import cost at module load.
    """
    from bs4 import BeautifulSoup  # type: ignore

    for parser in ("lxml", "html.parser"):
        try:
            return BeautifulSoup(html or "", parser)
        except Exception:  # noqa: BLE001
            continue
    return BeautifulSoup(html or "", "html.parser")


def extract_last_updated(html: str) -> str | None:
    """Pull the ``Last Updated: …`` banner text from any BPP page."""
    if not html:
        return None
    match = re.search(r"Last\s+Updated[:\s]*([^<\n]+)", html, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()[:128]


def _text(node) -> str:
    return re.sub(r"\s+", " ", (node.get_text(" ", strip=True) if node else "")).strip()


# Unicode glyphs BPP uses in column headers. Pre-translit so the regex
# downstream doesn't strip them to "".
_HEADER_TRANSLITERATIONS = {
    "Δ": "delta",
    "δ": "delta",
    "%": "pct",
    "·": "_",
    "—": "_",
    "–": "_",
}


def _normalize_header(text: str) -> str:
    raw = text or ""
    for char, repl in _HEADER_TRANSLITERATIONS.items():
        raw = raw.replace(char, repl)
    return re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")


def extract_table_rows(html: str, *, min_columns: int = 2) -> list[list[dict[str, Any]]]:
    """Yield every table on the page as a list of header→cell dicts.

    Returned as ``[ [row, row, ...], [row, row, ...] ]`` — one inner list
    per ``<table>`` element. Tables without ``<th>`` headers are skipped
    because we have no reliable way to label their columns.
    """
    soup = _soup(html)
    tables: list[list[dict[str, Any]]] = []
    for table in soup.find_all("table"):
        headers: list[str] = []
        header_row = table.find("tr")
        if header_row is None:
            continue
        header_cells = header_row.find_all(["th", "td"])
        raw_headers = [_normalize_header(_text(c)) for c in header_cells]
        # De-duplicate collisions (e.g. two Δ columns become "delta" and
        # "delta_2") so per-row dict keys stay distinct.
        seen: dict[str, int] = {}
        headers = []
        for name in raw_headers:
            base = name or ""
            if not base:
                headers.append("")
                continue
            seen[base] = seen.get(base, 0) + 1
            headers.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
        if len([h for h in headers if h]) < min_columns:
            continue
        rows: list[dict[str, Any]] = []
        for tr in table.find_all("tr")[1:]:
            cells = tr.find_all(["td", "th"])
            if not cells:
                continue
            row: dict[str, Any] = {}
            for idx, cell in enumerate(cells):
                key = headers[idx] if idx < len(headers) and headers[idx] else f"col_{idx}"
                row[key] = _text(cell)
                # Capture team-logo alt text — BPP uses <img alt="HOU"> in
                # many tables instead of cell text.
                img = cell.find("img")
                if img and img.get("alt"):
                    row.setdefault(f"{key}_alt", str(img.get("alt")).strip())
                link = cell.find("a")
                if link and link.get("href"):
                    row.setdefault(f"{key}_href", str(link.get("href")).strip())
            rows.append(row)
        if rows:
            tables.append(rows)
    return tables


def _to_float(value: Any) -> float | None:
    """Lenient float parse: strips %, commas, leading +, and currency."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text in {"-", "--", "—", "N/A", "n/a"}:
        return None
    text = text.replace(",", "").replace("%", "").replace("$", "")
    text = text.lstrip("+")
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    f = _to_float(value)
    return int(f) if f is not None else None


def _pick(row: dict[str, Any], *keys: str) -> Any:
    """Return the first non-empty value from a list of candidate header keys.

    Lets parsers stay resilient when BPP renames a column (e.g.
    ``proj`` → ``projected``).
    """
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


# ---- Positive EV ----------------------------------------------------------

def parse_positive_ev(html: str) -> dict[str, Any]:
    result = ParseResult(meta={"last_updated": extract_last_updated(html)})
    tables = extract_table_rows(html)
    if not tables:
        result.warnings.append("No tables found on Positive EV page.")
        return result.as_dict()
    biggest = max(tables, key=len)
    for raw in biggest:
        team = normalize_team_abbr(_pick(raw, "tm", "team", "tm_alt"))
        player = _pick(raw, "player", "name")
        book = _pick(raw, "bk", "book", "sportsbook", "bk_alt")
        market = _pick(raw, "market", "prop")
        over_under = _pick(raw, "o_u", "ou", "over_under")
        line = _to_float(_pick(raw, "line", "total"))
        odds = _to_float(_pick(raw, "odds", "price"))
        cs = _to_float(_pick(raw, "cs", "consensus", "consensus_odds"))
        delta_consensus = _to_float(_pick(raw, "delta", "sportsbook_edge", "delta_consensus"))
        bp = _to_float(_pick(raw, "bp", "ballparkpal_odds"))
        delta_bp = _to_float(_pick(raw, "delta_2", "ballparkpal_delta"))
        row = {
            "team": team,
            "player": player,
            "book": book,
            "market": market,
            "over_under": over_under,
            "line": line,
            "odds": odds,
            "consensus_odds": cs,
            "consensus_delta": delta_consensus,
            "ballparkpal_odds": bp,
            "ballparkpal_delta": delta_bp,
        }
        # Don't drop rows that are mostly empty — surface them in raw_meta
        # so the dashboard can show counts honestly.
        if any(v not in (None, "") for v in row.values()):
            result.rows.append(row)
    return result.as_dict()


# ---- Strikeout Center -----------------------------------------------------

def parse_strikeout_center(html: str) -> dict[str, Any]:
    result = ParseResult(meta={"last_updated": extract_last_updated(html)})
    tables = extract_table_rows(html)
    if not tables:
        result.warnings.append("No tables found on Strikeout Center page.")
        return result.as_dict()
    biggest = max(tables, key=len)
    for raw in biggest:
        team = normalize_team_abbr(_pick(raw, "team", "tm", "team_alt"))
        pitcher = _pick(raw, "pitcher", "player", "name")
        if not pitcher:
            continue
        projected_k = _to_float(_pick(raw, "k", "projected_k", "ks"))
        opponent = normalize_team_abbr(_pick(raw, "opp", "opp_alt", "opponent"))
        innings = _to_float(_pick(raw, "inn", "innings", "ip"))
        batters_faced = _to_float(_pick(raw, "bf", "batters_faced"))
        over_line = _to_float(_pick(raw, "over", "line", "k_line"))
        bp_odds = _to_float(_pick(raw, "bp", "ballparkpal_odds"))
        ka = _pick(raw, "ka", "k_advantage")
        prob_over = _to_float(_pick(raw, "p_over", "probability", "prob_over"))
        result.rows.append(
            {
                "team": team,
                "pitcher": pitcher,
                "projected_k": projected_k,
                "opponent": opponent,
                "projected_innings": innings,
                "batters_faced": batters_faced,
                "over_line": over_line,
                "over_odds": bp_odds,
                "k_advantage": ka,
                "probability_over": prob_over,
            }
        )
    return result.as_dict()


# ---- Home Run Zone --------------------------------------------------------

def parse_home_run_zone(html: str) -> dict[str, Any]:
    """Parse all four HR Zone tabs that ship in the page HTML.

    BPP renders all sub-tabs server-side (Totals / Game / Team / Hitters)
    so we get every table from a single fetch. Each is keyed by its
    detected header signature.
    """
    result = ParseResult(meta={"last_updated": extract_last_updated(html)})
    tables = extract_table_rows(html)
    if not tables:
        result.warnings.append("No tables found on Home Run Zone page.")
        return result.as_dict()
    totals: list[dict[str, Any]] = []
    by_game: list[dict[str, Any]] = []
    by_team: list[dict[str, Any]] = []
    hitters: list[dict[str, Any]] = []
    for table in tables:
        if not table:
            continue
        first = table[0]
        keys = set(first.keys())
        if "park" in keys and any(k in keys for k in {"total", "total_projected_hrs"}):
            for raw in table:
                totals.append(
                    {
                        "park": _pick(raw, "park"),
                        "total_projected_hrs": _to_float(_pick(raw, "total", "total_projected_hrs")),
                    }
                )
        elif "away" in keys and "home" in keys:
            # The Totals tab also renders an "Away | HRs | Home | HRs" matrix.
            for raw in table:
                by_game.append(
                    {
                        "away_team": normalize_team_abbr(_pick(raw, "away_alt", "away")),
                        "away_projected_hrs": _to_float(_pick(raw, "hrs", "away_hrs")),
                        "home_team": normalize_team_abbr(_pick(raw, "home_alt", "home")),
                        "home_projected_hrs": _to_float(_pick(raw, "hrs_2", "home_hrs")),
                    }
                )
        elif "team" in keys and any(k in keys for k in {"projected_hrs", "hrs"}):
            for raw in table:
                by_team.append(
                    {
                        "team": normalize_team_abbr(_pick(raw, "team_alt", "team")),
                        "projected_hrs": _to_float(_pick(raw, "projected_hrs", "hrs")),
                        "opponent": normalize_team_abbr(_pick(raw, "opp_alt", "opp", "opponent")),
                        "park": _pick(raw, "park"),
                    }
                )
        elif "player" in keys or "name" in keys:
            for raw in table:
                hitters.append(
                    {
                        "player": _pick(raw, "player", "name"),
                        "team": normalize_team_abbr(_pick(raw, "team_alt", "team", "tm")),
                        "opponent": normalize_team_abbr(_pick(raw, "opp_alt", "opp", "opponent")),
                        "hr_probability": _to_float(_pick(raw, "hr_probability", "p_hr", "probability")),
                        "fair_odds": _to_float(_pick(raw, "fair_odds", "odds")),
                    }
                )
        elif "prob" in keys and "bp" in keys:
            # BPP "Home Run Zone — Hitters" CSV export: per-batter rows
            # carrying HR probability + sportsbook odds. Scraper tools
            # vary in how they shift the team-logo cells, so any
            # attempt to synthesize "player" / "hr_probability" from a
            # fixed column would be wrong half the time. Preserve the
            # raw row verbatim — the dashboard's Hitters dataframe then
            # renders every column the operator uploaded, and they can
            # read the actual semantics from the columns themselves.
            for raw in table:
                hitters.append(dict(raw))
    # Canonical default: fall through to whichever subtable carries
    # data so an export with only hitters (not totals) doesn't end up
    # with row_count=0 on the cache overview.
    result.rows = totals or hitters or by_team or by_game
    result.meta.update(
        {
            "totals": totals,
            "by_game": by_game,
            "by_team": by_team,
            "hitters": hitters,
        }
    )
    return result.as_dict()


# ---- Hits -----------------------------------------------------------------

def parse_hits(html: str) -> dict[str, Any]:
    result = ParseResult(meta={"last_updated": extract_last_updated(html)})
    tables = extract_table_rows(html)
    if not tables:
        result.warnings.append("No tables found on Hits page.")
        return result.as_dict()
    biggest = max(tables, key=len)
    for raw in biggest:
        result.rows.append(
            {
                "player": _pick(raw, "player", "name"),
                "team": normalize_team_abbr(_pick(raw, "team_alt", "team", "tm")),
                "opponent": normalize_team_abbr(_pick(raw, "opp_alt", "opp", "opponent")),
                "projected_hits": _to_float(_pick(raw, "projected_hits", "p_hits", "hits")),
                "line": _to_float(_pick(raw, "line")),
                "over_odds": _to_float(_pick(raw, "over", "over_odds")),
                "under_odds": _to_float(_pick(raw, "under", "under_odds")),
                "probability_over": _to_float(_pick(raw, "p_over", "probability")),
                "fair_odds": _to_float(_pick(raw, "fair_odds")),
            }
        )
    return result.as_dict()


# ---- Game Simulations -----------------------------------------------------

def parse_game_simulations(html: str) -> dict[str, Any]:
    result = ParseResult(meta={"last_updated": extract_last_updated(html)})
    tables = extract_table_rows(html)
    if not tables:
        result.warnings.append("No tables found on Game Simulations page.")
        return result.as_dict()
    biggest = max(tables, key=len)
    for raw in biggest:
        away = normalize_team_abbr(_pick(raw, "away_alt", "away", "away_team"))
        home = normalize_team_abbr(_pick(raw, "home_alt", "home", "home_team"))
        runs_away = _to_float(_pick(raw, "runs_away", "away_runs", "away_proj", "away_score"))
        runs_home = _to_float(_pick(raw, "runs_home", "home_runs", "home_proj", "home_score"))
        total = _to_float(_pick(raw, "total", "projected_total", "o_u"))
        if total is None and runs_away is not None and runs_home is not None:
            total = round(runs_away + runs_home, 2)
        wp_away = _to_float(_pick(raw, "wp_away", "away_wp", "win_probability_away", "away_win"))
        wp_home = _to_float(_pick(raw, "wp_home", "home_wp", "win_probability_home", "home_win"))
        spread = _to_float(_pick(raw, "spread", "runline", "rl"))
        moneyline_away = _to_float(_pick(raw, "ml_away", "moneyline_away"))
        moneyline_home = _to_float(_pick(raw, "ml_home", "moneyline_home"))
        row = {
            "away_team": away,
            "home_team": home,
            "projected_runs_away": runs_away,
            "projected_runs_home": runs_home,
            "projected_total": total,
            "win_probability_away": wp_away,
            "win_probability_home": wp_home,
            "spread": spread,
            "moneyline_away": moneyline_away,
            "moneyline_home": moneyline_home,
        }
        if any(v not in (None, "") for v in row.values()):
            result.rows.append(row)
    return result.as_dict()


PARSERS: dict[str, Any] = {
    "positive_ev": parse_positive_ev,
    "strikeouts": parse_strikeout_center,
    "hr_zone": parse_home_run_zone,
    "hits": parse_hits,
    "game_sims": parse_game_simulations,
}


def parse_page(page: str, html: str) -> dict[str, Any]:
    """Dispatch to the right parser. Unknown page → empty result."""
    parser = PARSERS.get(page)
    if parser is None:
        return ParseResult(warnings=[f"Unknown page: {page}"]).as_dict()
    try:
        return parser(html)
    except Exception as exc:  # noqa: BLE001 — parser failures must not crash the run
        logger.exception("BallparkPal parser failed for %s", page)
        return ParseResult(warnings=[f"Parser exception: {exc}"]).as_dict()


# ---------------------------------------------------------------------------
# Playwright fetcher — imported lazily so unit tests don't need the browser.
# ---------------------------------------------------------------------------

DEFAULT_PROFILE_DIR = Path(".cache/ballparkpal_profile")
DEFAULT_HTML_DIR = Path(".cache/ballparkpal_html")


@dataclass
class FetchResult:
    page: str
    url: str
    status: str
    html: str = ""
    error: str | None = None
    raw_html_path: str | None = None


class BallparkPalFetcher:
    """Persistent-context Playwright wrapper. Use as a context manager
    so the browser is always cleaned up, even on exceptions."""

    def __init__(
        self,
        *,
        profile_dir: Path = DEFAULT_PROFILE_DIR,
        headless: bool = True,
        html_dir: Path | None = DEFAULT_HTML_DIR,
        slow_mo_ms: int = 0,
    ) -> None:
        self.profile_dir = Path(profile_dir)
        self.headless = headless
        self.html_dir = Path(html_dir) if html_dir else None
        self.slow_mo_ms = slow_mo_ms
        self._pw = None
        self._context = None

    def __enter__(self) -> "BallparkPalFetcher":
        from playwright.sync_api import sync_playwright  # type: ignore

        self.profile_dir.mkdir(parents=True, exist_ok=True)
        if self.html_dir:
            self.html_dir.mkdir(parents=True, exist_ok=True)
        self._pw = sync_playwright().start()
        self._context = self._pw.chromium.launch_persistent_context(
            user_data_dir=str(self.profile_dir),
            headless=self.headless,
            slow_mo=self.slow_mo_ms,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self._context is not None:
                self._context.close()
        finally:
            if self._pw is not None:
                self._pw.stop()

    def login_flow(self, *, login_url: str = "https://www.ballparkpal.com/Login.php") -> None:
        """Open the login page headed so the user can complete sign-in.

        We don't automate the credential submission — that's brittle and
        invites credential storage we don't want. The persistent context
        captures the resulting cookies once the user clicks through.
        """
        if self._context is None:
            raise RuntimeError("login_flow() must be called inside the context manager.")
        page = self._context.new_page()
        page.goto(login_url, wait_until="domcontentloaded")
        print(
            "\nSign in to BallparkPal in the opened browser. "
            "When the dashboard loads, press <Enter> in this terminal."
        )
        try:
            input()
        except EOFError:
            pass
        page.close()

    def fetch(
        self,
        *,
        page_name: str,
        url: str,
        slate_date: str | None = None,
        wait_selector: str | None = "table",
        wait_timeout_ms: int = 25_000,
        post_load_clicks: Iterable[str] | None = None,
    ) -> FetchResult:
        """Load one page, wait for the data to render, and return its HTML.

        Returns a FetchResult with status ``ok``, ``login_required``, or
        ``error``. We never raise out of here — a single page failure must
        not abort the whole nightly run.
        """
        if self._context is None:
            return FetchResult(page=page_name, url=url, status="error", error="Fetcher not entered.")
        page = self._context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=wait_timeout_ms)
            # Best-effort wait for tables. If the selector never appears we
            # still capture whatever HTML rendered.
            if wait_selector:
                try:
                    page.wait_for_selector(wait_selector, timeout=wait_timeout_ms)
                except Exception:  # noqa: BLE001
                    pass
            for selector in post_load_clicks or []:
                try:
                    page.click(selector, timeout=3_000)
                    page.wait_for_timeout(500)
                except Exception:  # noqa: BLE001
                    continue
            html = page.content()
            final_url = page.url
            if looks_like_login_page(url=final_url, html=html):
                return FetchResult(
                    page=page_name,
                    url=final_url,
                    status="login_required",
                    html=html,
                    error="Redirected to BallparkPal login page. Run --login to re-auth.",
                )
            raw_path = self._dump_html(page_name, slate_date, html)
            return FetchResult(
                page=page_name,
                url=final_url,
                status="ok",
                html=html,
                raw_html_path=raw_path,
            )
        except Exception as exc:  # noqa: BLE001 — single-page failure must not crash the run
            return FetchResult(page=page_name, url=url, status="error", error=str(exc))
        finally:
            try:
                page.close()
            except Exception:  # noqa: BLE001
                pass

    def _dump_html(self, page_name: str, slate_date: str | None, html: str) -> str | None:
        if not self.html_dir:
            return None
        stamp = slate_date or "latest"
        path = self.html_dir / f"{page_name}_{stamp}.html"
        try:
            path.write_text(html, encoding="utf-8")
            return str(path)
        except OSError as exc:
            logger.warning("Could not write raw HTML for %s: %s", page_name, exc)
            return None


def fetch_page_html(
    *,
    page_name: str,
    url: str | None = None,
    slate_date: str | None = None,
    headless: bool = True,
    profile_dir: Path = DEFAULT_PROFILE_DIR,
) -> FetchResult:
    """One-shot fetch helper. Opens a Playwright context, grabs one page,
    closes it. Use the class directly when fetching many pages — sharing
    the context is much faster than reopening a browser per page.
    """
    page_url = url or (PAGES.get(page_name) or {}).get("url")
    if not page_url:
        return FetchResult(page=page_name, url="", status="error", error=f"Unknown page: {page_name}")
    wait_selector = (PAGES.get(page_name) or {}).get("wait_selector", "table")
    with BallparkPalFetcher(headless=headless, profile_dir=profile_dir) as fetcher:
        return fetcher.fetch(
            page_name=page_name,
            url=page_url,
            slate_date=slate_date,
            wait_selector=wait_selector,
        )
