"""Match MLB StatsAPI games to Odds-API events by normalized team names.

This is the load-bearing fix for "all MLB edges show line=null / book_count=0":
the previous lookup used `search_events("AwayName HomeName")` and grabbed the
first result, which silently returned non-MLB events or no event at all. The
team-normalized matcher below is strict (city+nickname tokens must overlap)
and emits structured logs so /mlb/debug/odds/* can show why a game missed.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# Strip these tokens before matching; they're city/role noise that varies
# between MLB StatsAPI and Odds-API (e.g. "Los Angeles Angels of Anaheim").
_TEAM_NOISE = {
    "the",
    "of",
    "city",
    "club",
    "baseball",
    "team",
    "fc",
    "mlb",
    "al",
    "nl",
}

# Common city-only abbreviations Odds-API uses. The values are the canonical
# nickname tokens we expect MLB StatsAPI to emit; we add them so a payload
# with home="NY Yankees" still matches "New York Yankees".
_TEAM_ALIASES: dict[str, list[str]] = {
    "ny yankees": ["new york yankees"],
    "ny mets": ["new york mets"],
    "la dodgers": ["los angeles dodgers"],
    "la angels": ["los angeles angels"],
    "sf giants": ["san francisco giants"],
    "sd padres": ["san diego padres"],
    "tb rays": ["tampa bay rays"],
    "kc royals": ["kansas city royals"],
    "cws": ["chicago white sox"],
    "chw": ["chicago white sox"],
    "chc": ["chicago cubs"],
    "wsh nationals": ["washington nationals"],
    "ari diamondbacks": ["arizona diamondbacks"],
    "az diamondbacks": ["arizona diamondbacks"],
}


@dataclass(frozen=True)
class MatchResult:
    """Outcome of a single game/odds match attempt."""

    game_pk: int
    home_team: str
    away_team: str
    matched_event_id: str | None
    matched_event_home: str | None
    matched_event_away: str | None
    match_strength: float  # 0.0..1.0
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "game_pk": self.game_pk,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "matched_event_id": self.matched_event_id,
            "matched_event_home": self.matched_event_home,
            "matched_event_away": self.matched_event_away,
            "match_strength": round(self.match_strength, 3),
            "reason": self.reason,
        }


def normalize_team_name(value: str | None) -> str:
    """Lowercase + strip diacritics/punctuation + drop noise tokens.

    Returns a space-separated token string suitable for set comparison.
    """
    if not value:
        return ""
    cleaned = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", cleaned).lower()
    expanded = _TEAM_ALIASES.get(cleaned.strip(), None)
    if expanded:
        cleaned = expanded[0]
    tokens = [t for t in cleaned.split() if t and t not in _TEAM_NOISE]
    return " ".join(tokens)


def team_token_set(value: str | None) -> set[str]:
    return set(normalize_team_name(value).split())


def teams_match(a: str | None, b: str | None) -> bool:
    """True iff the two names share a meaningful nickname token.

    "Los Angeles Dodgers" vs "LA Dodgers" -> {dodgers} overlap, match.
    "New York Yankees" vs "Boston Red Sox" -> no overlap, no match.
    """
    set_a = team_token_set(a)
    set_b = team_token_set(b)
    if not set_a or not set_b:
        return False
    # If either side is already a strict subset, accept (handles "Yankees"
    # vs "New York Yankees").
    if set_a.issubset(set_b) or set_b.issubset(set_a):
        return True
    # Otherwise require the *non-city* nickname to overlap. We approximate
    # "nickname" as the last token of the longer string, which is the MLB
    # convention ("Diamondbacks", "Red Sox" → take the last word).
    nickname_a = next(iter(reversed(list(set_a))), None) if len(set_a) == 1 else list(set_a)[-1:]
    nickname_b = next(iter(reversed(list(set_b))), None) if len(set_b) == 1 else list(set_b)[-1:]
    # Simpler: any shared token wins.
    return bool(set_a & set_b)


def match_game_to_event(
    game: dict[str, Any],
    events: Iterable[dict[str, Any]],
) -> MatchResult:
    """Find the best Odds-API event for one MLB game.

    Strength = (home_match + away_match) / 2, where each side scores 1.0 for
    a strict token overlap. We pick the highest-scoring event and require
    strength >= 0.5 (i.e. at least one side matched) to call it a hit.
    """
    home = game.get("home_team") or ""
    away = game.get("away_team") or ""
    game_pk = int(game.get("game_pk") or 0)

    best: tuple[float, dict[str, Any] | None] = (0.0, None)
    for event in events:
        if not isinstance(event, dict):
            continue
        event_home = event.get("home") or event.get("homeTeam") or ""
        event_away = event.get("away") or event.get("awayTeam") or ""
        # Try both orientations — some providers swap home/away.
        score_aligned = (
            float(teams_match(home, event_home)) + float(teams_match(away, event_away))
        ) / 2
        score_swapped = (
            float(teams_match(home, event_away)) + float(teams_match(away, event_home))
        ) / 2
        score = max(score_aligned, score_swapped)
        if score > best[0]:
            best = (score, event)

    score, event = best
    if event is None or score < 0.5:
        return MatchResult(
            game_pk=game_pk,
            home_team=home,
            away_team=away,
            matched_event_id=None,
            matched_event_home=None,
            matched_event_away=None,
            match_strength=score,
            reason="no Odds-API event with matching team tokens",
        )
    return MatchResult(
        game_pk=game_pk,
        home_team=home,
        away_team=away,
        matched_event_id=str(event.get("id") or event.get("eventId") or ""),
        matched_event_home=event.get("home") or event.get("homeTeam"),
        matched_event_away=event.get("away") or event.get("awayTeam"),
        match_strength=score,
        reason="matched" if score >= 1.0 else "matched (one side only)",
    )


def match_all_games(
    games: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> tuple[list[MatchResult], list[dict[str, Any]]]:
    """Match every game and report which Odds-API events stayed unused."""
    results = [match_game_to_event(g, events) for g in games]
    used_ids = {r.matched_event_id for r in results if r.matched_event_id}
    unmatched_events = [
        e for e in events
        if str(e.get("id") or e.get("eventId") or "") not in used_ids
    ]
    matched_count = sum(1 for r in results if r.matched_event_id)
    logger.info(
        "MLB odds match: games=%d events=%d matched=%d unmatched_events=%d",
        len(games), len(events), matched_count, len(unmatched_events),
    )
    return results, unmatched_events
