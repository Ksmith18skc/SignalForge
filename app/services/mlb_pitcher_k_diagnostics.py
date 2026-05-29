"""Stage-by-stage diagnostics for the pitcher-K edge pipeline.

The dashboard's empty state used to collapse every failure mode into
"No qualifying edges." Operators couldn't tell whether BallparkPal had
zero projections, sportsbook had zero pitcher props, the names didn't
match, or the score threshold was eating everything.

This module owns the counter dataclass + a tiny helper for choosing
the right operator-facing message based on which stage actually
dropped to zero. The counters are populated during the edge engine
run and persisted onto the daily-card payload so the dashboard can
render them without re-running.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PitcherKDiagnostics:
    """Per-stage counters for one MLB edge scan.

    Stage ordering mirrors the engine's actual control flow so the
    operator can read the counters top-to-bottom and see exactly where
    the funnel collapsed.
    """

    # Input availability
    strikeout_projections_loaded: int = 0
    sportsbook_pitcher_k_props_loaded: int = 0
    games_with_pitchers: int = 0

    # Name matching (per-pitcher × per-source)
    pitcher_names_matched_sportsbook: int = 0
    pitcher_names_unmatched_sportsbook: int = 0
    pitcher_names_matched_ballparkpal: int = 0
    pitcher_names_unmatched_ballparkpal: int = 0

    # Line / candidate construction
    lines_matched_against_props: int = 0
    candidates_built_from_sportsbook: int = 0
    candidates_built_from_ballparkpal_fallback: int = 0
    candidates_built_total: int = 0

    # Threshold gating
    candidates_rejected_missing_odds: int = 0
    candidates_rejected_by_threshold: int = 0
    candidates_promoted_watchlist: int = 0
    candidates_promoted_candidate: int = 0
    candidates_promoted_strong: int = 0

    # Output
    cards_rendered: int = 0

    # Sanity-rejected BPP rows. A non-empty list here is the loud
    # operator-facing signal that the cache parser mis-mapped a column
    # (e.g. the K-line cell carried american odds because over_line /
    # over_odds got swapped).
    rejected_for_bad_mapping: int = 0
    bad_mapping_examples: list[dict[str, Any]] = field(default_factory=list)

    # Worked examples (kept tiny — the dashboard renders these inline).
    unmatched_pitcher_examples: list[dict[str, Any]] = field(default_factory=list)
    fallback_card_examples: list[dict[str, Any]] = field(default_factory=list)

    # Roster diagnostics — the dashboard renders these side-by-side so
    # the operator can SEE why a name didn't match without scrolling
    # through individual logs. Both are populated unconditionally each
    # scan.
    mlb_probable_pitchers_today: list[dict[str, Any]] = field(default_factory=list)
    ballparkpal_pitchers_in_cache: list[str] = field(default_factory=list)

    def empty_state_message(self) -> str:
        """Choose the most specific message for the current counter state.

        Branches are ordered from MOST specific to most generic — the
        first one whose precondition fires wins. That way a "30 BPP rows
        loaded but 0 matched today's pitchers" outcome reports the real
        cause instead of being swallowed by the generic "no fallback
        odds" branch (which was the bug behind the user-visible message
        ``30 projections loaded, 0 pitcher K odds found ... no
        BallparkPal fallback odds available either`` while name matching
        was actually the problem).
        """
        if self.strikeout_projections_loaded == 0:
            return (
                "Pitcher K scan did not run: 0 strikeout projections "
                "loaded. Upload a BallparkPal Strikeout CSV first."
            )
        # Bad-mapping first — when this fires the cache itself is the
        # problem and re-running won't help. Operator needs to re-upload.
        if (
            self.rejected_for_bad_mapping > 0
            and self.candidates_built_from_ballparkpal_fallback == 0
        ):
            return (
                f"{self.rejected_for_bad_mapping} BallparkPal row(s) "
                "matched today's pitchers but were rejected by sanity "
                "checks (line looked like american odds, or "
                "projected_k / over_line out of range). The cache may "
                "have been parsed before the explicit over_line / "
                "over_odds aliases shipped — re-upload the CSV."
            )
        # Name mismatch is the next most specific failure. 30 rows
        # loaded but none of today's MLB probable pitchers found a row.
        if (
            self.strikeout_projections_loaded > 0
            and self.pitcher_names_matched_ballparkpal == 0
            and self.pitcher_names_matched_sportsbook == 0
            and self.rejected_for_bad_mapping == 0
        ):
            mlb_count = len(self.mlb_probable_pitchers_today)
            return (
                f"{self.strikeout_projections_loaded} BallparkPal "
                f"projections loaded, {mlb_count} MLB probable pitchers "
                "for today — but 0 names matched. Check the "
                "'Pitchers MLB expects today' vs 'BallparkPal cache' "
                "panel below to see the name drift."
            )
        if (
            self.sportsbook_pitcher_k_props_loaded == 0
            and self.candidates_built_from_ballparkpal_fallback == 0
            and self.strikeout_projections_loaded == 0
        ):
            return (
                f"{self.strikeout_projections_loaded} strikeout projections "
                "loaded, 0 pitcher K odds found in the sportsbook cache "
                "and no BallparkPal fallback odds available either."
            )
        if (
            self.candidates_built_total > 0
            and self.cards_rendered == 0
            and self.candidates_rejected_by_threshold > 0
        ):
            return (
                f"{self.candidates_built_total} candidates built, "
                f"{self.candidates_rejected_by_threshold} rejected by "
                "K-edge thresholds — none passed the 0.15-run watchlist "
                "floor. Lower the threshold or check the projections "
                "for systematic bias."
            )
        if self.candidates_built_total == 0:
            return (
                "Pipeline reached the edge builder but produced 0 "
                "candidates. Open the Pitcher K diagnostics panel "
                "below — see the name match + rejection tables."
            )
        return (
            "No qualifying pitcher K edges in this band — open the "
            "diagnostics panel below for stage counters."
        )

    def add_unmatched_example(
        self, *, pitcher_name: str | None, source: str, reason: str,
    ) -> None:
        """Record a worked unmatched-pitcher example. Capped at 10 rows
        so the daily-card JSON stays small."""
        if len(self.unmatched_pitcher_examples) >= 10:
            return
        self.unmatched_pitcher_examples.append({
            "pitcher_name": pitcher_name,
            "source": source,
            "reason": reason,
        })

    def add_fallback_example(self, payload: dict[str, Any]) -> None:
        if len(self.fallback_card_examples) >= 10:
            return
        self.fallback_card_examples.append(payload)

    def add_bad_mapping_example(
        self,
        *,
        pitcher_name: str | None,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record a row that failed sanity validation. The dashboard
        renders these as a table with pitcher / projected_k / over_line
        / over_odds / rejection_reason so the operator can pinpoint the
        upstream parse bug rather than seeing aggregated counts."""
        self.rejected_for_bad_mapping += 1
        if len(self.bad_mapping_examples) >= 10:
            return
        entry = {
            "pitcher_name": pitcher_name,
            "rejection_reason": reason,
        }
        entry.update(details or {})
        self.bad_mapping_examples.append(entry)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict for the daily-card / scan-result payload."""
        return asdict(self)


# K-edge magnitude thresholds. The user explicitly asked for these to
# replace the "score >= 65" gate that was eating every pitcher card —
# K markets move in tighter ranges than game totals, so a small numeric
# edge on the projection vs line is meaningful even when the broader
# composite score sits below 65.
K_EDGE_WATCHLIST_FLOOR = 0.15
K_EDGE_CANDIDATE_FLOOR = 0.35
K_EDGE_STRONG_FLOOR = 0.65


def classify_k_edge_magnitude(k_edge: float | None) -> str:
    """Return ``"strong" | "candidate" | "watchlist" | "below"``.

    ``k_edge`` is the SIGNED projection-vs-line edge. Magnitude (not
    direction) decides the band — direction decides Over vs Under.
    """
    if k_edge is None:
        return "below"
    magnitude = abs(float(k_edge))
    if magnitude >= K_EDGE_STRONG_FLOOR:
        return "strong"
    if magnitude >= K_EDGE_CANDIDATE_FLOOR:
        return "candidate"
    if magnitude >= K_EDGE_WATCHLIST_FLOOR:
        return "watchlist"
    return "below"


def diagnostics_from_payload(payload: dict[str, Any] | None) -> PitcherKDiagnostics:
    """Round-trip a previously-persisted diagnostics dict back into the
    dataclass. Missing fields take the field default so older payloads
    keep deserializing."""
    diag = PitcherKDiagnostics()
    if not isinstance(payload, dict):
        return diag
    for key, value in payload.items():
        if hasattr(diag, key):
            setattr(diag, key, value)
    return diag
