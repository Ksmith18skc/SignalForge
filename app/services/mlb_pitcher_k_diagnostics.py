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

    # Worked examples (kept tiny — the dashboard renders these inline).
    unmatched_pitcher_examples: list[dict[str, Any]] = field(default_factory=list)
    fallback_card_examples: list[dict[str, Any]] = field(default_factory=list)

    def empty_state_message(self) -> str:
        """Choose the most specific message for the current counter state.

        The choices match the spec the user gave: surface the FIRST
        funnel collapse, so the operator can fix it and re-run rather
        than playing whack-a-mole on the downstream stages.
        """
        if self.strikeout_projections_loaded == 0:
            return (
                "Pitcher K scan did not run: 0 strikeout projections "
                "loaded. Upload a BallparkPal Strikeout CSV first."
            )
        if (
            self.sportsbook_pitcher_k_props_loaded == 0
            and self.candidates_built_from_ballparkpal_fallback == 0
        ):
            return (
                f"{self.strikeout_projections_loaded} strikeout projections "
                "loaded, 0 pitcher K odds found in the sportsbook cache "
                "and no BallparkPal fallback odds available either."
            )
        if (
            self.pitcher_names_matched_sportsbook == 0
            and self.pitcher_names_matched_ballparkpal == 0
        ):
            return (
                f"{self.strikeout_projections_loaded} projections loaded, "
                f"{self.sportsbook_pitcher_k_props_loaded} odds rows seen, "
                "but 0 pitcher names matched. Open the Pitcher K "
                "diagnostics panel to see the unmatched examples."
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
                "candidates. Open the Pitcher K diagnostics panel."
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
