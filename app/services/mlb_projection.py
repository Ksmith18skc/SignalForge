"""Model-projection helpers for MLB game totals.

The original totals model collapsed every projection signal into a single
sportsbook-price-edge factor — i.e. "is this side priced better than
consensus?" — which mixes line-shopping noise with real model conviction.
This module derives a separate, line-anchored *projection edge* by
adjusting the consensus line with the environment score and reporting
how many runs the model thinks a game should score.

Three primary outputs:

* :func:`model_projected_total` — the projected run total at scan time.
  Stored on the edge row so the projection-calibration report can compute
  the projected-vs-actual residual after grading.
* :func:`projection_edge_score` — a 0-100 score for the directional
  projection edge (projected_total − line, signed by side). Plugs into
  the scoring engine alongside the existing price-edge ``odds_edge_score``.
* :func:`projection_confidence_score` — a 0-100 confidence rating for the
  projection itself, derived from input data quality (weather present,
  book count, statcast availability). The scoring engine uses this to
  penalize edges whose projection rests on thin inputs.
"""

from __future__ import annotations

from typing import Any

# Maximum runs we'll let environment alone shift the consensus line by.
# Real run-environment effects are bounded by physics (a hot, windy
# Coors-like day might add ~2 runs at the absolute extreme). We cap at
# 1.5 to avoid the projection ever pretending it has lineup-level
# certainty it doesn't actually have.
MAX_ENV_RUN_SHIFT = 1.5
# Scale factor for converting projected-vs-line edge into a 0-100 score.
# 25 points per 1 run of edge is in line with the existing
# ``odds_edge_score`` price-edge scaling (25 per 1.0 price unit).
PROJECTION_EDGE_POINTS_PER_RUN = 25.0


def model_projected_total(
    *, consensus_line: float | None, environment: dict[str, Any] | None
) -> float | None:
    """Compute the model's projected run total.

    Defined as ``consensus_line + env_adjustment`` where the adjustment is
    a clipped function of ``run_environment_score`` (50 = neutral). Returns
    ``None`` when the consensus line is missing — without an anchor we
    have no business publishing a projection.
    """
    if consensus_line is None:
        return None
    try:
        line = float(consensus_line)
    except (TypeError, ValueError):
        return None
    env = environment or {}
    env_score = env.get("run_environment_score")
    if env_score is None:
        return round(line, 2)
    try:
        env_value = float(env_score)
    except (TypeError, ValueError):
        return round(line, 2)
    # Map env_score ∈ [0, 100] (50 = neutral) to a bounded run shift.
    shift = (env_value - 50.0) / 50.0 * MAX_ENV_RUN_SHIFT
    shift = max(-MAX_ENV_RUN_SHIFT, min(MAX_ENV_RUN_SHIFT, shift))
    return round(line + shift, 2)


def projection_edge_score(
    *,
    side: str,
    consensus_line: float | None,
    projected_total: float | None,
) -> float:
    """0-100 score for the directional model-projection-vs-line edge.

    The signed edge is ``projected_total − consensus_line`` for overs and
    ``consensus_line − projected_total`` for unders, so a "positive" edge
    always means "the model favors this side." Returns the neutral 50
    sentinel when either input is missing so the score doesn't silently
    fabricate signal.
    """
    if projected_total is None or consensus_line is None:
        return 50.0
    try:
        proj = float(projected_total)
        line = float(consensus_line)
    except (TypeError, ValueError):
        return 50.0
    signed = (proj - line) if side.lower() == "over" else (line - proj)
    raw = 50.0 + signed * PROJECTION_EDGE_POINTS_PER_RUN
    return _clamp(raw)


def projection_confidence_score(
    *,
    environment: dict[str, Any] | None,
    book_count: int,
    statcast_ok: bool,
) -> float:
    """0-100 confidence in the projection itself.

    Built only from inputs we already have at scan time: was weather
    actually fetched, do we have enough books for the line to be tight,
    and did statcast give us pitcher / batter data. The output is
    consumed both as a first-class factor (low confidence → diagnostics
    surface it) and as a score penalty (low confidence → the score is
    pulled toward the neutral 50).
    """
    env = environment or {}
    warnings = list(env.get("warnings") or [])
    has_temperature = not any("Temperature missing" in w for w in warnings)
    has_wind = not any("Wind speed missing" in w for w in warnings)
    has_wind_dir = not any("Wind direction missing" in w for w in warnings)
    no_weather_payload = any(
        "Weather missing" in w for w in warnings
    )

    score = 50.0
    if not no_weather_payload:
        score += 10.0
    if has_temperature:
        score += 6.0
    if has_wind:
        score += 6.0
    if has_wind_dir:
        score += 6.0
    if book_count >= 3:
        score += 12.0
    elif book_count >= 2:
        score += 6.0
    if statcast_ok:
        score += 5.0
    # Bound to [0, 100] — we never want a stub run to overflow the scale.
    return _clamp(score)


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))
