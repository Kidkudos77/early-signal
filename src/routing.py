"""Step 7 support — Trade-Off Matrix routing scenarios + equity-map helpers.

The four routing scenarios and their exact Safety / Protein Kept / Cost values
are specified verbatim by the brief and reproduced here without alteration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import geo


@dataclass(frozen=True)
class RoutingScenario:
    name: str
    safety: str
    protein_kept: str
    cost: str
    rationale: str


# EXACT values from the brief (Step 7, Trade-Off Matrix). Do not alter.
ROUTING_SCENARIOS: tuple[RoutingScenario, ...] = (
    RoutingScenario(
        name="Divert to Cooked Lines",
        safety="High via Heat kill-step",
        protein_kept="100%",
        cost="Moderate",
        rationale="Redirect raw lots to a cooking line; the heat step kills "
                  "Salmonella while retaining all protein.",
    ),
    RoutingScenario(
        name="Adjust Transit Radii",
        safety="Moderate",
        protein_kept="100%",
        cost="Low",
        rationale="Shorten cold-chain transit distance/time to limit growth "
                  "windows; cheapest, keeps all protein, moderate safety gain.",
    ),
    RoutingScenario(
        name="Trigger Enhanced Interventions",
        safety="Moderate via Antimicrobial washes",
        protein_kept="Variable",
        cost="Low",
        rationale="Apply antimicrobial washes/interventions on the line; "
                  "protein retention varies with the intervention.",
    ),
    RoutingScenario(
        name="Donate Cooked Product",
        safety="High",
        protein_kept="100%",
        cost="High",
        rationale="Fully cook and donate to high-vulnerability communities; "
                  "highest safety and full protein use, at highest cost.",
    ),
)


def trade_off_dataframe() -> pd.DataFrame:
    """Return the Trade-Off Matrix as a DataFrame for display."""
    return pd.DataFrame([
        {
            "Scenario": s.name,
            "Safety": s.safety,
            "Protein Kept": s.protein_kept,
            "Cost": s.cost,
        }
        for s in ROUTING_SCENARIOS
    ])


# ---------------------------------------------------------------------------
# Equity-map geometry
# ---------------------------------------------------------------------------
# Approximate state centroids (lat, lon) for drawing routing vectors. Enough
# states to cover the FSIS/NORS geography; others fall back to US center.
STATE_CENTROIDS: dict[str, tuple[float, float]] = {
    "AL": (32.8, -86.8), "AK": (64.7, -152.0), "AZ": (34.3, -111.7),
    "AR": (34.9, -92.4), "CA": (37.2, -119.4), "CO": (39.0, -105.5),
    "CT": (41.6, -72.7), "DE": (39.0, -75.5), "FL": (28.6, -82.4),
    "GA": (32.6, -83.4), "HI": (20.3, -156.4), "ID": (44.4, -114.6),
    "IL": (40.0, -89.2), "IN": (39.9, -86.3), "IA": (42.0, -93.5),
    "KS": (38.5, -98.4), "KY": (37.5, -85.3), "LA": (31.0, -92.0),
    "ME": (45.4, -69.2), "MD": (39.0, -76.8), "MA": (42.3, -71.8),
    "MI": (44.3, -85.4), "MN": (46.3, -94.3), "MS": (32.7, -89.7),
    "MO": (38.4, -92.5), "MT": (47.0, -109.6), "NE": (41.5, -99.8),
    "NV": (39.3, -116.6), "NH": (43.7, -71.6), "NJ": (40.2, -74.7),
    "NM": (34.4, -106.1), "NY": (42.9, -75.6), "NC": (35.6, -79.4),
    "ND": (47.5, -100.5), "OH": (40.3, -82.8), "OK": (35.6, -97.5),
    "OR": (44.0, -120.5), "PA": (40.9, -77.8), "RI": (41.7, -71.5),
    "SC": (33.9, -80.9), "SD": (44.4, -100.2), "TN": (35.9, -86.4),
    "TX": (31.5, -99.3), "UT": (39.3, -111.7), "VT": (44.1, -72.7),
    "VA": (37.5, -78.9), "WA": (47.4, -120.5), "WV": (38.6, -80.6),
    "WI": (44.6, -89.9), "WY": (43.0, -107.6), "DC": (38.9, -77.0),
    "PR": (18.2, -66.5),
}
US_CENTER = (39.5, -98.35)


def high_vulnerability_counties(
    svi: pd.DataFrame, *, top_frac: float = 0.10, min_overall: float = 0.75
) -> pd.DataFrame:
    """Select the most socially-vulnerable counties (Donate-Product targets).

    Returns counties with svi_overall >= min_overall (or the top `top_frac` by
    SVI, whichever yields more), carrying FIPS, state, county, svi_overall.
    These are the equity-priority destinations for cooked-product donation.
    """
    d = svi.dropna(subset=["svi_overall"]).copy()
    thresh = max(min_overall, d["svi_overall"].quantile(1 - top_frac))
    hv = d[d["svi_overall"] >= thresh]
    return hv[["FIPS", "state", "county", "svi_overall"]].reset_index(drop=True)


def donate_routing_vectors(
    risk_by_state: dict[str, float],
    svi: pd.DataFrame,
    *,
    top_sources: int = 5,
    top_targets_per_source: int = 3,
) -> pd.DataFrame:
    """Build 'Donate Cooked Product' routing vectors.

    From the highest-risk source states (by surge risk) to the highest-SVI
    counties, representing where cooked donated product should flow for maximum
    equity impact. Returns one row per (source_state -> target_county) vector
    with endpoint coordinates for the map.
    """
    hv = high_vulnerability_counties(svi)
    # County target coordinate = its state centroid (county polygons handled by
    # the choropleth; vectors are drawn at state granularity for clarity).
    sources = sorted(risk_by_state.items(), key=lambda kv: -kv[1])[:top_sources]

    rows = []
    # Rank high-vulnerability counties nationally; pick the top targets overall.
    targets = hv.sort_values("svi_overall", ascending=False).head(
        top_sources * top_targets_per_source
    )
    tgt_list = list(targets.itertuples(index=False))

    ti = 0
    for src_state, risk in sources:
        s_lat, s_lon = STATE_CENTROIDS.get(src_state, US_CENTER)
        for _ in range(top_targets_per_source):
            if ti >= len(tgt_list):
                break
            t = tgt_list[ti]; ti += 1
            t_lat, t_lon = STATE_CENTROIDS.get(t.state, US_CENTER)
            rows.append({
                "source_state": src_state,
                "source_risk": risk,
                "target_fips": t.FIPS,
                "target_state": t.state,
                "target_county": t.county,
                "target_svi": t.svi_overall,
                "src_lat": s_lat, "src_lon": s_lon,
                "tgt_lat": t_lat, "tgt_lon": t_lon,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Decision logic — map risk to an automatic recommended routing action.
# ---------------------------------------------------------------------------
# Mapping from risk tier to the recommended scenario NAME (must match a
# ROUTING_SCENARIOS entry). This is the backend "heavy lifting" that frees the
# manager from toggling values themselves.
def recommend_scenario(p_exceeds: float | None, posterior_mean: float,
                       tau: float) -> tuple[str, str]:
    """Return (scenario_name, plain-English rationale) for the current risk.

    Decision tiers (driven by P(risk > tau), the actionable alert signal):
      * >= 0.50  -> Divert to Cooked Lines   (decisive protective action)
      * >= tier2 -> Trigger Enhanced Interventions (low-cost mitigation)
      * else     -> Adjust Transit Radii     (cheapest baseline posture)
    When P(risk>tau) is unavailable (national baseline), fall back to the
    posterior mean vs. tau.
    """
    signal = p_exceeds if p_exceeds is not None else posterior_mean
    if signal >= 0.50:
        return ("Divert to Cooked Lines",
                "Risk is high enough to justify a decisive heat kill-step while "
                "retaining all protein.")
    if signal >= 0.20:
        return ("Trigger Enhanced Interventions",
                "Elevated risk warrants low-cost antimicrobial interventions on "
                "the line.")
    return ("Adjust Transit Radii",
            "Baseline risk — maintain the cheapest protective posture by "
            "tightening cold-chain transit.")


RISK_TIERS = (
    # (min P(risk>tau), label, emoji, color)
    (0.50, "HIGH", "🔴", "#c62828"),
    (0.20, "ELEVATED", "🟠", "#ef6c00"),
    (0.00, "BASELINE", "🟢", "#2e7d32"),
)


def risk_tier(p_exceeds: float | None, posterior_mean: float) -> tuple[str, str, str]:
    """Return (label, emoji, color) for the risk level — the executive KPI tier."""
    signal = p_exceeds if p_exceeds is not None else posterior_mean
    for threshold, label, emoji, color in RISK_TIERS:
        if signal >= threshold:
            return label, emoji, color
    return "BASELINE", "🟢", "#2e7d32"


def risk_drivers(
    *, state: str, month: int, posterior_mean: float, prior_mean: float,
    national_month: float, national_overall: float, is_peak_season: bool,
    shap_top: list[tuple[str, float]] | None = None,
) -> list[str]:
    """Plain-English risk drivers — honest, derived from what the model computes.

    We explain the drivers the (state, month) Bayesian model + Step 3 SHAP
    actually use: the SEASONAL lift (month vs. national average), the GEOGRAPHIC
    lift (this state/month vs. national this month), and — when available — the
    top SHAP features. We deliberately do NOT invent signals the model never saw
    (e.g. a specific NOAA temperature spike); where a driver is structural we
    say so.
    """
    drivers: list[str] = []
    month_names = ["", "January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]

    # 1. Seasonal effect.
    season_lift = national_month - national_overall
    if is_peak_season and season_lift > 0.01:
        drivers.append(
            f"Seasonal peak — {month_names[month]} historically runs "
            f"{season_lift * 100:+.0f} pts above the yearly average "
            f"(summer Salmonella surge).")
    elif season_lift < -0.01:
        drivers.append(
            f"Off-peak season — {month_names[month]} runs {season_lift * 100:+.0f} "
            f"pts vs. the yearly average, lowering baseline risk.")
    else:
        drivers.append(f"Seasonal effect for {month_names[month]} is near the "
                       f"yearly average.")

    # 2. Geographic effect (cell posterior vs. national this month).
    geo_lift = posterior_mean - national_month
    if abs(geo_lift) > 0.01:
        direction = "above" if geo_lift > 0 else "below"
        drivers.append(
            f"Geographic effect — {state} in {month_names[month]} sits "
            f"{abs(geo_lift) * 100:.0f} pts {direction} the national level for "
            f"this month.")

    # 3. SHAP top features (Step 3), if provided.
    if shap_top:
        feats = ", ".join(f"{name}" for name, _ in shap_top[:3])
        drivers.append(f"Top model features for this prediction: {feats}.")
    else:
        drivers.append(
            "Model features (full SHAP breakdown in Detailed Diagnostics): "
            "season encoding, exposure state, and outbreak setting.")

    return drivers


__all__ = [
    "RoutingScenario",
    "ROUTING_SCENARIOS",
    "trade_off_dataframe",
    "STATE_CENTROIDS",
    "high_vulnerability_counties",
    "donate_routing_vectors",
    "recommend_scenario",
    "risk_tier",
    "risk_drivers",
    "RISK_TIERS",
]
