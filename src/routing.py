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


__all__ = [
    "RoutingScenario",
    "ROUTING_SCENARIOS",
    "trade_off_dataframe",
    "STATE_CENTROIDS",
    "US_CENTER",
    "high_vulnerability_counties",
    "donate_routing_vectors",
]
