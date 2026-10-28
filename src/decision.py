"""Operational decision logic for the Early Signal command center.

Turns the Bayesian risk model into plain operational answers, with no manual
tuning required from the plant manager:

  1. Surge check (automatic). A "surge" means the true Salmonella risk for a
     region and month is above the normal yearly level, defined as the national
     average risk across all months from CDC NORS history (~15%). The chance of
     a surge is P(w > normal level), read off the Beta posterior. Because the
     threshold is fixed and data-derived, the chance of a surge rises and falls
     with the season on its own (Jan ~8%, Jul ~78% median across states).

  2. Local need. Using CDC/ATSDR SVI 2022 and Census county locations, count the
     counties within a 250-mile service radius of the facility's state center
     and the share that are high-vulnerability (overall SVI >= 0.75).

  3. Recommendation:
       surge chance >= 75% and high local need (>= 30% of nearby counties
       high-vulnerability)                     -> Donate Cooked Product
       surge chance >= 75%                     -> Divert to Cooked Lines
       surge chance >= 40%                     -> Trigger Enhanced Interventions
       otherwise                               -> Adjust Transit Radii

     Donate and Divert both apply the heat kill-step. Donate is chosen when the
     region the product would serve has many high-need communities, so the
     safely cooked protein reaches food banks there instead of being discarded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import routing

SERVICE_RADIUS_MI = 250
HIGH_VULN_SVI = 0.75
HIGH_SURGE = 0.75
ELEVATED_SURGE = 0.40
HIGH_NEED_SHARE = 0.30
MAX_ROUTES = 8

MONTHS = ["", "January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
PEAK_MONTHS = (6, 7, 8, 9)


# ---------------------------------------------------------------------------
# Risk tiers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Tier:
    label: str
    emoji: str
    level: str  # "error" | "warning" | "success" for native Streamlit alerts


def risk_tier(p_surge: float) -> Tier:
    if p_surge >= HIGH_SURGE:
        return Tier("HIGH", "🔴", "error")
    if p_surge >= ELEVATED_SURGE:
        return Tier("ELEVATED", "🟠", "warning")
    return Tier("BASELINE", "🟢", "success")


# ---------------------------------------------------------------------------
# Local vulnerability within the service radius
# ---------------------------------------------------------------------------
def haversine_mi(lat1, lon1, lat2, lon2):
    r = 3958.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


@dataclass
class LocalNeed:
    facility: tuple[float, float]
    nearby: pd.DataFrame        # all counties within the radius (+ distance_mi)
    high_need: pd.DataFrame     # nearby counties with SVI >= HIGH_VULN_SVI
    share_high: float
    seniors: float              # residents 65+ in high-need counties
    children: float             # residents <=17 in high-need counties
    population: float           # total residents in high-need counties


def local_need(state: str, vuln: pd.DataFrame,
               radius_mi: float = SERVICE_RADIUS_MI) -> LocalNeed:
    lat, lon = routing.STATE_CENTROIDS.get(state, routing.US_CENTER)
    d = haversine_mi(lat, lon, vuln["lat"].to_numpy(), vuln["lon"].to_numpy())
    nearby = vuln.assign(distance_mi=d)[d <= radius_mi].copy()
    high = nearby[nearby["svi_overall"] >= HIGH_VULN_SVI].copy()
    share = len(high) / len(nearby) if len(nearby) else 0.0
    seniors = float((high["population"] * high["pct_age65"] / 100).sum())
    children = float((high["population"] * high["pct_age17"] / 100).sum())
    return LocalNeed(facility=(lat, lon), nearby=nearby, high_need=high,
                     share_high=share, seniors=seniors, children=children,
                     population=float(high["population"].sum()))


def donation_targets(need: LocalNeed, n: int = MAX_ROUTES) -> pd.DataFrame:
    """Highest-need counties in the radius, most vulnerable first."""
    hv = need.high_need.sort_values(["svi_overall", "distance_mi"],
                                    ascending=[False, True]).head(n).copy()
    hv["seniors_65"] = (hv["population"] * hv["pct_age65"] / 100).round()
    hv["children_17"] = (hv["population"] * hv["pct_age17"] / 100).round()
    return hv


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------
@dataclass
class Recommendation:
    scenario: str
    why: str


def recommend(p_surge: float, need_share: float | None) -> Recommendation:
    high_need = need_share is not None and need_share >= HIGH_NEED_SHARE
    if p_surge >= HIGH_SURGE and high_need:
        return Recommendation(
            "Donate Cooked Product",
            "Surge is likely and many nearby communities are high-vulnerability. "
            "Cook the at-risk lots (heat kill-step) and route the safe protein to "
            "food banks serving those communities instead of discarding it.")
    if p_surge >= HIGH_SURGE:
        return Recommendation(
            "Divert to Cooked Lines",
            "Surge is likely. Move raw lots to cooked lines so the heat kill-step "
            "neutralizes Salmonella while keeping all of the protein.")
    if p_surge >= ELEVATED_SURGE:
        return Recommendation(
            "Trigger Enhanced Interventions",
            "Risk is building. Add antimicrobial washes on the line now, at low "
            "cost, before it becomes a surge.")
    return Recommendation(
        "Adjust Transit Radii",
        "Risk is at its normal level. Keep the cheapest protective posture: "
        "shorten cold-chain transit to limit growth time.")


# ---------------------------------------------------------------------------
# Plain-English insight and drivers
# ---------------------------------------------------------------------------
def headline(loc_name: str, month: int, tier: Tier, p_surge: float,
             normal_level: float) -> str:
    season = "peak Salmonella season" if month in PEAK_MONTHS else "off-peak season"
    return (f"**Risk level: {tier.label.title()}.** {loc_name} in {MONTHS[month]} "
            f"is in its {season}. There is a **{p_surge:.0%} chance** that risk is "
            f"above the normal yearly level of {normal_level:.0%}.")


def drivers(*, loc_name: str, month: int, risk: float, national_month: float,
            normal_level: float, has_local_history: bool,
            fsis_k: int, fsis_n: int) -> list[str]:
    out = []
    lift = national_month - normal_level
    if abs(lift) >= 0.01:
        word = "above" if lift > 0 else "below"
        out.append(f"**Season:** nationally, {MONTHS[month]} runs "
                   f"{abs(lift) * 100:.0f} points {word} the yearly normal.")
    else:
        out.append(f"**Season:** {MONTHS[month]} is close to the yearly normal.")
    if has_local_history:
        geo = risk - national_month
        if abs(geo) >= 0.01:
            word = "higher" if geo > 0 else "lower"
            out.append(f"**Region:** {loc_name}'s own outbreak history is "
                       f"{abs(geo) * 100:.0f} points {word} than the national level "
                       f"for {MONTHS[month]}.")
        else:
            out.append(f"**Region:** {loc_name}'s history matches the national level "
                       f"for {MONTHS[month]}.")
    else:
        out.append(f"**Region:** limited outbreak history for {loc_name} in "
                   f"{MONTHS[month]}; the national seasonal pattern is used.")
    if fsis_n > 0:
        out.append(f"**Plant sampling:** {fsis_k} of {fsis_n} recent FSIS poultry "
                   f"samples tested positive, and the estimate was updated with them.")
    else:
        out.append("**Plant sampling:** no FSIS samples for this region and month yet; "
                   "the estimate rests on outbreak history.")
    return out


__all__ = [
    "SERVICE_RADIUS_MI", "HIGH_VULN_SVI", "HIGH_SURGE", "ELEVATED_SURGE",
    "HIGH_NEED_SHARE", "MONTHS", "PEAK_MONTHS",
    "Tier", "risk_tier", "LocalNeed", "local_need", "donation_targets",
    "Recommendation", "recommend", "headline", "drivers", "haversine_mi",
]
