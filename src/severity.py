"""Step 5 support — serotype classes and per-outbreak severity from NORS.

These feed the danger-weighted loss machinery:
  * serotype class per Salmonella outbreak (SubtypeName_* aligned to the
    Salmonella GenusName_* slot) -> defines the class index k for L_kj.
  * severity (hospitalizations, deaths, estimated cases) -> a continuous
    danger scale used to build per-sample weights w_i.

The brief singles out Enteritidis and Typhimurium as severe human-illness
serotypes whose misses must be penalized heavily; CONCERN_SEROTYPES below is
ordered so these receive the highest base danger.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import ingest_local, targets

# Severe human-illness serotypes (brief) + high-concern NARMS serotypes.
# Base danger rank used when no severity data is available. Enteritidis and
# Typhimurium are the brief's explicit "severe" serotypes.
SEROTYPE_BASE_DANGER: dict[str, float] = {
    "Enteritidis": 3.0,
    "Typhimurium": 3.0,
    "Infantis": 2.5,      # MDR driver (NARMS) — elevated for Step 6
    "I 4,[5],12:i:-": 2.0,
    "Newport": 2.0,
    "Heidelberg": 2.0,
    "Montevideo": 1.5,
}
DEFAULT_SEROTYPE_DANGER = 1.0  # any other / unknown Salmonella serotype


def salmonella_serotype(nors: pd.DataFrame) -> pd.Series:
    """Return the Salmonella serotype per row (NaN if not a Salmonella outbreak).

    Aligns SubtypeName_<i> to the GenusName_<i> slot that is Salmonella; takes
    the first such slot per outbreak.
    """
    gen_cols = [c for c in nors.columns if c.startswith("GenusName_")]
    sub_cols = [c for c in nors.columns if c.startswith("SubtypeName_")]
    sero = pd.Series(pd.NA, index=nors.index, dtype="string")
    for gc, sc in zip(gen_cols, sub_cols):
        is_s = nors[gc].astype("string").str.contains("Salmonella", case=False, na=False)
        vals = nors[sc].astype("string")
        take = is_s & sero.isna() & vals.notna()
        sero = sero.mask(take, vals)
    return sero


def severity_score(nors: pd.DataFrame) -> pd.Series:
    """Continuous per-outbreak severity in roughly [0, 1+].

    Combines hospitalizations, deaths, and estimated cases on a log scale so a
    few very large outbreaks don't dominate. Deaths weighted most heavily.
    """
    hosp = pd.to_numeric(nors.get("HospitalNum"), errors="coerce").fillna(0)
    deaths = pd.to_numeric(nors.get("DeathsNum"), errors="coerce").fillna(0)
    cases = pd.to_numeric(nors.get("EstimatedPrimary"), errors="coerce").fillna(0)

    # Log1p compresses heavy tails; weights reflect public-health gravity.
    score = (
        0.5 * np.log1p(deaths) * 3.0
        + 0.3 * np.log1p(hosp)
        + 0.2 * np.log1p(cases)
    )
    # Normalize to ~[0,1] by the 99th percentile so weights stay bounded.
    denom = np.nanpercentile(score[score > 0], 99) if (score > 0).any() else 1.0
    denom = denom if denom and denom > 0 else 1.0
    return pd.Series(np.clip(score / denom, 0, None), index=nors.index)


def serotype_base_danger(sero: pd.Series) -> pd.Series:
    """Map serotype names to their base danger multiplier."""
    return sero.map(lambda s: SEROTYPE_BASE_DANGER.get(str(s), DEFAULT_SEROTYPE_DANGER)
                    if pd.notna(s) else DEFAULT_SEROTYPE_DANGER)


def build_severity_frame(nors: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-row serotype, Salmonella flag, severity score, and base danger."""
    if nors is None:
        nors = ingest_local.load_nors(foodborne_only=True)
    sero = salmonella_serotype(nors)
    return pd.DataFrame({
        "is_salmonella": targets._salmonella_mask(nors).astype(int).to_numpy(),
        "serotype": sero.to_numpy(),
        "severity": severity_score(nors).to_numpy(),
        "base_danger": serotype_base_danger(sero).to_numpy(),
    })


__all__ = [
    "SEROTYPE_BASE_DANGER",
    "DEFAULT_SEROTYPE_DANGER",
    "salmonella_serotype",
    "severity_score",
    "serotype_base_danger",
    "build_severity_frame",
]
