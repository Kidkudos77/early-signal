"""Step 6 — Dynamic NARMS danger weighting.

Extends the Step 5 per-sample danger weights with the FSIS NARMS Multi-Year
Report framework. Per the brief, weights are increased for:

  * Multidrug-resistant (MDR) strains, especially serotype INFANTIS, which the
    NARMS report identifies as the major contributor to the rise in MDR
    Salmonella in chicken (the pESI plasmid, bla_CTX-M-65).
  * Isolates with increased resistance to critically important antimicrobials,
    specifically CIPROFLOXACIN (decreased susceptibility to ciprofloxacin, DSC).

The weighting is DYNAMIC: rather than only fixed multipliers, it can ingest the
live BEAM Dashboard's per (state, month, serotype) clinically-important-AMR
percentage so weights track current resistance levels.

Resulting weight for instance i (multiplicative on the Step 5 danger weight):

    narms_mult_i = 1
        + mdr_infantis_boost      if serotype == Infantis (MDR driver)
        + mdr_boost               if MDR (resistant to >= 3 classes)
        + dsc_cipro_boost         if ciprofloxacin-nonsusceptible (DSC)
        + ci_amr_coef * ci_amr_pct   (dynamic BEAM clinically-important-AMR share)

Then w_final_i = w_step5_i * narms_mult_i.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# NARMS boost magnitudes (additive inside the multiplier).
MDR_INFANTIS_BOOST = 2.0    # serotype Infantis — the MDR driver in chicken
MDR_BOOST = 1.0             # any MDR isolate (>=3 resistant classes)
DSC_CIPRO_BOOST = 1.5       # decreased susceptibility to ciprofloxacin
CI_AMR_COEF = 2.0           # scales the BEAM clinically-important-AMR fraction


def narms_multiplier(
    *,
    serotype: pd.Series | np.ndarray,
    is_mdr: pd.Series | np.ndarray | None = None,
    is_dsc_cipro: pd.Series | np.ndarray | None = None,
    ci_amr_pct: pd.Series | np.ndarray | None = None,
    mdr_infantis_boost: float = MDR_INFANTIS_BOOST,
    mdr_boost: float = MDR_BOOST,
    dsc_cipro_boost: float = DSC_CIPRO_BOOST,
    ci_amr_coef: float = CI_AMR_COEF,
) -> np.ndarray:
    """Per-instance NARMS danger multiplier (>= 1).

    Parameters
    ----------
    serotype:
        Serotype name per instance (used to flag Infantis).
    is_mdr:
        Boolean/0-1 MDR flag per instance (resistant to >=3 classes). Optional.
    is_dsc_cipro:
        Boolean/0-1 ciprofloxacin-nonsusceptible (DSC) flag. Optional.
    ci_amr_pct:
        Clinically-important-AMR fraction in [0,1] per instance (e.g. from the
        BEAM dashboard for the instance's state/month/serotype). Optional.
    """
    sero = pd.Series(serotype).astype("string")
    n = len(sero)

    mult = np.ones(n, dtype="float64")

    # Infantis — the NARMS MDR driver.
    is_infantis = sero.str.contains("Infantis", case=False, na=False).to_numpy()
    mult += mdr_infantis_boost * is_infantis.astype("float64")

    if is_mdr is not None:
        mdr = np.asarray(pd.Series(is_mdr).fillna(0)).astype("float64")
        mult += mdr_boost * (mdr > 0).astype("float64")

    if is_dsc_cipro is not None:
        dsc = np.asarray(pd.Series(is_dsc_cipro).fillna(0)).astype("float64")
        mult += dsc_cipro_boost * (dsc > 0).astype("float64")

    if ci_amr_pct is not None:
        pct = np.asarray(pd.Series(ci_amr_pct).fillna(0), dtype="float64")
        pct = np.clip(pct, 0, 1)
        mult += ci_amr_coef * pct

    return mult


def ci_amr_lookup_from_beam(beam_report: pd.DataFrame) -> dict[tuple, float]:
    """Build a (state, month, serotype) -> clinically-important-AMR fraction map.

    Uses the BEAM 'beam_report_data' columns. The AMR column is a percentage of
    outbreak-associated isolates with clinically important antimicrobial
    resistance; we convert to a [0,1] fraction and average within each cell.
    Only Salmonella rows are retained.
    """
    col_amr = "percentage_of_outbreak_associated_isolates_with_clinically_important_antimicrobial_resistance"
    df = beam_report
    if col_amr not in df.columns:
        return {}
    sub = df[df["pathogen"].astype("string").str.contains("Salmonella", case=False, na=False)].copy()
    sub["_state"] = sub["state"].astype("string")
    sub["_month"] = pd.to_numeric(sub["month"], errors="coerce")
    sub["_sero"] = sub["serotype_species"].astype("string")
    pct = pd.to_numeric(sub[col_amr], errors="coerce")
    # Values appear as percentages (0-100); normalize to [0,1].
    sub["_amr"] = np.clip(pct / 100.0, 0, 1)
    sub = sub.dropna(subset=["_state", "_month"])
    sub["_month"] = sub["_month"].astype(int)
    grp = sub.groupby(["_state", "_month", "_sero"])["_amr"].mean()
    return {tuple(k): float(v) for k, v in grp.items()}


def apply_narms_to_weights(
    step5_weights: np.ndarray,
    narms_mult: np.ndarray,
) -> np.ndarray:
    """Combine Step 5 danger weights with the NARMS multiplier (element-wise)."""
    w = np.asarray(step5_weights, dtype="float64") * np.asarray(narms_mult, dtype="float64")
    return np.clip(w, 1e-3, None)


__all__ = [
    "MDR_INFANTIS_BOOST", "MDR_BOOST", "DSC_CIPRO_BOOST", "CI_AMR_COEF",
    "narms_multiplier",
    "ci_amr_lookup_from_beam",
    "apply_narms_to_weights",
]
