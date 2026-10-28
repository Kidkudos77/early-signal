"""Step 6 — BEAM ground-truth validation.

Cross-references the model's Out-of-Fold (OOF) predicted Salmonella risk
hotspots against the CDC BEAM Dashboard's historical outbreak/isolate data, to
serve as a ground-truth validation tool (the brief).

Approach
--------
  1. Aggregate OOF predictions to a (state, month) risk surface:
       predicted_risk[state, month] = mean OOF p(salmonella | x) over NORS
       outbreaks in that cell  (the model's "hotspot" signal).
  2. Pull BEAM Salmonella isolate counts per (state, month) as ground truth.
  3. Join and quantify agreement (Spearman/Pearson correlation, top-decile
     hotspot overlap).
  4. Render a side-by-side choropleth (predicted risk vs. BEAM observed) with
     Plotly, saved as an interactive HTML artifact.

BEAM uses 2-letter state codes and year/month, which align with our geo keys.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import geo, ingest_api


@dataclass
class ValidationResult:
    merged: pd.DataFrame          # per (state) predicted vs. observed
    pearson: float
    spearman: float
    topk_overlap: float           # fraction of predicted top-k states in BEAM top-k
    n_states: int


def predicted_risk_by_state(
    oof_pred: np.ndarray,
    state_series: pd.Series,
) -> pd.DataFrame:
    """Aggregate OOF probabilities to mean predicted risk per state (abbr)."""
    df = pd.DataFrame({
        "state": pd.Series(state_series).map(geo.to_abbr),
        "risk": np.asarray(oof_pred, dtype="float64"),
    }).dropna(subset=["state"])
    out = df.groupby("state")["risk"].mean().rename("predicted_risk").reset_index()
    return out


def beam_salmonella_by_state(
    beam_report: pd.DataFrame | None = None,
    *,
    years: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """BEAM observed Salmonella signal per state (abbr).

    Returns, per state:
      * beam_salmonella_isolates : total Salmonella isolates (absolute volume,
                                   population-dominated).
      * beam_salmonella_share    : Salmonella isolates / all-pathogen isolates
                                   (a RATE, directly comparable to the model's
                                   predicted Salmonella risk).

    The share is the apples-to-apples ground truth: the model predicts the
    probability that an outbreak is Salmonella (a rate), so we validate against
    BEAM's Salmonella fraction of isolates, not raw counts (which track state
    population rather than Salmonella propensity).
    """
    if beam_report is None:
        beam_report = ingest_api.fetch_cdc_socrata("beam_report_data")
    df = beam_report.copy()
    if years is not None:
        df = df[pd.to_numeric(df["year"], errors="coerce").isin(years)]

    df["_state"] = df["state"].map(geo.to_abbr)
    df["_iso"] = pd.to_numeric(df["number_of_isolates"], errors="coerce").fillna(0)
    df["_is_salm"] = df["pathogen"].astype("string").str.contains(
        "Salmonella", case=False, na=False)
    df = df.dropna(subset=["_state"])

    total_by_state = df.groupby("_state")["_iso"].sum().rename("beam_total_isolates")
    salm_by_state = (df[df["_is_salm"]].groupby("_state")["_iso"].sum()
                     .rename("beam_salmonella_isolates"))
    out = pd.concat([total_by_state, salm_by_state], axis=1).reset_index()
    out = out.rename(columns={"_state": "state"})
    out["beam_salmonella_isolates"] = out["beam_salmonella_isolates"].fillna(0)
    out["beam_salmonella_share"] = (
        out["beam_salmonella_isolates"] / out["beam_total_isolates"].replace(0, np.nan)
    )
    return out


def validate_against_beam(
    oof_pred: np.ndarray,
    state_series: pd.Series,
    *,
    beam_report: pd.DataFrame | None = None,
    top_k: int = 10,
) -> ValidationResult:
    """Join predicted risk vs. BEAM observed and quantify agreement.

    Agreement is measured against BEAM's Salmonella SHARE of isolates (a rate),
    which is the like-for-like comparison to the model's predicted Salmonella
    risk. Raw isolate counts are retained in the merged table for context but
    are NOT used for the headline correlation (they track state population).
    """
    from scipy.stats import pearsonr, spearmanr

    pred = predicted_risk_by_state(oof_pred, state_series)
    obs = beam_salmonella_by_state(beam_report)
    merged = pred.merge(obs, on="state", how="inner")
    merged = merged.dropna(subset=["beam_salmonella_share"])

    if len(merged) < 3:
        return ValidationResult(merged=merged, pearson=float("nan"),
                                spearman=float("nan"), topk_overlap=float("nan"),
                                n_states=len(merged))

    merged["beam_log"] = np.log1p(merged["beam_salmonella_isolates"])
    # Headline agreement: predicted risk (rate) vs. BEAM Salmonella share (rate).
    pear = float(pearsonr(merged["predicted_risk"], merged["beam_salmonella_share"])[0])
    spear = float(spearmanr(merged["predicted_risk"], merged["beam_salmonella_share"])[0])

    k = min(top_k, len(merged))
    pred_top = set(merged.nlargest(k, "predicted_risk")["state"])
    obs_top = set(merged.nlargest(k, "beam_salmonella_share")["state"])
    overlap = len(pred_top & obs_top) / k

    return ValidationResult(merged=merged, pearson=pear, spearman=spear,
                            topk_overlap=overlap, n_states=len(merged))


def render_hotspot_map(
    result: ValidationResult,
    out_path: Path,
) -> Path:
    """Render a side-by-side choropleth: predicted risk vs. BEAM observed.

    Saves an interactive self-contained HTML (Plotly). US state choropleth
    using 2-letter codes.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m = result.merged

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "choropleth"}, {"type": "choropleth"}]],
        subplot_titles=("Model predicted Salmonella risk (OOF)",
                        "BEAM observed Salmonella share of isolates (ground truth)"),
    )
    fig.add_trace(
        go.Choropleth(
            locations=m["state"], locationmode="USA-states",
            z=m["predicted_risk"], colorscale="OrRd",
            colorbar=dict(title="risk", x=0.46),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Choropleth(
            locations=m["state"], locationmode="USA-states",
            z=m["beam_salmonella_share"], colorscale="Blues",
            colorbar=dict(title="Salm. share", x=1.0),
        ),
        row=1, col=2,
    )
    fig.update_layout(
        title_text=(f"Ground-truth validation (rate vs. rate) — "
                    f"Pearson(r)={result.pearson:.2f}, "
                    f"Spearman={result.spearman:.2f}, "
                    f"top-{min(10,result.n_states)} overlap={result.topk_overlap:.0%}"),
        geo=dict(scope="usa"), geo2=dict(scope="usa"),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return out_path


__all__ = [
    "ValidationResult",
    "predicted_risk_by_state",
    "beam_salmonella_by_state",
    "validate_against_beam",
    "render_hotspot_map",
]
