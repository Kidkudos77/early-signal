"""Step 3 — Target construction from NORS.

The FSIS RTE sample cannot supply a supervised target (its Salmonella result is
single-class: all Negative/NULL, no positives). NORS foodborne outbreak data is
the modeling substrate: 8,167 foodborne outbreaks, 2014-2024, with pathogen,
food, geography, season, and severity signals.

Three complementary targets (all built here), layering into the decision flow:

  1. is_salmonella          (binary)   - any foodborne Salmonella outbreak
  2. is_poultry_salmonella  (binary)   - Salmonella outbreak attributed to a
                                         poultry food vehicle (RTE-poultry focus)
  3. monthly_surge          (regression)- per (state x year-month) Salmonella
                                         outbreak COUNT; the time-aware surge axis

LEAKAGE DISCIPLINE: the label is derived from GenusName_* (pathogen) and
Food_*_FoodName (vehicle). Those columns therefore MUST NOT appear in X. The
feature set is restricted to season (from DateFirstIll), geography
(ExposureState, MultiStateExposure), setting (SettingExposure_*), and outbreak
context (EstimatedPrimary, HospitalNum, DeathsNum) — none of which encode the
pathogen identity directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from . import ingest_local

# Keyword set reused from ingest_local for poultry vehicle detection.
_POULTRY_PATTERN = "|".join(ingest_local.POULTRY_KEYWORDS)

# Feature columns allowed into X for the two classification targets. These are
# deliberately NON-label-bearing (no pathogen/food-name columns).
CLASSIFIER_CONTEXT_NUMERIC = ("EstimatedPrimary", "HospitalNum", "DeathsNum",
                              "MultiStateExposure")
CLASSIFIER_CONTEXT_CATEGORICAL = ("ExposureState",)
SETTING_PREFIX = "SettingExposure_"
DATE_COLUMN = "DateFirstIll"


TaskType = Literal["classification", "regression"]


@dataclass
class TargetBundle:
    """A ready-to-model (X, y) pair plus metadata for one target."""

    name: str
    task: TaskType
    X: pd.DataFrame
    y: pd.Series
    positive_rate: float | None  # for classification; None for regression
    description: str

    def __repr__(self) -> str:  # concise, informative
        pr = f" pos_rate={self.positive_rate:.3f}" if self.positive_rate is not None else ""
        return (f"TargetBundle(name={self.name!r}, task={self.task}, "
                f"X={self.X.shape}, y_len={len(self.y)}{pr})")


# ---------------------------------------------------------------------------
# Shared label helpers (operate on raw NORS)
# ---------------------------------------------------------------------------
def _salmonella_mask(nors: pd.DataFrame) -> pd.Series:
    """True where any GenusName_* slot is Salmonella (case-insensitive)."""
    cols = [c for c in nors.columns if c.startswith("GenusName_")]
    mask = pd.Series(False, index=nors.index)
    for c in cols:
        mask |= nors[c].astype("string").str.contains("Salmonella", case=False, na=False)
    return mask


def _poultry_vehicle_mask(nors: pd.DataFrame) -> pd.Series:
    """True where any Food_*_FoodName names a poultry vehicle."""
    cols = [c for c in nors.columns if c.endswith("_FoodName")]
    mask = pd.Series(False, index=nors.index)
    for c in cols:
        mask |= nors[c].astype("string").str.contains(
            _POULTRY_PATTERN, case=False, na=False)
    return mask


def _primary_setting(nors: pd.DataFrame) -> pd.Series:
    """Collapse SettingExposure_1..N into a single primary-setting category.

    Uses the first non-null setting per outbreak (SettingExposure_1 dominates;
    falls back to later slots). High-cardinality but << the cap, so it one-hot
    encodes cleanly downstream.
    """
    set_cols = [c for c in nors.columns if c.startswith(SETTING_PREFIX)]
    if not set_cols:
        return pd.Series(pd.NA, index=nors.index, dtype="string")
    # Row-wise first non-null.
    sub = nors[set_cols].astype("string")
    first = sub.bfill(axis=1).iloc[:, 0]
    return first


def _build_classifier_features(nors: pd.DataFrame) -> pd.DataFrame:
    """Assemble the leakage-safe feature frame for the two classifiers.

    Columns: DateFirstIll (consumed by Step 2 DateFeatureEngineer), ExposureState,
    primary setting, and numeric outbreak-context fields. NO pathogen/food-name
    columns.
    """
    X = pd.DataFrame(index=nors.index)
    # Date passthrough (Step 2's DateFeatureEngineer turns it into season feats).
    X[DATE_COLUMN] = pd.to_datetime(nors[DATE_COLUMN], errors="coerce")
    # Geography.
    for c in CLASSIFIER_CONTEXT_CATEGORICAL:
        if c in nors.columns:
            X[c] = nors[c].astype("string")
    X["primary_setting"] = _primary_setting(nors)
    # Numeric context.
    for c in CLASSIFIER_CONTEXT_NUMERIC:
        if c in nors.columns:
            X[c] = pd.to_numeric(nors[c], errors="coerce")
    return X


# ---------------------------------------------------------------------------
# Target 1: is_salmonella (binary)
# ---------------------------------------------------------------------------
def build_is_salmonella(nors: pd.DataFrame | None = None) -> TargetBundle:
    """Binary: is this foodborne outbreak a Salmonella outbreak?"""
    if nors is None:
        nors = ingest_local.load_nors(foodborne_only=True)
    y = _salmonella_mask(nors).astype(int)
    X = _build_classifier_features(nors)
    return TargetBundle(
        name="is_salmonella",
        task="classification",
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        positive_rate=float(y.mean()),
        description="Any foodborne Salmonella outbreak (NORS GenusName_* contains Salmonella).",
    )


# ---------------------------------------------------------------------------
# Target 2: is_poultry_salmonella (binary)
# ---------------------------------------------------------------------------
def build_is_poultry_salmonella(nors: pd.DataFrame | None = None) -> TargetBundle:
    """Binary: is this a Salmonella outbreak attributed to a poultry vehicle?

    Positive = Salmonella AND poultry food vehicle. This is the narrowest,
    most RTE-poultry-aligned target; fewer positives, so calibration is watched.
    """
    if nors is None:
        nors = ingest_local.load_nors(foodborne_only=True)
    y = (_salmonella_mask(nors) & _poultry_vehicle_mask(nors)).astype(int)
    X = _build_classifier_features(nors)
    return TargetBundle(
        name="is_poultry_salmonella",
        task="classification",
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        positive_rate=float(y.mean()),
        description="Salmonella outbreak with a poultry food vehicle (RTE-poultry focus).",
    )


# ---------------------------------------------------------------------------
# Target 3: monthly_surge (regression over state x year-month)
# ---------------------------------------------------------------------------
def build_monthly_surge(nors: pd.DataFrame | None = None) -> TargetBundle:
    """Regression: Salmonella outbreak COUNT per (state x year-month).

    Aggregates NORS Salmonella outbreaks into a panel indexed by state and
    calendar month, yielding the seasonal surge signal. Features describe the
    month/state context (cyclical season, year, state, multistate share) so the
    model learns *when and where* surges occur — the proactive, time-aware axis.

    The panel is dense over the observed (state, month) grid: months with zero
    Salmonella outbreaks are included as count=0 so the regressor sees troughs,
    not just peaks.
    """
    if nors is None:
        nors = ingest_local.load_nors(foodborne_only=True)

    df = nors.copy()
    df["_dt"] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df["_is_salm"] = _salmonella_mask(df).astype(int)
    df["_state"] = df["ExposureState"].astype("string")
    df["_multistate"] = pd.to_numeric(df.get("MultiStateExposure"), errors="coerce")
    df = df.dropna(subset=["_dt", "_state"])
    df["_year"] = df["_dt"].dt.year
    df["_month"] = df["_dt"].dt.month

    # Salmonella outbreak count per (state, year, month).
    salm = df[df["_is_salm"] == 1]
    counts = (
        salm.groupby(["_state", "_year", "_month"]).size().rename("surge_count")
    )

    # Build the full dense grid over observed states x the full observed
    # year-month span, so zero-count periods are represented.
    states = df["_state"].dropna().unique()
    periods = (
        df[["_year", "_month"]].drop_duplicates().sort_values(["_year", "_month"])
    )
    grid = (
        pd.MultiIndex.from_product(
            [states, list(map(tuple, periods.to_numpy()))],
            names=["_state", "_ym"],
        )
        .to_frame(index=False)
    )
    grid["_year"] = grid["_ym"].apply(lambda t: int(t[0]))
    grid["_month"] = grid["_ym"].apply(lambda t: int(t[1]))
    grid = grid.drop(columns="_ym")

    panel = grid.merge(
        counts.reset_index(), on=["_state", "_year", "_month"], how="left"
    )
    panel["surge_count"] = panel["surge_count"].fillna(0).astype(float)

    # Multistate share per (state, year, month) as context (mean of flag).
    ms = (
        df.groupby(["_state", "_year", "_month"])["_multistate"]
        .mean()
        .rename("multistate_share")
        .reset_index()
    )
    panel = panel.merge(ms, on=["_state", "_year", "_month"], how="left")
    panel["multistate_share"] = panel["multistate_share"].fillna(0.0)

    # Features: cyclical month, year, state, multistate share. A synthetic
    # mid-month date lets the Step 2 DateFeatureEngineer reuse its season logic.
    X = pd.DataFrame()
    X[DATE_COLUMN] = pd.to_datetime(
        dict(year=panel["_year"], month=panel["_month"], day=15), errors="coerce"
    )
    X["ExposureState"] = panel["_state"].astype("string")
    X["multistate_share"] = panel["multistate_share"]
    X["year"] = panel["_year"].astype(float)

    y = panel["surge_count"]
    return TargetBundle(
        name="monthly_surge",
        task="regression",
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        positive_rate=None,
        description=("Salmonella outbreak count per state x year-month "
                     "(dense panel incl. zero-count periods)."),
    )


def build_all_targets(nors: pd.DataFrame | None = None) -> dict[str, TargetBundle]:
    """Build all three targets from a single NORS load."""
    if nors is None:
        nors = ingest_local.load_nors(foodborne_only=True)
    return {
        "is_salmonella": build_is_salmonella(nors),
        "is_poultry_salmonella": build_is_poultry_salmonella(nors),
        "monthly_surge": build_monthly_surge(nors),
    }


__all__ = [
    "TargetBundle",
    "TaskType",
    "build_is_salmonella",
    "build_is_poultry_salmonella",
    "build_monthly_surge",
    "build_all_targets",
    "DATE_COLUMN",
]
