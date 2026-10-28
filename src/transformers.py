"""Step 2 — Custom domain transformers.

All transformers inherit from scikit-learn's BaseEstimator and TransformerMixin
(per the brief) so they compose inside Pipelines / ColumnTransformers and
honor fit/transform semantics. They are STATEFUL where appropriate: any value
learned from data (e.g. category vocabularies, medians) is learned in fit()
on the TRAINING data only and reused in transform() — this is what prevents
leakage from the test set.

Transformers provided:
  * NullStringCleaner      - turn FSIS literal 'NULL'/'' strings into real NaN
  * DateFeatureEngineer    - derive time-aware features (month, season,
                             day-of-year sin/cos) from collection_date; this is
                             the seasonal axis the surge model needs
  * AMRDangerFlagger       - collapse the per-drug *_resistant_ast / _wgs
                             columns into interpretable danger features
                             (resistant drug count, DSC/ciprofloxacin flag,
                             ceftriaxone flag, MDR flag) used by Steps 5-6
  * SerotypeConcernFlagger - flag high-concern serotypes (Infantis, Enteritidis,
                             Typhimurium) from salmonella_serotype

Each is written to be safe on the sparse 20-row SAMPLE (all-NULL columns, tiny
n) and correct on the full export.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# FSIS uses these literal strings for "no value"; treat as missing.
_NULL_TOKENS = {"NULL", "null", "", "NA", "N/A", "None", "none"}

# Critically important drugs (NARMS) -> column stems in the FSIS schema.
# Used by AMRDangerFlagger to build danger features for Steps 5-6.
CRITICAL_DRUG_STEMS = {
    "ciprofloxacin": "salmonella_ciprofloxacin_resistant_ast",   # DSC proxy
    "ceftriaxone": "salmonella_ceftriaxone_resistant_ast",
    "azithromycin": "salmonella_azithromycin_resistant_ast",
    "trim_sulfa": "salmonella_trimethoprim_sulfamethoxazole_resistant_ast",
    "meropenem": "salmonella_meropenem_resistant_ast",
}

# All per-drug AST resistance columns (for the resistant-count / MDR feature).
_ALL_AST_DRUGS = (
    "ampicillin", "amoxicillin_clavulanic_acid", "ceftriaxone", "azithromycin",
    "chloramphenicol", "ciprofloxacin", "colistin",
    "trimethoprim_sulfamethoxazole", "cefoxitin", "gentamicin", "kanamycin",
    "meropenem", "nalidixic_acid", "sulfisoxazole", "streptomycin",
    "tetracycline", "ceftiofur",
)

# High-human-illness / high-concern serotypes called out in the brief + NARMS.
CONCERN_SEROTYPES = ("Infantis", "Enteritidis", "Typhimurium")

# Values that count as "resistant" in the *_resistant_ast / _wgs fields.
_RESISTANT_TRUE = {"R", "RESISTANT", "TRUE", "YES", "1", "Y"}


def _is_resistant(series: pd.Series) -> pd.Series:
    """Map a resistance column to 1.0 (resistant) / 0.0 (not) / NaN (unknown)."""
    s = series.astype("string").str.strip().str.upper()
    s = s.where(~s.isin({t.upper() for t in _NULL_TOKENS}), other=pd.NA)
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out[s.isin(_RESISTANT_TRUE)] = 1.0
    out[s.notna() & ~s.isin(_RESISTANT_TRUE)] = 0.0
    return out


class NullStringCleaner(BaseEstimator, TransformerMixin):
    """Replace FSIS literal null tokens ('NULL', '', 'NA', ...) with np.nan.

    Stateless (nothing learned), but implemented as a proper transformer so it
    can sit at the head of a Pipeline. Operates on all columns of the input
    DataFrame and returns a DataFrame (so downstream column selection by name
    keeps working).
    """

    def __init__(self, tokens: set[str] | None = None):
        self.tokens = tokens if tokens is not None else set(_NULL_TOKENS)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        # Convert pandas nullable dtypes (string/Int/boolean) to plain object/
        # float so downstream sklearn imputers see numpy np.nan, not pandas
        # pd.NA — the latter raises "boolean value of NA is ambiguous" inside
        # SimpleImputer's object-NaN check.
        for col in df.columns:
            if isinstance(df[col].dtype, pd.StringDtype) or str(df[col].dtype) in (
                "string", "boolean",
            ) or pd.api.types.is_extension_array_dtype(df[col].dtype):
                df[col] = df[col].astype(object)
        df = df.replace(list(self.tokens), np.nan)
        # Replace any residual pandas NA with numpy nan for sklearn safety.
        return df.where(df.notna(), other=np.nan)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features) if input_features is not None else None


class DateFeatureEngineer(BaseEstimator, TransformerMixin):
    """Derive time-aware features from a date column (the seasonal surge axis).

    Produces: <prefix>_year, _month, _doy, _sin_doy, _cos_doy, _is_summer.
    The sin/cos day-of-year encoding gives the model a smooth cyclical
    representation of season (summer Salmonella surge). Stateless beyond
    parsing; emits numeric columns suitable for the numeric pipeline.
    """

    def __init__(self, date_column: str = "collection_date", prefix: str = "date"):
        self.date_column = date_column
        self.prefix = prefix

    def fit(self, X, y=None):
        self._feature_names = [
            f"{self.prefix}_year",
            f"{self.prefix}_month",
            f"{self.prefix}_doy",
            f"{self.prefix}_sin_doy",
            f"{self.prefix}_cos_doy",
            f"{self.prefix}_is_summer",
        ]
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        if self.date_column in df.columns:
            dt = pd.to_datetime(df[self.date_column], errors="coerce")
        else:
            dt = pd.Series(pd.NaT, index=df.index)

        doy = dt.dt.dayofyear
        out = pd.DataFrame(index=df.index)
        out[f"{self.prefix}_year"] = dt.dt.year
        out[f"{self.prefix}_month"] = dt.dt.month
        out[f"{self.prefix}_doy"] = doy
        # Cyclical season encoding (period = 365.25 days).
        radians = 2.0 * np.pi * (doy / 365.25)
        out[f"{self.prefix}_sin_doy"] = np.sin(radians)
        out[f"{self.prefix}_cos_doy"] = np.cos(radians)
        # Summer flag (Jun-Sep), when poultry Salmonella typically surges.
        out[f"{self.prefix}_is_summer"] = dt.dt.month.isin([6, 7, 8, 9]).astype("float64")
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self._feature_names)


class AMRDangerFlagger(BaseEstimator, TransformerMixin):
    """Collapse per-drug resistance columns into danger features for Steps 5-6.

    Outputs:
      amr_resistant_count : number of drug classes flagged resistant (AST)
      amr_is_mdr          : 1.0 if resistant to >= 3 classes (NARMS MDR def.)
      amr_dsc_cipro       : 1.0 if ciprofloxacin-nonsusceptible (DSC proxy)
      amr_ceftriaxone     : 1.0 if ceftriaxone-resistant
      amr_azithromycin    : 1.0 if azithromycin-resistant
      amr_meropenem       : 1.0 if meropenem-resistant (carbapenem; rare/severe)

    All derived from *_resistant_ast columns. Rows with no resistance data at
    all (entirely NULL, as in the SAMPLE poultry subset) yield NaN count and
    0.0 flags, which the downstream imputer handles.
    """

    MDR_THRESHOLD = 3  # NARMS: MDR = resistant to 3+ classes

    def fit(self, X, y=None):
        self._feature_names = [
            "amr_resistant_count",
            "amr_is_mdr",
            "amr_dsc_cipro",
            "amr_ceftriaxone",
            "amr_azithromycin",
            "amr_meropenem",
        ]
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        n = len(df)
        out = pd.DataFrame(index=df.index)

        # Resistant count across all AST drug columns present. A row with no
        # resistance evidence (all NULL, as in the SAMPLE) counts as 0 observed
        # resistant classes — a definite value, not NaN — so this feature is
        # always emitted with stable width across datasets.
        res_frame = pd.DataFrame(index=df.index)
        for drug in _ALL_AST_DRUGS:
            col = f"salmonella_{drug}_resistant_ast"
            if col in df.columns:
                res_frame[drug] = _is_resistant(df[col])
        if res_frame.shape[1] > 0:
            # sum() with skipna treats NaN as 0; a fully-NaN row -> 0.0.
            count = res_frame.sum(axis=1, skipna=True).astype("float64")
        else:
            count = pd.Series(0.0, index=df.index)
        out["amr_resistant_count"] = count
        out["amr_is_mdr"] = (count >= self.MDR_THRESHOLD).astype("float64")

        # Specific critical-drug flags (0.0 when absent/unknown).
        def flag(stem_key: str) -> pd.Series:
            col = CRITICAL_DRUG_STEMS[stem_key]
            if col in df.columns:
                return _is_resistant(df[col]).fillna(0.0)
            return pd.Series(0.0, index=df.index)

        out["amr_dsc_cipro"] = flag("ciprofloxacin")
        out["amr_ceftriaxone"] = flag("ceftriaxone")
        out["amr_azithromycin"] = flag("azithromycin")
        out["amr_meropenem"] = flag("meropenem")
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self._feature_names)


class SerotypeConcernFlagger(BaseEstimator, TransformerMixin):
    """One-hot-ish flags for high-concern Salmonella serotypes.

    Outputs a binary column per serotype in CONCERN_SEROTYPES plus
    serotype_is_concern (any of them). Case-insensitive substring match on
    salmonella_serotype. NULL/absent -> all zeros.
    """

    def __init__(self, serotype_column: str = "salmonella_serotype",
                 concern=CONCERN_SEROTYPES):
        self.serotype_column = serotype_column
        self.concern = tuple(concern)

    def fit(self, X, y=None):
        self._feature_names = (
            [f"serotype_{s.lower()}" for s in self.concern] + ["serotype_is_concern"]
        )
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        out = pd.DataFrame(index=df.index)
        if self.serotype_column in df.columns:
            s = df[self.serotype_column].astype("string").str.strip()
            s = s.where(~s.isin(list(_NULL_TOKENS)), other=pd.NA)
        else:
            s = pd.Series(pd.NA, index=df.index, dtype="string")

        any_flag = pd.Series(0.0, index=df.index)
        for sero in self.concern:
            col = f"serotype_{sero.lower()}"
            flag = s.str.contains(sero, case=False, na=False).astype("float64")
            out[col] = flag
            any_flag = np.maximum(any_flag, flag)
        out["serotype_is_concern"] = any_flag
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self._feature_names)


__all__ = [
    "NullStringCleaner",
    "DateFeatureEngineer",
    "AMRDangerFlagger",
    "SerotypeConcernFlagger",
    "CRITICAL_DRUG_STEMS",
    "CONCERN_SEROTYPES",
]
