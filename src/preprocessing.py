"""Step 2 — Stateful preprocessing pipeline.

Builds a scikit-learn ColumnTransformer that strictly isolates transformations
so they can be fit on TRAINING data only (preventing leakage from the test
set). Per the brief:

  * Numerical features  -> SimpleImputer(strategy="median") then StandardScaler()
  * Categorical features -> OneHotEncoder() (dummy attributes)
  * Custom combinations -> the BaseEstimator/TransformerMixin transformers in
                           src.transformers (date/season, AMR danger, serotype)

The whole thing is a single fitted object. Call build_preprocessor() to get an
unfitted ColumnTransformer, then fit it on X_train only:

    pre = build_preprocessor(schema)
    pre.fit(X_train)                 # learns medians, scales, categories here
    Xt_train = pre.transform(X_train)
    Xt_test  = pre.transform(X_test)  # reuses TRAIN-learned state — no leakage
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from .transformers import (
    AMRDangerFlagger,
    DateFeatureEngineer,
    NullStringCleaner,
    SerotypeConcernFlagger,
)

# Columns that are identifiers / leakage risks / pure noise — excluded from the
# model matrix. form_id & sample numbers are keys; accession/allele codes are
# unique per isolate; *_canceled flags and the raw date (we engineer from it).
DEFAULT_DROP_COLUMNS = (
    "form_id",
    "sample_number",
    "secondary_sample_number",
    "tertiary_sample_number",
    "establishment_id",
    "establishment_name",
    "establishment_number",
    "pathogen_number",
    "fsis_number",
    "_merge",
    "salmonella_allele_code",
    "salmonella_allele_code_version",
    "salmonella_allele_code_date",
    "salmonella_sra_accession_number",
    "salmonella_bio_sample_accession_number",
    "salmonella_bio_project_number",
    "salmonella_multi_locus_sequence",
    "salmonella_primary_pfge_pattern",
    "salmonella_secondary_pfge_pattern",
    "lm_listeria_allele_code",
    "lm_listeria_allele_code_version",
    "lm_listeria_allele_code_date",
    "lm_listeria_sra_accession_number",
    "lm_listeria_bio_sample_accession_number",
    "lm_listeria_bio_project_number",
    "lm_listeria_multi_locus_sequence",
    "lm_listeria_primary_pfge_pattern",
    "lm_listeria_secondary_pfge_pattern",
)

# The raw column consumed by the date engineer (dropped from passthrough since
# it is transformed into cyclical season features instead).
DATE_COLUMN = "collection_date"


@dataclass
class FeatureSchema:
    """Resolved feature roles for the ColumnTransformer.

    numeric      : columns routed to median-impute + standard-scale
    categorical  : columns routed to one-hot encoding
    date_column  : column consumed by DateFeatureEngineer (or None)
    amr_columns  : per-drug resistance columns consumed by AMRDangerFlagger
    serotype_column : serotype column consumed by SerotypeConcernFlagger
    dropped      : columns excluded from the matrix
    """

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    date_column: str | None = None
    amr_columns: list[str] = field(default_factory=list)
    serotype_column: str | None = None
    dropped: list[str] = field(default_factory=list)


# Column names recognized as the engineered date source, in priority order.
# collection_date = FSIS; DateFirstIll = NORS. Any datetime64 column is also
# auto-detected regardless of name.
CANDIDATE_DATE_COLUMNS = ("collection_date", "DateFirstIll")


def infer_feature_schema(
    df: pd.DataFrame,
    *,
    drop_columns: tuple[str, ...] = DEFAULT_DROP_COLUMNS,
    max_cardinality: int = 60,
    numeric_coerce_threshold: float = 0.6,
    date_column: str | None = None,
) -> FeatureSchema:
    """Classify each column of `df` into a modeling role.

    Decisions (after cleaning literal NULLs):
      * a column is NUMERIC if >= numeric_coerce_threshold of its non-null
        values parse as numbers;
      * otherwise CATEGORICAL if its cardinality <= max_cardinality;
      * very high-cardinality text (free-text product names beyond the cap) is
        dropped to avoid an explosive one-hot;
      * all-empty columns are dropped;
      * the date, AMR, and serotype columns are routed to custom transformers.

    Inference should be done on the TRAINING frame so the schema itself does
    not peek at the test set.
    """
    cleaned = df.replace(list({"NULL", "null", "", "NA", "N/A", "None", "none"}), pd.NA)

    schema = FeatureSchema(date_column=None, serotype_column=None)
    drop_set = set(drop_columns)

    amr_cols = [c for c in df.columns
                if c.endswith("_resistant_ast") or c.endswith("_resistant_wgs")]
    schema.amr_columns = amr_cols
    amr_set = set(amr_cols)

    # Resolve the date column: explicit arg > known name > any datetime64 col.
    if date_column and date_column in df.columns:
        schema.date_column = date_column
    else:
        for cand in CANDIDATE_DATE_COLUMNS:
            if cand in df.columns:
                schema.date_column = cand
                break
        if schema.date_column is None:
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    schema.date_column = col
                    break
    if "salmonella_serotype" in df.columns:
        schema.serotype_column = "salmonella_serotype"

    for col in df.columns:
        if col in drop_set or col in amr_set:
            schema.dropped.append(col) if col in drop_set else None
            continue
        if col == schema.date_column or col == schema.serotype_column:
            # Consumed by custom transformers; not routed to num/cat blocks.
            continue
        if col.endswith("_operator") or col.endswith("_mic"):
            # MIC raw values / operators are redundant with the AST flags and
            # mostly empty in the sample; drop to keep the matrix lean.
            schema.dropped.append(col)
            continue

        # Never route a datetime column through the numeric imputer — it would
        # collide with float columns in the shared branch. Any stray datetime
        # (not chosen as the date source) is dropped rather than misclassified.
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            schema.dropped.append(col)
            continue

        s = cleaned[col]
        nonnull = int(s.notna().sum())
        if nonnull == 0:
            schema.dropped.append(col)
            continue

        coerced = pd.to_numeric(s, errors="coerce")
        if coerced.notna().sum() >= max(1, int(numeric_coerce_threshold * nonnull)):
            schema.numeric.append(col)
        else:
            card = int(s.nunique(dropna=True))
            if card <= max_cardinality:
                schema.categorical.append(col)
            else:
                schema.dropped.append(col)  # high-card free text

    return schema


def build_numeric_pipeline() -> Pipeline:
    """Numeric branch: median imputation then standardization (brief Step 2)."""
    return Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])


def build_categorical_pipeline() -> Pipeline:
    """Categorical branch: impute a constant 'missing' then one-hot encode.

    OneHotEncoder(handle_unknown="ignore") so categories unseen at fit time
    (in the test set or at inference) don't break transform — important given
    the tiny SAMPLE. sparse_output=False keeps the matrix dense for SHAP later.
    """
    return Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


def build_preprocessor(
    schema: FeatureSchema,
    *,
    include_custom: bool = True,
) -> ColumnTransformer:
    """Assemble the full ColumnTransformer from a resolved FeatureSchema.

    The transformer is UNFITTED. Fit it on X_train only. A NullStringCleaner is
    applied inside each branch that needs it so the literal 'NULL' tokens are
    converted to real NaN before imputation/encoding.
    """
    transformers: list[tuple] = []

    if schema.numeric:
        num_pipe = Pipeline(steps=[
            ("nulls", NullStringCleaner()),
            ("numeric", build_numeric_pipeline()),
        ])
        transformers.append(("num", num_pipe, schema.numeric))

    if schema.categorical:
        cat_pipe = Pipeline(steps=[
            ("nulls", NullStringCleaner()),
            ("categorical", build_categorical_pipeline()),
        ])
        transformers.append(("cat", cat_pipe, schema.categorical))

    if include_custom:
        # Date / season features (numeric output -> impute+scale).
        if schema.date_column:
            date_pipe = Pipeline(steps=[
                ("engineer", DateFeatureEngineer(date_column=schema.date_column)),
                ("numeric", build_numeric_pipeline()),
            ])
            transformers.append(("date", date_pipe, [schema.date_column]))

        # AMR danger flags (numeric output -> impute; already 0/1 scaled-ish,
        # but we median-impute the count then scale for consistency).
        if schema.amr_columns:
            amr_pipe = Pipeline(steps=[
                ("flag", AMRDangerFlagger()),
                ("numeric", build_numeric_pipeline()),
            ])
            transformers.append(("amr", amr_pipe, schema.amr_columns))

        # Serotype concern flags (binary; pass through after construction).
        if schema.serotype_column:
            sero_pipe = Pipeline(steps=[
                ("flag", SerotypeConcernFlagger(serotype_column=schema.serotype_column)),
                ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ])
            transformers.append(("serotype", sero_pipe, [schema.serotype_column]))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_preprocessor_from_frame(
    df: pd.DataFrame, **kwargs
) -> tuple[ColumnTransformer, FeatureSchema]:
    """Convenience: infer schema from `df` and build the (unfitted) preprocessor.

    Pass the TRAINING frame so neither schema inference nor fitting sees test
    data.
    """
    schema = infer_feature_schema(df, **kwargs)
    pre = build_preprocessor(schema)
    return pre, schema


__all__ = [
    "FeatureSchema",
    "DEFAULT_DROP_COLUMNS",
    "DATE_COLUMN",
    "infer_feature_schema",
    "build_numeric_pipeline",
    "build_categorical_pipeline",
    "build_preprocessor",
    "build_preprocessor_from_frame",
]
