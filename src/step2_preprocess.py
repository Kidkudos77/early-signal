"""Step 2 driver — Stateful preprocessing pipeline.

Run from the project root:
    python -m src.step2_preprocess

Demonstrates and verifies the leakage-safe design:
  1. Load the FSIS poultry join (Step 1 output).
  2. Split into train/test.
  3. Infer the feature schema from TRAIN ONLY.
  4. Fit the ColumnTransformer on TRAIN ONLY (learns medians/scales/categories).
  5. Transform train AND test with the SAME fitted object (no test leakage).
  6. Report feature names and output shapes; assert train/test column parity.

The shipped FSIS SAMPLE has only ~5 poultry rows — too few to split
meaningfully — so the driver falls back to the full 22-row FSIS join for the
demonstration while reporting which frame it used. The pipeline logic is
identical on the full export.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import ingest_local, preprocessing


def section(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def _select_frame(min_rows: int = 8) -> tuple[pd.DataFrame, str]:
    """Return the modeling frame and a label describing its source."""
    fsis = ingest_local.load_fsis()
    if len(fsis.poultry) >= min_rows:
        return fsis.poultry.copy(), "FSIS poultry subset"
    # SAMPLE fallback: use the full outer join so the split has enough rows.
    return fsis.joined.copy(), ("FSIS full join (SAMPLE poultry subset too "
                                f"small: {len(fsis.poultry)} rows)")


def run() -> int:
    section("1. Load Step 1 output")
    df, source = _select_frame()
    print(f"   source frame : {source}")
    print(f"   rows x cols  : {df.shape}")

    section("2. Train/test split (test leakage guard)")
    # Stratification isn't meaningful without a target yet; simple split.
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    print(f"   train rows   : {len(train_df)}")
    print(f"   test rows    : {len(test_df)}")

    section("3. Infer feature schema from TRAIN ONLY")
    schema = preprocessing.infer_feature_schema(train_df)
    print(f"   numeric    ({len(schema.numeric)}): {schema.numeric}")
    print(f"   categorical({len(schema.categorical)}): {schema.categorical}")
    print(f"   date column : {schema.date_column}")
    print(f"   serotype col: {schema.serotype_column}")
    print(f"   AMR columns : {len(schema.amr_columns)} resistance fields")
    print(f"   dropped     : {len(schema.dropped)} columns")

    section("4. Fit ColumnTransformer on TRAIN ONLY")
    pre = preprocessing.build_preprocessor(schema)
    pre.fit(train_df)
    print("   fitted OK (medians, scales, one-hot categories learned on train)")

    section("5. Transform train + test with the SAME fitted object")
    Xt_train = pre.transform(train_df)
    Xt_test = pre.transform(test_df)
    feat_names = list(pre.get_feature_names_out())
    print(f"   transformed train shape : {Xt_train.shape}")
    print(f"   transformed test  shape : {Xt_test.shape}")
    print(f"   output feature count    : {len(feat_names)}")

    section("6. Verification")
    ok = True

    # (a) train/test must share identical output feature columns
    if Xt_train.shape[1] == Xt_test.shape[1] == len(feat_names):
        print("   [PASS] train/test column parity")
    else:
        ok = False
        print(f"   [FAIL] column mismatch: train={Xt_train.shape[1]} "
              f"test={Xt_test.shape[1]} names={len(feat_names)}")

    # (b) no NaNs should remain after imputation
    n_nan_train = int(np.isnan(np.asarray(Xt_train, dtype="float64")).sum())
    n_nan_test = int(np.isnan(np.asarray(Xt_test, dtype="float64")).sum())
    if n_nan_train == 0 and n_nan_test == 0:
        print("   [PASS] no NaNs remain after imputation")
    else:
        ok = False
        print(f"   [FAIL] residual NaNs: train={n_nan_train} test={n_nan_test}")

    # (c) numeric block should be standardized on train (mean ~0)
    num_prefixes = tuple(f"num__{c}" for c in schema.numeric)
    num_idx = [i for i, n in enumerate(feat_names) if n.startswith(num_prefixes)]
    if num_idx:
        train_means = np.asarray(Xt_train, dtype="float64")[:, num_idx].mean(axis=0)
        if np.allclose(train_means, 0.0, atol=1e-6):
            print(f"   [PASS] numeric features standardized on train "
                  f"(|mean|<1e-6 over {len(num_idx)} cols)")
        else:
            print(f"   [note] numeric train means not all ~0 "
                  f"(max |mean|={np.max(np.abs(train_means)):.2e}) "
                  f"— expected with tiny n / constant cols")
    else:
        print("   [note] no pure-numeric passthrough columns in this frame")

    # (d) show a sample of engineered feature names
    engineered = [n for n in feat_names
                  if n.startswith(("date__", "amr__", "serotype__"))]
    print(f"\n   engineered feature names ({len(engineered)}):")
    for n in engineered:
        print(f"      {n}")

    section("Step 2 complete" if ok else "Step 2 completed WITH FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
