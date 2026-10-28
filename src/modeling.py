"""Step 3 — Out-of-fold (OOF) cross-validation engine.

Produces an UNBIASED prediction for every training instance: each row is
predicted by a model that never saw it in training (it was in that row's
held-out fold). This is the brief's "individual-level tracking" requirement —
not just aggregate metrics, but a per-instance OOF prediction for all rows.

Design:
  * The estimator is wrapped in a Pipeline [Step 2 preprocessor -> XGBoost],
    and the WHOLE pipeline is re-fit inside each fold. The preprocessor
    (imputers, scalers, one-hot vocab) is therefore fit on fold-train only —
    no leakage from fold-test into preprocessing.
  * Classification: StratifiedKFold (preserves class balance per fold) ->
    OOF class probabilities. Imbalance handled via scale_pos_weight.
  * Regression: KFold -> OOF point predictions.
  * The fitted per-fold pipelines are retained so Step 3's SHAP stage can
    explain each held-out row with the model that produced its OOF prediction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from . import preprocessing
from .targets import TargetBundle


# ---------------------------------------------------------------------------
# Estimator factories
# ---------------------------------------------------------------------------
def make_classifier(scale_pos_weight: float = 1.0) -> XGBClassifier:
    """XGBoost classifier tuned for modest, imbalanced tabular data.

    scale_pos_weight rebalances the positive class (ratio neg/pos). Trees are
    shallow and few to avoid overfitting the ~8k-row NORS data; this is a
    sensible default, not a tuned optimum.
    """
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=0,
        random_state=42,
    )


def make_regressor() -> XGBRegressor:
    """XGBoost count regressor for the monthly-surge target.

    Poisson objective suits non-negative count targets (outbreaks per month).
    """
    return XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="count:poisson",
        eval_metric="poisson-nloglik",
        tree_method="hist",
        n_jobs=0,
        random_state=42,
    )


def build_model_pipeline(bundle: TargetBundle) -> Pipeline:
    """Assemble [Step 2 preprocessor -> XGBoost] for a target.

    The preprocessor is inferred from the bundle's X schema. It is UNFITTED;
    the caller (OOF loop) fits it per fold on fold-train only.
    """
    schema = preprocessing.infer_feature_schema(bundle.X)
    pre = preprocessing.build_preprocessor(schema)

    if bundle.task == "classification":
        pos = max(int(bundle.y.sum()), 1)
        neg = max(int((bundle.y == 0).sum()), 1)
        est = make_classifier(scale_pos_weight=neg / pos)
    else:
        est = make_regressor()

    return Pipeline(steps=[("preprocess", pre), ("model", est)])


# ---------------------------------------------------------------------------
# OOF result container
# ---------------------------------------------------------------------------
@dataclass
class OOFResult:
    """Out-of-fold predictions and the per-fold fitted pipelines."""

    name: str
    task: str
    oof_pred: np.ndarray          # OOF probability (clf) or point pred (reg)
    y_true: np.ndarray
    fold_index: np.ndarray        # which fold each row was held out in
    fold_models: list[Pipeline] = field(default_factory=list)
    fold_test_idx: list[np.ndarray] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# OOF cross-validation
# ---------------------------------------------------------------------------
def run_oof(
    bundle: TargetBundle,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    sample_weight: np.ndarray | None = None,
) -> OOFResult:
    """Generate OOF predictions for every row of `bundle`.

    Returns an OOFResult carrying the per-row OOF prediction, the fold each row
    belonged to, the fitted per-fold pipelines (for SHAP), and summary metrics.

    Parameters
    ----------
    sample_weight:
        Optional per-row danger weights (Step 5). When given, each fold's model
        is fit with the fold-restricted slice of these weights via XGBoost's
        sample_weight, so dangerous instances pull harder on the gradient.
        Weights are sliced per fold, so no information leaks across folds.
    """
    X = bundle.X.reset_index(drop=True)
    y = bundle.y.reset_index(drop=True).to_numpy()
    n = len(X)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype="float64")
        if len(sample_weight) != n:
            raise ValueError(
                f"sample_weight length {len(sample_weight)} != n rows {n}."
            )

    oof_pred = np.full(n, np.nan, dtype="float64")
    fold_index = np.full(n, -1, dtype="int64")
    fold_models: list[Pipeline] = []
    fold_test_idx: list[np.ndarray] = []

    if bundle.task == "classification":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=random_state)
        split_iter = splitter.split(X, y)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = splitter.split(X)

    for fold_id, (tr_idx, te_idx) in enumerate(split_iter):
        pipe = build_model_pipeline(bundle)
        X_tr = X.iloc[tr_idx]
        y_tr = y[tr_idx]
        if sample_weight is not None:
            # Pass fold-train weights to the final 'model' step of the pipeline.
            pipe.fit(X_tr, y_tr, model__sample_weight=sample_weight[tr_idx])
        else:
            pipe.fit(X_tr, y_tr)

        if bundle.task == "classification":
            proba = pipe.predict_proba(X.iloc[te_idx])[:, 1]
            oof_pred[te_idx] = proba
        else:
            pred = pipe.predict(X.iloc[te_idx])
            oof_pred[te_idx] = pred

        fold_index[te_idx] = fold_id
        fold_models.append(pipe)
        fold_test_idx.append(te_idx)

    metrics = _compute_metrics(bundle.task, y, oof_pred)
    return OOFResult(
        name=bundle.name,
        task=bundle.task,
        oof_pred=oof_pred,
        y_true=y,
        fold_index=fold_index,
        fold_models=fold_models,
        fold_test_idx=fold_test_idx,
        metrics=metrics,
    )


def _compute_metrics(task: str, y_true: np.ndarray, pred: np.ndarray) -> dict:
    """Summary metrics over the OOF predictions."""
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        mean_absolute_error,
        mean_poisson_deviance,
        r2_score,
        roc_auc_score,
    )

    out: dict = {}
    if task == "classification":
        # Guard against degenerate single-class folds.
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, pred))
        except ValueError:
            out["roc_auc"] = float("nan")
        try:
            out["average_precision"] = float(average_precision_score(y_true, pred))
        except ValueError:
            out["average_precision"] = float("nan")
        out["brier"] = float(brier_score_loss(y_true, pred))
        out["positive_rate"] = float(np.mean(y_true))
        out["n"] = int(len(y_true))
    else:
        out["mae"] = float(mean_absolute_error(y_true, pred))
        out["r2"] = float(r2_score(y_true, pred))
        # Poisson deviance needs strictly positive predictions.
        safe_pred = np.clip(pred, 1e-6, None)
        try:
            out["poisson_deviance"] = float(mean_poisson_deviance(y_true, safe_pred))
        except ValueError:
            out["poisson_deviance"] = float("nan")
        out["mean_count"] = float(np.mean(y_true))
        out["n"] = int(len(y_true))
    return out


def oof_frame(bundle: TargetBundle, result: OOFResult) -> pd.DataFrame:
    """Return a per-instance OOF table (the individual-level tracking output).

    One row per training instance, carrying its features' index, the fold it was
    held out in, the true label, and its unbiased OOF prediction.
    """
    df = bundle.X.reset_index(drop=True).copy()
    df["_fold"] = result.fold_index
    df["_y_true"] = result.y_true
    pred_col = "oof_proba" if bundle.task == "classification" else "oof_pred"
    df[pred_col] = result.oof_pred
    return df


__all__ = [
    "make_classifier",
    "make_regressor",
    "build_model_pipeline",
    "OOFResult",
    "run_oof",
    "oof_frame",
]
