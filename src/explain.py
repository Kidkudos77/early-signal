"""Step 3 — SHAP explainability.

Generates SHapley Additive exPlanations for every observation, so each alert
can be traced to the feature interactions that drove it (the brief's
"individual force plots for every observation").

Consistency with OOF: each held-out row is explained by the SAME fold model
that produced its OOF prediction. We run shap.TreeExplainer on each fold's
fitted XGBoost booster, transform that fold's held-out rows through the fold's
fitted preprocessor, and stitch the per-fold SHAP values back into one
full-length array aligned to the original row order.

Outputs:
  * oof_shap_values(...)  -> (shap_values, transformed_X, feature_names)
                             aligned 1:1 with the OOF predictions
  * save_force_plot(...)  -> write a single observation's force plot (HTML)
  * save_all_force_plots(...) -> write force plots for many/all observations
  * save_summary_plot(...) -> global SHAP summary (beeswarm) as PNG

Force plots are saved as self-contained HTML (shap's native interactive form);
a static matplotlib force plot is also supported for PNG export.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .modeling import OOFResult
from .targets import TargetBundle


def _extract_booster_and_pre(pipe):
    """Pull the fitted preprocessor and XGBoost model out of a fold pipeline."""
    pre = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]
    return pre, model


def oof_shap_values(
    bundle: TargetBundle,
    result: OOFResult,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute per-observation SHAP values aligned to the OOF rows.

    Each fold's held-out rows are explained with that fold's model. Returns:
      shap_values   : (n_rows, n_features) in the transformed feature space
      X_transformed : (n_rows, n_features) the transformed inputs (for plots)
      feature_names : list of transformed feature names
    Row order matches bundle.X / result.oof_pred.
    """
    import shap

    X = bundle.X.reset_index(drop=True)
    n = len(X)

    # Each fold's preprocessor is refit on fold-train, so its one-hot vocabulary
    # (and hence transformed width/column order) can differ between folds. We
    # therefore align every fold's output to the UNION of feature names across
    # folds, filling absent columns with 0. This yields a single consistent
    # (n_rows, n_union_features) matrix while keeping the per-fold model that
    # produced each row's OOF prediction.
    per_fold_names: list[list[str]] = []
    for pipe in result.fold_models:
        pre, _ = _extract_booster_and_pre(pipe)
        per_fold_names.append(list(pre.get_feature_names_out()))

    # Stable union: preserve first-seen order across folds.
    seen: dict[str, None] = {}
    for names in per_fold_names:
        for nm in names:
            seen.setdefault(nm, None)
    feature_names = list(seen.keys())
    n_feat = len(feature_names)
    col_pos = {nm: i for i, nm in enumerate(feature_names)}

    shap_full = np.full((n, n_feat), np.nan, dtype="float64")
    xt_full = np.zeros((n, n_feat), dtype="float64")

    for pipe, te_idx, names in zip(
        result.fold_models, result.fold_test_idx, per_fold_names
    ):
        pre, model = _extract_booster_and_pre(pipe)
        X_te = X.iloc[te_idx]
        Xt = np.asarray(pre.transform(X_te), dtype="float64")
        fold_nf = len(names)

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xt)
        # Binary XGBClassifier returns a single (n, f) array of log-odds SHAP;
        # some shap versions return a list — normalize to 2D.
        if isinstance(sv, list):
            sv = sv[-1]
        sv = np.asarray(sv, dtype="float64")
        # XGBoost SHAP can carry a trailing bias column (n_features + 1); drop.
        if sv.shape[1] == fold_nf + 1:
            sv = sv[:, :fold_nf]

        # Scatter this fold's columns into their position in the union matrix.
        target_cols = np.array([col_pos[nm] for nm in names])
        # Initialize SHAP rows for this fold to 0 (absent features contribute 0).
        shap_full[np.ix_(te_idx, target_cols)] = sv
        # Any union column not present in this fold stays NaN->set 0 for SHAP.
        present = set(target_cols.tolist())
        absent = [j for j in range(n_feat) if j not in present]
        if absent:
            shap_full[np.ix_(te_idx, absent)] = 0.0
        xt_full[np.ix_(te_idx, target_cols)] = Xt

    return shap_full, xt_full, feature_names


def _expected_value_for(result: OOFResult) -> float:
    """Mean base value across folds (TreeExplainer expected_value)."""
    import shap

    bases = []
    for pipe in result.fold_models:
        _, model = _extract_booster_and_pre(pipe)
        ev = shap.TreeExplainer(model).expected_value
        ev = float(np.ravel(ev)[-1])
        bases.append(ev)
    return float(np.mean(bases)) if bases else 0.0


def save_force_plot(
    row_pos: int,
    shap_values: np.ndarray,
    X_transformed: np.ndarray,
    feature_names: list[str],
    expected_value: float,
    out_path: Path,
) -> Path:
    """Save one observation's SHAP force plot as self-contained HTML."""
    import shap

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fp = shap.force_plot(
        base_value=expected_value,
        shap_values=shap_values[row_pos, :],
        features=X_transformed[row_pos, :],
        feature_names=feature_names,
    )
    shap.save_html(str(out_path), fp)
    return out_path


def save_all_force_plots(
    bundle: TargetBundle,
    result: OOFResult,
    shap_values: np.ndarray,
    X_transformed: np.ndarray,
    feature_names: list[str],
    out_dir: Path,
    *,
    max_plots: int | None = None,
) -> int:
    """Write per-observation force plots (HTML). Returns count written.

    By default writes one per observation (the brief: "for every observation").
    `max_plots` caps the count for quick runs / demos.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_value = _expected_value_for(result)

    n = shap_values.shape[0]
    limit = n if max_plots is None else min(max_plots, n)
    written = 0
    for i in range(limit):
        if np.isnan(shap_values[i]).all():
            continue  # row never got a fold assignment (shouldn't happen)
        save_force_plot(
            i, shap_values, X_transformed, feature_names, expected_value,
            out_dir / f"force_{bundle.name}_row{i:05d}.html",
        )
        written += 1
    return written


def save_summary_plot(
    shap_values: np.ndarray,
    X_transformed: np.ndarray,
    feature_names: list[str],
    out_path: Path,
    *,
    max_display: int = 20,
) -> Path:
    """Save a global SHAP summary (beeswarm) plot as PNG."""
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    import shap

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Drop any all-NaN rows defensively.
    good = ~np.isnan(shap_values).all(axis=1)
    shap.summary_plot(
        shap_values[good],
        X_transformed[good],
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close("all")
    return out_path


def shap_contributions_frame(
    bundle: TargetBundle,
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Per-observation top-contributor table (tabular form of the force plots).

    For each row, record the signed SHAP contribution of every feature. This is
    the machine-readable companion to the HTML force plots — useful for the
    Step 7 dashboard "why this alert fired" panel.
    """
    sv = pd.DataFrame(shap_values, columns=feature_names)
    sv.insert(0, "_row", range(len(sv)))
    return sv


__all__ = [
    "oof_shap_values",
    "save_force_plot",
    "save_all_force_plots",
    "save_summary_plot",
    "shap_contributions_frame",
]
