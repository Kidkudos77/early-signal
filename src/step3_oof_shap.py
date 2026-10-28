"""Step 3 driver — Individual-level OOF predictions + SHAP explainability.

Run from the project root:
    python -m src.step3_oof_shap [--force-plots N] [--all-force-plots]

For each of the three NORS targets (is_salmonella, is_poultry_salmonella,
monthly_surge):
  1. Build the leakage-safe (X, y).
  2. Generate OOF predictions for EVERY training instance (StratifiedKFold /
     KFold), so each row has an unbiased prediction from a model that did not
     train on it.
  3. Write the per-instance OOF table to artifacts/oof/.
  4. Compute SHAP values per observation (each row explained by its own fold
     model) and write a global summary plot plus per-observation force plots.

Artifacts land under artifacts/ (git-ignored by the data/cache rules).
By default a capped number of force plots are written for speed; pass
--all-force-plots to emit one per observation (the brief's literal ask).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, explain, modeling, targets

ARTIFACT_ROOT = config.PROJECT_ROOT / "artifacts"
OOF_DIR = ARTIFACT_ROOT / "oof"
SHAP_DIR = ARTIFACT_ROOT / "shap"


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _fmt_metrics(task: str, m: dict) -> str:
    if task == "classification":
        return (f"ROC-AUC={m['roc_auc']:.3f}  AP={m['average_precision']:.3f}  "
                f"Brier={m['brier']:.3f}  pos_rate={m['positive_rate']:.3f}  n={m['n']}")
    return (f"MAE={m['mae']:.3f}  R2={m['r2']:.3f}  "
            f"Poisson-dev={m['poisson_deviance']:.3f}  mean_count={m['mean_count']:.3f}  "
            f"n={m['n']}")


def run(force_plots: int = 25, all_force_plots: bool = False) -> int:
    section("Build all three NORS targets")
    bundles = targets.build_all_targets()
    for b in bundles.values():
        print(f"   {b!r}")
        print(f"      {b.description}")

    OOF_DIR.mkdir(parents=True, exist_ok=True)
    SHAP_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for name, bundle in bundles.items():
        section(f"Target: {name}  ({bundle.task})")

        # --- OOF cross-validation ---
        print("   running OOF cross-validation (5-fold)...")
        result = modeling.run_oof(bundle, n_splits=5)
        print(f"   OOF metrics : {_fmt_metrics(bundle.task, result.metrics)}")

        # coverage: every row must have an OOF prediction
        covered = int(np.sum(~np.isnan(result.oof_pred)))
        print(f"   OOF coverage: {covered}/{len(bundle.y)} rows "
              f"({'ALL' if covered == len(bundle.y) else 'INCOMPLETE'})")

        # --- per-instance OOF table ---
        oof_tbl = modeling.oof_frame(bundle, result)
        oof_path = OOF_DIR / f"oof_{name}.csv"
        oof_tbl.to_csv(oof_path, index=False)
        print(f"   OOF table   : {oof_path.relative_to(config.PROJECT_ROOT)} "
              f"({oof_tbl.shape[0]}x{oof_tbl.shape[1]})")

        # --- SHAP ---
        print("   computing SHAP values (per-fold TreeExplainer)...")
        shap_vals, Xt, feat_names = explain.oof_shap_values(bundle, result)
        print(f"   SHAP shape  : {shap_vals.shape}  features={len(feat_names)}")

        # global summary plot
        summ_path = SHAP_DIR / f"summary_{name}.png"
        explain.save_summary_plot(shap_vals, Xt, feat_names, summ_path)
        print(f"   summary plot: {summ_path.relative_to(config.PROJECT_ROOT)}")

        # per-observation force plots
        fp_dir = SHAP_DIR / name
        n_plots = None if all_force_plots else force_plots
        written = explain.save_all_force_plots(
            bundle, result, shap_vals, Xt, feat_names, fp_dir, max_plots=n_plots,
        )
        print(f"   force plots : {written} written -> "
              f"{fp_dir.relative_to(config.PROJECT_ROOT)}/"
              f"{' (ALL observations)' if all_force_plots else f' (capped at {force_plots})'}")

        # machine-readable contributions table
        contrib = explain.shap_contributions_frame(bundle, shap_vals, feat_names)
        contrib_path = SHAP_DIR / f"contributions_{name}.csv"
        contrib.to_csv(contrib_path, index=False)
        print(f"   contrib tbl : {contrib_path.relative_to(config.PROJECT_ROOT)} "
              f"({contrib.shape[0]}x{contrib.shape[1]})")

        summary_rows.append({
            "target": name, "task": bundle.task, "rows": len(bundle.y),
            "oof_covered": covered, "force_plots": written,
            **{k: v for k, v in result.metrics.items() if k != "n"},
        })

    section("Step 3 summary")
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))
    summary.to_csv(ARTIFACT_ROOT / "step3_summary.csv", index=False)

    # Verification: every target achieved full OOF coverage.
    all_covered = all(r["oof_covered"] == r["rows"] for r in summary_rows)
    section("Step 3 complete" if all_covered else "Step 3 completed WITH GAPS")
    return 0 if all_covered else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Step 3 OOF + SHAP driver.")
    p.add_argument("--force-plots", type=int, default=25,
                   help="Max per-observation force plots per target (default 25).")
    p.add_argument("--all-force-plots", action="store_true",
                   help="Emit a force plot for EVERY observation (slow).")
    args = p.parse_args(argv)
    return run(force_plots=args.force_plots, all_force_plots=args.all_force_plots)


if __name__ == "__main__":
    sys.exit(main())
