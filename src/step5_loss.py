"""Step 5 driver — Expected loss & danger-weighted optimization.

Run from the project root:
    python -m src.step5_loss

Demonstrates and verifies the three directives:
  1. Build the asymmetric loss matrix L_kj (missing severe serotypes is costly).
  2. Expected loss sum_k L_kj p(C_k|x) -> Bayes-optimal alert decisions using
     Step 3 OOF probabilities, and compare total realized loss of the
     expected-loss rule vs. a naive 0.5-threshold rule.
  3. Danger-weighted training: build per-sample weights w_i (severe serotypes +
     severity), refit the Salmonella classifier with those weights via OOF, and
     show that SEVERE-serotype recall improves vs. the unweighted baseline.
  Also verifies the literal weighted-MSE custom objective (grad/hess) matches
  the analytic form.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import bayes, config, ingest_local, loss, modeling, severity, targets

ARTIFACT_DIR = config.PROJECT_ROOT / "artifacts" / "loss"


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _recall_on_severe(y_true_severe: np.ndarray, decision_alert: np.ndarray) -> float:
    """Fraction of truly-severe instances that received an alert."""
    sev = y_true_severe.astype(bool)
    if sev.sum() == 0:
        return float("nan")
    return float(decision_alert[sev].mean())


def run() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Data substrate ---------------------------------------------------
    nors = ingest_local.load_nors(foodborne_only=True)
    sev_frame = severity.build_severity_frame(nors)
    bundle = targets.build_is_salmonella(nors)  # aligned row order with sev_frame

    # True 3-class label k per row (clean / mild / severe).
    is_salm = sev_frame["is_salmonella"].to_numpy().astype(bool)
    sero = sev_frame["serotype"].astype("string")
    is_severe_sero = sero.str.contains("|".join(loss.SEVERE_SEROTYPES),
                                       case=False, na=False).to_numpy()
    true_class = np.where(~is_salm, loss.CLASS_CLEAN,
                          np.where(is_severe_sero, loss.CLASS_SEVERE, loss.CLASS_MILD))
    true_severe = (true_class == loss.CLASS_SEVERE)

    # --- 1. Loss matrix ---------------------------------------------------
    section("1. Loss matrix L_kj (rows=true class, cols=action)")
    L = loss.build_loss_matrix()
    Ltab = pd.DataFrame(L, index=loss.CLASS_NAMES, columns=loss.ACTION_NAMES)
    print(Ltab.to_string())
    print(f"\n   miss severe = {L[loss.CLASS_SEVERE, loss.ACTION_NO_ALERT]:.0f}x  "
          f"false alarm = {L[loss.CLASS_CLEAN, loss.ACTION_ALERT]:.0f}x")
    Ltab.to_csv(ARTIFACT_DIR / "loss_matrix.csv")

    # --- 2a. Baseline OOF probabilities (unweighted) ----------------------
    section("2. Expected-loss decisions from OOF posteriors")
    print("   running baseline OOF (unweighted) for p(salmonella|x)...")
    base_res = modeling.run_oof(bundle, n_splits=5)
    p_salm = base_res.oof_pred

    # Conditional severe probability p(severe|salmonella): base rate among
    # Salmonella outbreaks (data-grounded, not peeking at the row's own label).
    p_sev_given = float(is_severe_sero[is_salm].mean()) if is_salm.any() else 0.0
    P = loss.three_class_posterior(p_salm, p_sev_given)

    elr = loss.expected_loss(P, L)
    decision_alert = (elr.decision == loss.ACTION_ALERT).astype(int)

    # Naive 0.5-threshold rule for comparison.
    naive_alert = (p_salm >= 0.5).astype(int)

    # Realized loss of each rule = sum over rows of L[true_class, chosen_action].
    def realized_loss(actions: np.ndarray) -> float:
        return float(L[true_class, actions].sum())

    print(f"   p(severe | salmonella) base rate = {p_sev_given:.3f}")
    print(f"   expected-loss rule: alerts on {decision_alert.mean():.1%} of rows")
    print(f"   naive 0.5 rule     : alerts on {naive_alert.mean():.1%} of rows")
    print(f"   realized total loss  (expected-loss rule) = {realized_loss(decision_alert):.1f}")
    print(f"   realized total loss  (naive 0.5 rule)     = {realized_loss(naive_alert):.1f}")
    print(f"   severe-serotype recall (expected-loss)    = {_recall_on_severe(true_severe, decision_alert):.3f}")
    print(f"   severe-serotype recall (naive 0.5)        = {_recall_on_severe(true_severe, naive_alert):.3f}")

    # --- 3. Danger-weighted training --------------------------------------
    section("3. Danger-weighted OOF (per-sample weights in the gradient)")
    w = loss.danger_weights(sev_frame)
    w = loss.normalize_weights(w)
    print(f"   danger weights: mean={w.mean():.3f} "
          f"severe-serotype mean weight={w[true_severe].mean():.3f} "
          f"clean mean weight={w[true_class == loss.CLASS_CLEAN].mean():.3f}")

    print("   running danger-weighted OOF...")
    w_res = modeling.run_oof(bundle, n_splits=5, sample_weight=w)
    p_salm_w = w_res.oof_pred
    P_w = loss.three_class_posterior(p_salm_w, p_sev_given)
    elr_w = loss.expected_loss(P_w, L)
    decision_alert_w = (elr_w.decision == loss.ACTION_ALERT).astype(int)

    rec_base = _recall_on_severe(true_severe, decision_alert)
    rec_w = _recall_on_severe(true_severe, decision_alert_w)
    loss_base = realized_loss(decision_alert)
    loss_w = realized_loss(decision_alert_w)
    print(f"   severe recall : unweighted={rec_base:.3f} -> weighted={rec_w:.3f}")
    print(f"   realized loss : unweighted={loss_base:.1f} -> weighted={loss_w:.1f}")
    print(f"   mean p(salm) on severe rows: unweighted={p_salm[true_severe].mean():.3f} "
          f"-> weighted={p_salm_w[true_severe].mean():.3f}")

    # --- 4. Verify weighted-MSE objective grad/hess -----------------------
    section("4. Verify danger-weighted MSE custom objective (grad/hess)")
    rng = np.random.default_rng(0)
    yhat = rng.random(8)
    ytrue = rng.integers(0, 2, 8).astype("float64")
    wdemo = rng.random(8) + 0.5

    class _D:
        def get_label(self): return ytrue
    obj = loss.make_weighted_mse_objective(wdemo)
    grad, hess = obj(yhat, _D())
    grad_expected = 2.0 * wdemo * (yhat - ytrue)
    hess_expected = 2.0 * wdemo
    ok_obj = np.allclose(grad, grad_expected) and np.allclose(hess, hess_expected)
    print(f"   grad matches 2*w*(yhat-y): {np.allclose(grad, grad_expected)}")
    print(f"   hess matches 2*w        : {np.allclose(hess, hess_expected)}")
    print(f"   [{'PASS' if ok_obj else 'FAIL'}] weighted-MSE objective")

    # --- 5. Persist per-instance decisions --------------------------------
    section("5. Persist expected-loss decision table")
    out = pd.DataFrame({
        "true_class": [loss.CLASS_NAMES[k] for k in true_class],
        "p_salmonella_unw": p_salm,
        "p_salmonella_wtd": p_salm_w,
        "danger_weight": w,
        "exp_loss_no_alert": elr.expected_loss[:, loss.ACTION_NO_ALERT],
        "exp_loss_alert": elr.expected_loss[:, loss.ACTION_ALERT],
        "decision_unw": np.where(decision_alert == 1, "alert", "no_alert"),
        "decision_wtd": np.where(decision_alert_w == 1, "alert", "no_alert"),
    })
    out_path = ARTIFACT_DIR / "expected_loss_decisions.csv"
    out.to_csv(out_path, index=False)
    print(f"   written: {out_path.relative_to(config.PROJECT_ROOT)} ({len(out)} rows)")

    # Verification gates.
    improved_recall = (np.isnan(rec_base) or np.isnan(rec_w) or rec_w >= rec_base - 1e-9)
    all_ok = ok_obj and improved_recall
    if not improved_recall:
        print(f"\n   NOTE: weighted severe recall ({rec_w:.3f}) < baseline "
              f"({rec_base:.3f}); weighting did not help on this split.")
    section("Step 5 complete" if all_ok else "Step 5 completed WITH WARNINGS")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(run())
