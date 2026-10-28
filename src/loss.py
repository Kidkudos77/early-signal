"""Step 5 — Expected loss & danger-weighted optimization.

Three directives from the brief, implemented here:

1. LOSS MATRIX  L_kj
   The cost of taking decision j when the true class is k. We frame the
   operational decision as a binary ALERT choice (j in {no_alert, alert}) and
   the true class k over a danger-graded space {clean, mild_salmonella,
   severe_salmonella}. Missing a severe Salmonella serotype (Enteritidis,
   Typhimurium) — i.e. choosing no_alert when k=severe — carries the heaviest
   penalty; a false alarm (alert when clean) carries a small penalty.

2. EXPECTED LOSS  sum_k L_kj p(C_k | x)
   Using the model's posterior class probabilities p(C_k|x), compute the
   expected loss of each decision j and choose the Bayes-optimal action
   (argmin_j). This turns calibrated probabilities into danger-aware alerts.

3. DANGER-WEIGHTED GRADIENT DESCENT  theta <- theta - eta * grad_theta MSE(theta)
   The per-sample danger weight w_i scales each instance's gradient, so the
   optimizer is pulled harder toward getting dangerous instances right. We
   provide (a) weights for XGBoost's sample_weight path and (b) a literal
   weighted-MSE custom objective returning grad/hess for the exact formula.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Class space (true k) — danger-graded.
CLASS_CLEAN = 0          # not a Salmonella event
CLASS_MILD = 1           # Salmonella, non-severe serotype
CLASS_SEVERE = 2         # Salmonella, severe serotype (Enteritidis/Typhimurium)
CLASS_NAMES = ("clean", "mild_salmonella", "severe_salmonella")

# Decision space (action j).
ACTION_NO_ALERT = 0
ACTION_ALERT = 1
ACTION_NAMES = ("no_alert", "alert")

# Serotypes treated as "severe human-illness" (brief). Case-insensitive.
SEVERE_SEROTYPES = ("Enteritidis", "Typhimurium")


# ---------------------------------------------------------------------------
# Loss matrix L_kj
# ---------------------------------------------------------------------------
def build_loss_matrix(
    *,
    miss_severe: float = 20.0,   # k=severe, j=no_alert  (worst: missed severe)
    miss_mild: float = 5.0,      # k=mild,   j=no_alert  (missed mild Salmonella)
    false_alarm: float = 1.0,    # k=clean,  j=alert     (cost of a false alert)
    correct: float = 0.0,        # correct decisions
    alert_mild: float = 0.0,     # alerting on true mild/severe is correct -> 0
) -> np.ndarray:
    """Return the 3x2 loss matrix L[k, j].

    Rows k = (clean, mild, severe); columns j = (no_alert, alert). The
    asymmetry encodes the brief: a missed severe serotype costs `miss_severe`
    (default 20x a false alarm), a missed mild case costs `miss_mild`, and a
    false alarm costs `false_alarm`. Correct actions cost 0.
    """
    L = np.zeros((3, 2), dtype="float64")
    # True clean:
    L[CLASS_CLEAN, ACTION_NO_ALERT] = correct          # correct dismissal
    L[CLASS_CLEAN, ACTION_ALERT] = false_alarm         # false alarm
    # True mild Salmonella:
    L[CLASS_MILD, ACTION_NO_ALERT] = miss_mild         # missed mild
    L[CLASS_MILD, ACTION_ALERT] = alert_mild           # correct alert
    # True severe Salmonella:
    L[CLASS_SEVERE, ACTION_NO_ALERT] = miss_severe     # missed severe (worst)
    L[CLASS_SEVERE, ACTION_ALERT] = alert_mild         # correct alert
    return L


# ---------------------------------------------------------------------------
# Posterior class probabilities p(C_k | x)
# ---------------------------------------------------------------------------
def three_class_posterior(
    p_salmonella: np.ndarray,
    p_severe_given_salm: np.ndarray | float,
) -> np.ndarray:
    """Build p(C_k|x) over (clean, mild, severe) from binary signals.

    We decompose: p(severe) = p(salm) * p(severe | salm),
                  p(mild)   = p(salm) * (1 - p(severe | salm)),
                  p(clean)  = 1 - p(salm).
    `p_salmonella` is the OOF Salmonella probability from Step 3.
    `p_severe_given_salm` is the conditional severe-serotype probability (an
    array per row, or a scalar base rate when per-row isn't available).
    """
    p_salm = np.clip(np.asarray(p_salmonella, dtype="float64"), 0, 1)
    if np.isscalar(p_severe_given_salm):
        p_sev_g = np.full_like(p_salm, float(p_severe_given_salm))
    else:
        p_sev_g = np.clip(np.asarray(p_severe_given_salm, dtype="float64"), 0, 1)

    p_severe = p_salm * p_sev_g
    p_mild = p_salm * (1 - p_sev_g)
    p_clean = 1 - p_salm
    P = np.column_stack([p_clean, p_mild, p_severe])
    # Numerical safety: renormalize rows to sum to 1.
    P = P / P.sum(axis=1, keepdims=True)
    return P


# ---------------------------------------------------------------------------
# Expected loss  sum_k L_kj p(C_k|x)  and Bayes-optimal decision
# ---------------------------------------------------------------------------
@dataclass
class ExpectedLossResult:
    expected_loss: np.ndarray    # (n, n_actions): expected loss of each action
    decision: np.ndarray         # (n,): argmin action index
    posterior: np.ndarray        # (n, n_classes): p(C_k|x) used


def expected_loss(P: np.ndarray, L: np.ndarray) -> ExpectedLossResult:
    """Compute expected loss of each action and the Bayes-optimal decision.

    E[loss | x, action j] = sum_k L[k, j] * p(C_k | x).
    Returns per-row expected loss for every action and the argmin decision.
    """
    P = np.asarray(P, dtype="float64")          # (n, K)
    L = np.asarray(L, dtype="float64")          # (K, J)
    if P.shape[1] != L.shape[0]:
        raise ValueError(f"P has {P.shape[1]} classes but L has {L.shape[0]} rows.")
    # (n, K) @ (K, J) -> (n, J)
    exp_loss = P @ L
    decision = exp_loss.argmin(axis=1)
    return ExpectedLossResult(expected_loss=exp_loss, decision=decision, posterior=P)


# ---------------------------------------------------------------------------
# Per-sample danger weights  w_i  (feed XGBoost sample_weight / gradient)
# ---------------------------------------------------------------------------
def danger_weights(
    severity_frame: pd.DataFrame,
    *,
    base: float = 1.0,
    severity_coef: float = 2.0,
    severe_serotype_boost: float = 3.0,
) -> np.ndarray:
    """Per-sample weights w_i emphasizing dangerous instances in training.

    w_i = base
        + severity_coef * severity_i                 (continuous severity)
        + (serotype base danger - 1)                 (serotype class danger)
        + severe_serotype_boost on severe serotypes  (brief's explicit misses)

    `severity_frame` has columns: is_salmonella, serotype, severity, base_danger
    (from src.severity.build_severity_frame). Non-Salmonella rows keep ~base
    weight. Weights are returned unnormalized (XGBoost handles scale); callers
    may normalize to mean 1 if desired.
    """
    sf = severity_frame
    sev = pd.to_numeric(sf["severity"], errors="coerce").fillna(0).to_numpy()
    base_danger = pd.to_numeric(sf["base_danger"], errors="coerce").fillna(1.0).to_numpy()
    sero = sf["serotype"].astype("string")

    is_severe = sero.str.contains("|".join(SEVERE_SEROTYPES), case=False, na=False).to_numpy()

    w = (
        base
        + severity_coef * sev
        + (base_danger - 1.0)
        + severe_serotype_boost * is_severe.astype("float64")
    )
    return np.clip(w, 1e-3, None)


def normalize_weights(w: np.ndarray) -> np.ndarray:
    """Scale weights to mean 1 (keeps effective learning rate comparable)."""
    w = np.asarray(w, dtype="float64")
    m = w.mean()
    return w / m if m > 0 else w


# ---------------------------------------------------------------------------
# Danger-weighted MSE custom objective  (literal theta <- theta - eta grad MSE)
# ---------------------------------------------------------------------------
def make_weighted_mse_objective(weights: np.ndarray):
    """Return an XGBoost custom objective for danger-weighted MSE.

    Implements L(theta) = (1/N) sum_i w_i (yhat_i - y_i)^2, whose gradient and
    Hessian w.r.t. the raw prediction are:
        grad_i = 2 w_i (yhat_i - y_i)
        hess_i = 2 w_i
    XGBoost's booster then performs theta <- theta - eta * grad (the Newton/
    gradient step in the brief's formula), so the per-sample danger weight w_i
    scales each instance's pull on the parameters exactly as intended.

    Signature matches XGBoost's custom objective: fn(y_pred, dtrain) ->
    (grad, hess).
    """
    w = np.asarray(weights, dtype="float64")

    def objective(y_pred: np.ndarray, dtrain):
        y_true = dtrain.get_label()
        wloc = w
        if len(wloc) != len(y_true):
            # Fallback if sizes mismatch (e.g., subsampled) — use uniform.
            wloc = np.ones_like(y_true)
        residual = y_pred - y_true
        grad = 2.0 * wloc * residual
        hess = 2.0 * wloc
        return grad, hess

    return objective


__all__ = [
    "CLASS_NAMES", "ACTION_NAMES", "SEVERE_SEROTYPES",
    "CLASS_CLEAN", "CLASS_MILD", "CLASS_SEVERE",
    "ACTION_NO_ALERT", "ACTION_ALERT",
    "build_loss_matrix",
    "three_class_posterior",
    "ExpectedLossResult",
    "expected_loss",
    "danger_weights",
    "normalize_weights",
    "make_weighted_mse_objective",
]
