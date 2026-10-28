"""Figure generation — visualize the mathematics of every pipeline step.

Renders one or more matplotlib PNGs per step into artifacts/figures/, so the
math behind each step (joins, cyclical season encoding, OOF calibration, the
Bayesian prior->posterior update, the expected-loss decision rule, danger
weighting, and BEAM validation) is visible as pictures, not just numbers.

Run:  python -m src.figures
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless, file-only rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import (bayes, config, ingest_local, loss, modeling, routing,
               severity, targets)

FIG_DIR = config.PROJECT_ROOT / "artifacts" / "figures"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
          "Oct", "Nov", "Dec"]
ACCENT = "#d84315"
PURPLE = "#6a1b9a"
BLUE = "#1565c0"
GREY = "#9e9e9e"


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ===========================================================================
# STEP 1 — Secure ingestion
# ===========================================================================
def step1_figures(nors: pd.DataFrame, fsis) -> list[Path]:
    out = []

    # (a) FSIS form_id join provenance (primary-only / both / secondary-only).
    j = fsis.joined
    if "_merge" in j.columns:
        counts = j["_merge"].value_counts()
        label_map = {"left_only": "primary only", "right_only": "secondary only",
                     "both": "matched (both)"}
        labels = [label_map.get(k, k) for k in counts.index]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, counts.values, color=[GREY, ACCENT, BLUE][:len(counts)])
        for i, v in enumerate(counts.values):
            ax.text(i, v + 0.2, str(v), ha="center", fontweight="bold")
        ax.set_title("Step 1 · FSIS full outer join on form_id — provenance")
        ax.set_ylabel("rows")
        out.append(_save(fig, "step1_fsis_join_provenance.png"))

    # (b) NORS data coverage: foodborne outbreaks per year.
    dt = pd.to_datetime(nors["DateFirstIll"], errors="coerce")
    yr = dt.dt.year.value_counts().sort_index()
    salm = targets._salmonella_mask(nors)
    yr_salm = dt[salm].dt.year.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(yr.index, yr.values, color=GREY, label="all foodborne")
    ax.bar(yr_salm.index, yr_salm.reindex(yr.index, fill_value=0).values,
           color=ACCENT, label="Salmonella")
    ax.set_title("Step 1 · NORS foodborne outbreak coverage by year")
    ax.set_xlabel("year"); ax.set_ylabel("outbreaks"); ax.legend()
    out.append(_save(fig, "step1_nors_coverage.png"))
    return out


# ===========================================================================
# STEP 2 — Stateful preprocessing (cyclical season encoding)
# ===========================================================================
def step2_figures() -> list[Path]:
    out = []
    # The DateFeatureEngineer maps day-of-year -> (sin, cos) on a unit circle.
    doy = np.arange(1, 366)
    rad = 2 * np.pi * doy / 365.25
    sin_d, cos_d = np.sin(rad), np.cos(rad)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    # left: the two cyclical features over the year, summer shaded.
    ax1.plot(doy, sin_d, color=ACCENT, label="sin(day-of-year)")
    ax1.plot(doy, cos_d, color=BLUE, label="cos(day-of-year)")
    ax1.axvspan(152, 273, color="#ffcc80", alpha=0.4, label="summer (Jun-Sep)")
    ax1.set_title("Step 2 · Cyclical season encoding")
    ax1.set_xlabel("day of year"); ax1.set_ylabel("feature value"); ax1.legend(fontsize=8)
    # right: the unit circle it traces (why Dec 31 is adjacent to Jan 1).
    ax2.plot(cos_d, sin_d, color=PURPLE)
    for m_doy, lbl in [(1, "Jan 1"), (182, "Jul 1"), (91, "Apr 1"), (274, "Oct 1")]:
        r = 2 * np.pi * m_doy / 365.25
        ax2.scatter([np.cos(r)], [np.sin(r)], color=ACCENT, zorder=5)
        ax2.annotate(lbl, (np.cos(r), np.sin(r)), fontsize=8,
                     textcoords="offset points", xytext=(6, 6))
    ax2.set_aspect("equal"); ax2.set_title("…maps the calendar onto a circle")
    ax2.set_xlabel("cos"); ax2.set_ylabel("sin")
    out.append(_save(fig, "step2_cyclical_season_encoding.png"))
    return out


# ===========================================================================
# STEP 3 — OOF predictions + calibration
# ===========================================================================
def step3_figures(bundles: dict) -> list[Path]:
    out = []
    # Run OOF for the two classifiers; show OOF probability distribution by
    # true class + a reliability (calibration) curve.
    for name in ("is_salmonella", "is_poultry_salmonella"):
        b = bundles[name]
        res = modeling.run_oof(b, n_splits=5)
        p = res.oof_pred
        y = res.y_true

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        # (a) OOF probability histograms by true class.
        ax1.hist(p[y == 0], bins=30, alpha=0.6, color=GREY, label="true negative")
        ax1.hist(p[y == 1], bins=30, alpha=0.7, color=ACCENT, label="true positive")
        ax1.set_title(f"Step 3 · OOF probabilities — {name}")
        ax1.set_xlabel("OOF p(positive | x)"); ax1.set_ylabel("count")
        ax1.legend(fontsize=8); ax1.set_yscale("log")

        # (b) Calibration curve (reliability diagram).
        bins = np.linspace(0, 1, 11)
        idx = np.digitize(p, bins) - 1
        xs, ys = [], []
        for bb in range(10):
            m = idx == bb
            if m.sum() > 0:
                xs.append(p[m].mean()); ys.append(y[m].mean())
        ax2.plot([0, 1], [0, 1], "--", color=GREY, label="perfect")
        ax2.plot(xs, ys, "o-", color=BLUE, label="model")
        ax2.set_title(f"Reliability — AUC={res.metrics['roc_auc']:.3f}")
        ax2.set_xlabel("mean predicted prob"); ax2.set_ylabel("observed freq")
        ax2.legend(fontsize=8); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
        out.append(_save(fig, f"step3_oof_calibration_{name}.png"))
    return out


# ===========================================================================
# STEP 4 — Sequential Bayesian updating
# ===========================================================================
def step4_figures(priors, posteriors) -> list[Path]:
    out = []
    national = {m: priors[("__NATIONAL__", m)].prior_mean for m in range(1, 13)}

    # (a) Seasonal prior curve with summer band.
    fig, ax = plt.subplots(figsize=(8, 4.2))
    vals = [national[m] for m in range(1, 13)]
    ax.plot(range(1, 13), vals, "o-", color=ACCENT, linewidth=2)
    ax.axvspan(6, 9, color="#ffcc80", alpha=0.4, label="summer surge")
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTHS)
    ax.set_title("Step 4 · Bayesian prior — national seasonal Salmonella risk")
    ax.set_ylabel("prior risk  E[w]"); ax.legend()
    summer = np.mean([national[m] for m in (6, 7, 8, 9)])
    winter = np.mean([national[m] for m in (12, 1, 2, 3)])
    ax.annotate(f"summer/winter = {summer/winter:.1f}x",
                xy=(7, national[7]), xytext=(8.5, national[7] * 0.7),
                arrowprops=dict(arrowstyle="->", color=PURPLE), color=PURPLE)
    out.append(_save(fig, "step4_seasonal_prior.png"))

    # (b) Prior -> posterior Beta update for a representative cell.
    if posteriors:
        (st_, mo), post = max(posteriors.items(), key=lambda kv: kv[1].prior_mean)
        prior = bayes.get_prior(priors, st_, mo)
        grid = np.linspace(0, 1, 500)
        prior_pdf = stats.beta.pdf(grid, prior.alpha, prior.beta)
        post_pdf = stats.beta.pdf(grid, post.alpha, post.beta)
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.plot(grid, prior_pdf, color=GREY, linewidth=2,
                label=f"prior  Beta({prior.alpha:.1f}, {prior.beta:.1f})")
        ax.fill_between(grid, prior_pdf, color=GREY, alpha=0.2)
        ax.plot(grid, post_pdf, color=ACCENT, linewidth=2,
                label=f"posterior  Beta({post.alpha:.1f}, {post.beta:.1f})")
        ax.fill_between(grid, post_pdf, color=ACCENT, alpha=0.25)
        ax.axvline(prior.prior_mean, color=GREY, linestyle=":")
        ax.axvline(post.mean, color=ACCENT, linestyle=":")
        ax.set_title(f"Step 4 · Bayes update  p(w|D) ∝ p(D|w)p(w) — cell {st_}/{MONTHS[mo-1]}  "
                     f"(FSIS {post.evidence_k}/{post.evidence_n})")
        ax.set_xlabel("surge risk  w"); ax.set_ylabel("density"); ax.legend(fontsize=9)
        ax.set_xlim(0, min(1, prior.prior_mean * 4 + 0.1))
        out.append(_save(fig, "step4_bayes_update.png"))
    return out


# ===========================================================================
# STEP 5 — Expected loss & danger-weighted optimization
# ===========================================================================
def step5_figures(nors: pd.DataFrame) -> list[Path]:
    out = []
    L = loss.build_loss_matrix()

    # (a) Loss matrix heatmap.
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(L, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(loss.ACTION_NAMES)))
    ax.set_xticklabels(loss.ACTION_NAMES)
    ax.set_yticks(range(len(loss.CLASS_NAMES)))
    ax.set_yticklabels(loss.CLASS_NAMES)
    for k in range(L.shape[0]):
        for jj in range(L.shape[1]):
            ax.text(jj, k, f"{L[k, jj]:.0f}", ha="center", va="center",
                    color="white" if L[k, jj] > L.max() / 2 else "black",
                    fontweight="bold")
    ax.set_title("Step 5 · Loss matrix  L[k,j]  (true class × action)")
    ax.set_xlabel("action j"); ax.set_ylabel("true class k")
    fig.colorbar(im, ax=ax, label="loss")
    out.append(_save(fig, "step5_loss_matrix.png"))

    # (b) Expected-loss decision boundary as a function of p(salmonella).
    p_grid = np.linspace(0, 1, 300)
    p_sev_given = 0.347  # base rate from Step 5 driver
    P = loss.three_class_posterior(p_grid, p_sev_given)
    el = P @ L  # (n, 2)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(p_grid, el[:, loss.ACTION_NO_ALERT], color=BLUE, label="E[loss | no_alert]")
    ax.plot(p_grid, el[:, loss.ACTION_ALERT], color=ACCENT, label="E[loss | alert]")
    # decision boundary where they cross.
    cross = p_grid[np.argmin(np.abs(el[:, 0] - el[:, 1]))]
    ax.axvline(cross, color=PURPLE, linestyle="--",
               label=f"alert threshold ≈ {cross:.2f}")
    ax.fill_between(p_grid, 0, el.max(), where=(el[:, 1] < el[:, 0]),
                    color="#ffcdd2", alpha=0.4)
    ax.set_title("Step 5 · Expected loss  Σ_k L[k,j] p(C_k|x)  → Bayes-optimal action")
    ax.set_xlabel("p(Salmonella | x)"); ax.set_ylabel("expected loss")
    ax.legend(fontsize=9); ax.set_ylim(0, el.max() * 1.05)
    out.append(_save(fig, "step5_expected_loss_decision.png"))

    # (c) Severe-serotype recall: naive vs expected-loss vs danger-weighted.
    # Recompute the three recalls quickly (small, deterministic).
    sev_frame = severity.build_severity_frame(nors)
    b = targets.build_is_salmonella(nors)
    is_salm = sev_frame["is_salmonella"].to_numpy().astype(bool)
    sero = sev_frame["serotype"].astype("string")
    is_sev = sero.str.contains("|".join(loss.SEVERE_SEROTYPES), case=False,
                               na=False).to_numpy()
    p_sev_g = float(is_sev[is_salm].mean()) if is_salm.any() else 0.0

    base = modeling.run_oof(b, n_splits=5)
    P0 = loss.three_class_posterior(base.oof_pred, p_sev_g)
    dec0 = loss.expected_loss(P0, L).decision == loss.ACTION_ALERT
    naive = base.oof_pred >= 0.5

    w = loss.normalize_weights(loss.danger_weights(sev_frame))
    wres = modeling.run_oof(b, n_splits=5, sample_weight=w)
    Pw = loss.three_class_posterior(wres.oof_pred, p_sev_g)
    decw = loss.expected_loss(Pw, L).decision == loss.ACTION_ALERT

    def rec(mask):
        return float(mask[is_sev].mean()) if is_sev.any() else float("nan")
    recalls = [rec(naive), rec(dec0), rec(decw)]
    labels = ["naive\n(p≥0.5)", "expected-loss\nrule", "danger-weighted\n+ exp-loss"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, recalls, color=[GREY, BLUE, ACCENT])
    for bar, v in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.1%}",
                ha="center", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_title("Step 5 · Severe-serotype recall (Enteritidis/Typhimurium)")
    ax.set_ylabel("recall of true severe outbreaks")
    out.append(_save(fig, "step5_severe_recall.png"))
    return out


# ===========================================================================
# STEP 6 — BEAM validation + NARMS weighting
# ===========================================================================
def step6_figures(nors: pd.DataFrame, bundles: dict) -> list[Path]:
    out = []
    from . import narms, validation, ingest_api

    # (a) Predicted risk vs BEAM Salmonella share — scatter + correlation.
    res = modeling.run_oof(bundles["is_salmonella"], n_splits=5)
    try:
        beam = ingest_api.fetch_cdc_socrata("beam_report_data")
        vr = validation.validate_against_beam(res.oof_pred, nors["ExposureState"],
                                              beam_report=beam)
        m = vr.merged
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(m["predicted_risk"], m["beam_salmonella_share"],
                   color=ACCENT, alpha=0.7)
        for r in m.itertuples(index=False):
            ax.annotate(r.state, (r.predicted_risk, r.beam_salmonella_share),
                        fontsize=7, alpha=0.6)
        # regression line.
        if len(m) > 2:
            z = np.polyfit(m["predicted_risk"], m["beam_salmonella_share"], 1)
            xs = np.linspace(m["predicted_risk"].min(), m["predicted_risk"].max(), 50)
            ax.plot(xs, np.polyval(z, xs), color=PURPLE,
                    label=f"fit (Spearman={vr.spearman:.2f})")
            ax.legend()
        ax.set_title("Step 6 · BEAM ground-truth validation (rate vs. rate)")
        ax.set_xlabel("model predicted Salmonella risk")
        ax.set_ylabel("BEAM observed Salmonella share")
        out.append(_save(fig, "step6_beam_validation_scatter.png"))
        beam_ok = True
    except Exception as exc:  # network/BEAM failure shouldn't kill the batch
        print(f"   [step6 scatter skipped: {type(exc).__name__}: {exc}]")
        beam = None
        beam_ok = False

    # (b) NARMS danger weights by serotype group.
    sev_frame = severity.build_severity_frame(nors)
    w5 = loss.normalize_weights(loss.danger_weights(sev_frame))
    sero = sev_frame["serotype"].astype("string")
    ci_amr = np.zeros(len(sev_frame))  # keep figure offline-safe
    is_inf = sero.str.contains("Infantis", case=False, na=False)
    mult = narms.narms_multiplier(serotype=sero, is_mdr=is_inf.astype(int),
                                  is_dsc_cipro=np.zeros(len(sev_frame)),
                                  ci_amr_pct=ci_amr)
    w_final = narms.apply_narms_to_weights(w5, mult)

    groups = {
        "non-Salmonella": (sev_frame["is_salmonella"].to_numpy() == 0),
        "generic\nSalmonella": (sev_frame["is_salmonella"].to_numpy() == 1)
            & ~is_inf.to_numpy()
            & ~sero.str.contains("Enteritidis|Typhimurium", case=False, na=False).to_numpy(),
        "Enteritidis/\nTyphimurium": sero.str.contains(
            "Enteritidis|Typhimurium", case=False, na=False).to_numpy(),
        "Infantis\n(MDR driver)": is_inf.to_numpy(),
    }
    means = [w_final[mask].mean() if mask.any() else 0 for mask in groups.values()]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(list(groups.keys()), means,
                  color=[GREY, BLUE, "#ef6c00", ACCENT])
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05, f"{v:.2f}",
                ha="center", fontweight="bold")
    ax.set_title("Step 6 · Dynamic NARMS danger weights by serotype group")
    ax.set_ylabel("mean per-sample weight  w_i")
    out.append(_save(fig, "step6_narms_weights.png"))
    return out


# ===========================================================================
# Driver
# ===========================================================================
def generate_all() -> list[Path]:
    print("Loading data...")
    nors = ingest_local.load_nors(foodborne_only=True)
    fsis = ingest_local.load_fsis()
    bundles = targets.build_all_targets(nors)
    priors = bayes.build_priors()
    posteriors = bayes.posteriors_for_fsis(poultry_only=True)

    all_figs: list[Path] = []
    print("Step 1 figures..."); all_figs += step1_figures(nors, fsis)
    print("Step 2 figures..."); all_figs += step2_figures()
    print("Step 3 figures..."); all_figs += step3_figures(bundles)
    print("Step 4 figures..."); all_figs += step4_figures(priors, posteriors)
    print("Step 5 figures..."); all_figs += step5_figures(nors)
    print("Step 6 figures..."); all_figs += step6_figures(nors, bundles)

    print(f"\nGenerated {len(all_figs)} figures in "
          f"{FIG_DIR.relative_to(config.PROJECT_ROOT)}:")
    for p in all_figs:
        print(f"  {p.name}")
    return all_figs


if __name__ == "__main__":
    generate_all()
