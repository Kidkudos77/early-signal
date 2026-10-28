"""Step 4 driver — Sequential Bayesian risk updating.

Run from the project root:
    python -m src.step4_bayes

Demonstrates and verifies:
  1. Prior construction from NORS: Beta priors per (state, month) encoding the
     seasonal + geographic Salmonella baseline.
  2. The seasonal signal is present in the priors (summer > winter).
  3. Bayesian update with FSIS supply-chain evidence -> posterior per cell.
  4. Sequential = pooled: applying FSIS batches one-by-one equals applying them
     together (a property of the Beta-Binomial conjugate model).
  5. Posterior summaries (mean surge probability, 95% credible interval,
     P(w > tau) alert signal) written to artifacts/bayes/.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import bayes, config

ARTIFACT_DIR = config.PROJECT_ROOT / "artifacts" / "bayes"


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def run(alert_tau: float = 0.20) -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Priors from NORS ---------------------------------------------
    section("1. Build NORS priors (Beta per state x month)")
    priors = bayes.build_priors()
    cell_priors = {k: v for k, v in priors.items() if k[0] != "__NATIONAL__"}
    print(f"   cells with priors : {len(cell_priors)}")
    print(f"   prior strength    : {bayes.DEFAULT_PRIOR_STRENGTH} pseudo-obs")

    # --- 2. Seasonal signal sanity check ---------------------------------
    section("2. Seasonal signal in priors (national-month baseline)")
    nat = {m: priors[("__NATIONAL__", m)].prior_mean for m in range(1, 13)}
    for m in range(1, 13):
        bar = "#" * int(round(nat[m] * 100))
        print(f"   month {m:2d}: prior risk {nat[m]:.3f}  {bar}")
    summer = np.mean([nat[m] for m in (6, 7, 8, 9)])
    winter = np.mean([nat[m] for m in (12, 1, 2, 3)])
    print(f"   summer(JJAS) mean risk = {summer:.3f} vs winter(DJFM) = {winter:.3f}")
    print(f"   -> seasonal surge {'CONFIRMED' if summer > winter else 'NOT seen'} "
          f"(summer/winter ratio = {summer/winter:.2f}x)")

    # --- 3. Update with FSIS evidence ------------------------------------
    section("3. Posterior = prior updated by FSIS supply-chain evidence")
    batches = bayes.fsis_batches_by_cell(poultry_only=True)
    if not batches:
        print("   no poultry FSIS cells with analyzed results; using full join")
        batches = bayes.fsis_batches_by_cell(poultry_only=False)
    print(f"   FSIS evidence cells: {len(batches)}")

    rows = []
    for (state, month), (k, n) in sorted(batches.items()):
        prior = bayes.get_prior(priors, state, month)
        post = bayes.update_posterior(bayes.init_posterior(prior), k, n)
        lo, hi = post.credible_interval(0.95)
        rows.append({
            "state": state, "month": month,
            "nors_total": prior.nors_total, "nors_salmonella": prior.nors_salmonella,
            "prior_mean": round(prior.prior_mean, 4),
            "fsis_k": k, "fsis_n": n,
            "posterior_mean": round(post.mean, 4),
            "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
            f"P(w>{alert_tau})": round(post.prob_exceeds(alert_tau), 4),
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    out_csv = ARTIFACT_DIR / "posteriors_fsis.csv"
    table.to_csv(out_csv, index=False)
    print(f"\n   written: {out_csv.relative_to(config.PROJECT_ROOT)}")

    # --- 4. Sequential == pooled verification ----------------------------
    section("4. Verify sequential updating = pooled (conjugacy property)")
    # Take one cell and split its evidence into two sequential batches.
    demo_cell = next(iter(batches))
    k, n = batches[demo_cell]
    prior = bayes.get_prior(priors, *demo_cell)
    # Split n into two batches (halves), preserving total k.
    n1 = n // 2
    k1 = min(k, n1)
    b1, b2 = (k1, n1), (k - k1, n - n1)

    pooled = bayes.update_posterior(bayes.init_posterior(prior), k, n)
    seq = bayes.sequential_update(prior, [b1, b2])
    match = np.isclose(pooled.alpha, seq.alpha) and np.isclose(pooled.beta, seq.beta)
    print(f"   cell {demo_cell}: total (k={k}, n={n}) split into {b1} then {b2}")
    print(f"   pooled posterior  : Beta(a={pooled.alpha:.3f}, b={pooled.beta:.3f}) "
          f"mean={pooled.mean:.4f}")
    print(f"   sequential posterior: Beta(a={seq.alpha:.3f}, b={seq.beta:.3f}) "
          f"mean={seq.mean:.4f}")
    print(f"   [{'PASS' if match else 'FAIL'}] sequential == pooled")

    # --- 5. Direction-of-update check ------------------------------------
    section("5. Update direction (clean FSIS samples lower estimated risk)")
    moved_down = int((table["posterior_mean"] < table["prior_mean"]).sum())
    moved_up = int((table["posterior_mean"] > table["prior_mean"]).sum())
    total_k = int(table["fsis_k"].sum())
    total_n = int(table["fsis_n"].sum())
    print(f"   FSIS evidence overall: {total_k} positive / {total_n} analyzed")
    print(f"   cells moved DOWN: {moved_down}, UP: {moved_up}")
    if total_k == 0 and total_n > 0:
        ok_dir = moved_down == len(table)
        print(f"   all evidence negative -> every posterior should drop: "
              f"{'PASS' if ok_dir else 'FAIL'}")
    else:
        ok_dir = True

    # --- 6. Posterior density artifact for one cell ----------------------
    section("6. Posterior distribution artifact")
    state, month = demo_cell
    grid = np.linspace(0, 1, 501)
    dens = pd.DataFrame({
        "w": grid,
        "prior_pdf": bayes.init_posterior(prior).pdf(grid),
        "posterior_pdf": pooled.pdf(grid),
    })
    dens_csv = ARTIFACT_DIR / f"density_{state}_m{month}.csv"
    dens.to_csv(dens_csv, index=False)
    print(f"   prior vs posterior density for {demo_cell}: "
          f"{dens_csv.relative_to(config.PROJECT_ROOT)}")

    all_ok = match and ok_dir
    section("Step 4 complete" if all_ok else "Step 4 completed WITH FAILURES")
    return 0 if all_ok else 1


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Step 4 Bayesian risk updating.")
    p.add_argument("--alert-tau", type=float, default=0.20,
                   help="Risk threshold tau for the P(w>tau) alert signal.")
    args = p.parse_args(argv)
    return run(alert_tau=args.alert_tau)


if __name__ == "__main__":
    sys.exit(main())
