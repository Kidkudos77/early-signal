"""Step 6 driver — BEAM ground-truth validation + dynamic NARMS weighting.

Run from the project root:
    python -m src.step6_beam_narms

Part A — BEAM ground-truth validation:
  * Build OOF Salmonella risk per state from Step 3's is_salmonella model.
  * Fetch BEAM Dashboard observed Salmonella isolates per state (live Socrata).
  * Quantify agreement (Pearson/Spearman, top-k hotspot overlap) and render a
    side-by-side predicted-vs-observed choropleth HTML.

Part B — Dynamic NARMS weighting:
  * Build the NARMS multiplier (heavier for Infantis/MDR and ciprofloxacin/DSC),
    ingesting the live BEAM clinically-important-AMR fraction per cell.
  * Combine with the Step 5 danger weights and show the effect on the weighting
    distribution (Infantis and DSC isolates get the heaviest penalties).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import (config, geo, ingest_api, ingest_local, loss, modeling, narms,
               severity, targets, validation)

ARTIFACT_DIR = config.PROJECT_ROOT / "artifacts" / "beam_narms"


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def run() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    nors = ingest_local.load_nors(foodborne_only=True)
    bundle = targets.build_is_salmonella(nors)

    # =====================================================================
    # Part A — BEAM ground-truth validation
    # =====================================================================
    section("A. BEAM ground-truth validation of OOF risk hotspots")
    print("   running OOF is_salmonella for the predicted risk surface...")
    res = modeling.run_oof(bundle, n_splits=5)

    print("   fetching BEAM Dashboard report data (live CDC Socrata)...")
    beam_report = ingest_api.fetch_cdc_socrata("beam_report_data")
    print(f"   BEAM rows: {len(beam_report)}")

    vr = validation.validate_against_beam(
        res.oof_pred, nors["ExposureState"], beam_report=beam_report, top_k=10,
    )
    print(f"   states compared        : {vr.n_states}")
    print(f"   Pearson(pred, log BEAM): {vr.pearson:.3f}")
    print(f"   Spearman(pred, BEAM)   : {vr.spearman:.3f}")
    print(f"   top-10 hotspot overlap : {vr.topk_overlap:.0%}")

    # Save the merged comparison and the map.
    merged_path = ARTIFACT_DIR / "beam_validation_by_state.csv"
    vr.merged.to_csv(merged_path, index=False)
    print(f"   comparison table: {merged_path.relative_to(config.PROJECT_ROOT)}")

    map_path = validation.render_hotspot_map(vr, ARTIFACT_DIR / "hotspot_validation_map.html")
    print(f"   hotspot map     : {map_path.relative_to(config.PROJECT_ROOT)}")

    # Show the top predicted vs. top observed (by SHARE) states explicitly.
    top_pred = vr.merged.nlargest(8, "predicted_risk")[["state", "predicted_risk"]]
    top_obs = vr.merged.nlargest(8, "beam_salmonella_share")[["state", "beam_salmonella_share"]]
    print("\n   top-8 predicted-risk states      :", list(top_pred["state"]))
    print("   top-8 BEAM Salmonella-share states:", list(top_obs["state"]))

    # =====================================================================
    # Part B — Dynamic NARMS weighting
    # =====================================================================
    section("B. Dynamic NARMS danger weighting")
    sev_frame = severity.build_severity_frame(nors)

    # Step 5 base danger weights.
    w5 = loss.normalize_weights(loss.danger_weights(sev_frame))

    # NARMS signals per instance.
    sero = sev_frame["serotype"]
    # MDR / DSC flags are not in NORS; we derive proxy flags where possible.
    # Infantis is the key MDR driver; DSC (ciprofloxacin) is flagged via the
    # BEAM clinically-important-AMR fraction joined by (state, month, serotype).
    dt = pd.to_datetime(nors["DateFirstIll"], errors="coerce")
    state = nors["ExposureState"].map(geo.to_abbr)
    month = dt.dt.month

    print("   building BEAM clinically-important-AMR lookup (state,month,serotype)...")
    amr_lookup = narms.ci_amr_lookup_from_beam(beam_report)
    print(f"   BEAM AMR cells: {len(amr_lookup)}")

    # Per-instance CI-AMR fraction via lookup (0 when absent).
    ci_amr = np.array([
        amr_lookup.get((st, int(mo) if pd.notna(mo) else -1, str(se)), 0.0)
        if pd.notna(st) else 0.0
        for st, mo, se in zip(state, month, sero)
    ], dtype="float64")

    # MDR proxy: treat Infantis as MDR driver; no per-row AST in NORS, so the
    # explicit MDR flag is serotype-driven here and would come from FSIS AST
    # fields in production.
    is_infantis = sero.astype("string").str.contains("Infantis", case=False, na=False)

    mult = narms.narms_multiplier(
        serotype=sero,
        is_mdr=is_infantis.astype(int),       # Infantis as MDR proxy in NORS
        is_dsc_cipro=(ci_amr > 0.15).astype(int),  # high CI-AMR -> DSC proxy
        ci_amr_pct=ci_amr,
    )
    w_final = narms.apply_narms_to_weights(w5, mult)

    # Report the effect.
    is_inf = is_infantis.to_numpy()
    sev_sero = sero.astype("string").str.contains("Enteritidis|Typhimurium",
                                                  case=False, na=False).to_numpy()
    print(f"   NARMS multiplier: mean={mult.mean():.3f} "
          f"Infantis mean={mult[is_inf].mean() if is_inf.any() else float('nan'):.3f} "
          f"other mean={mult[~is_inf].mean():.3f}")
    print(f"   final weight: overall mean={w_final.mean():.3f}")
    print(f"     Infantis rows mean weight   = "
          f"{w_final[is_inf].mean() if is_inf.any() else float('nan'):.3f}")
    print(f"     Enteritidis/Typhi rows mean = {w_final[sev_sero].mean():.3f}")
    print(f"     non-Salmonella rows mean    = "
          f"{w_final[sev_frame['is_salmonella'].to_numpy()==0].mean():.3f}")
    print(f"   rows with BEAM CI-AMR signal  = {int((ci_amr>0).sum())}")

    # Persist the weight table.
    wtab = pd.DataFrame({
        "serotype": sero.astype("string"),
        "is_salmonella": sev_frame["is_salmonella"],
        "severity": sev_frame["severity"],
        "step5_weight": w5,
        "ci_amr_fraction": ci_amr,
        "narms_multiplier": mult,
        "final_weight": w_final,
    })
    wpath = ARTIFACT_DIR / "narms_weights.csv"
    wtab.to_csv(wpath, index=False)
    print(f"   weights table: {wpath.relative_to(config.PROJECT_ROOT)} ({len(wtab)} rows)")

    # =====================================================================
    # Verification
    # =====================================================================
    section("Verification")
    ok_map = map_path.exists() and map_path.stat().st_size > 0
    # Infantis must be weighted strictly higher than generic Salmonella.
    generic_salm = (sev_frame["is_salmonella"].to_numpy() == 1) & (~is_inf) & (~sev_sero)
    ok_infantis = (is_inf.any() and
                   w_final[is_inf].mean() > w_final[generic_salm].mean())
    print(f"   [{'PASS' if ok_map else 'FAIL'}] hotspot validation map rendered")
    print(f"   [{'PASS' if ok_infantis else 'FAIL'}] Infantis weighted above generic Salmonella")
    # Correlation sign check (predicted risk should positively track BEAM).
    ok_corr = (not np.isnan(vr.spearman)) and vr.spearman > 0
    print(f"   [{'PASS' if ok_corr else 'WARN'}] predicted risk positively correlates with BEAM "
          f"(Spearman={vr.spearman:.3f})")

    all_ok = ok_map and ok_infantis
    section("Step 6 complete" if all_ok else "Step 6 completed WITH WARNINGS")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(run())
