"""Step 1 driver — Secure Data Ingestion & API Integration.

Run as a module from the project root:
    python -m src.step1_ingest

What it does (and verifies):
  1. Loads local FSIS RTE JSON; builds the primary/secondary full-outer-join
     on form_id; filters poultry.
  2. Loads CDC NORS outbreak workbook (foodborne subset) as the Step 4 prior
     source.
  3. Reports credential presence WITHOUT revealing any token value.
  4. Performs one live, keyless CDC Socrata pull (NORS via query.csv) to prove
     the dynamic-API path works end to end.

Credentialed APIs (NOAA, NASS, AMS, Census) are wired and ready; this driver
only *reports readiness* for them by default so it can run offline/without
burning rate limits. Pass --live-credentialed to attempt a small real call
for each (requires valid keys in notebooks/.env).
"""
from __future__ import annotations

import argparse
import sys

from . import ingest_api, ingest_local
from .config import settings


def section(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def run(live_credentialed: bool = False) -> int:
    # --- 1. Local FSIS ----------------------------------------------------
    section("1. FSIS RTE JSON — load, full-outer-join on form_id, poultry filter")
    fsis = ingest_local.load_fsis(require_poultry=False)
    print(f"   primary_table_data rows   : {len(fsis.primary)}")
    print(f"   secondary_table_data rows : {len(fsis.secondary)}")
    print(f"   joined (outer on form_id) : {len(fsis.joined)} rows, "
          f"{fsis.joined.shape[1] if not fsis.joined.empty else 0} cols")
    if not fsis.joined.empty and "_merge" in fsis.joined.columns:
        print(f"   join provenance           : "
              f"{fsis.joined['_merge'].value_counts().to_dict()}")
    print(f"   poultry rows              : {len(fsis.poultry)}")
    # Surface a couple of key AMR/serotype columns if present (Steps 5-6 inputs)
    amr_cols = [c for c in fsis.joined.columns
                if c in ("salmonella_serotype",
                         "salmonella_ciprofloxacin_resistant_ast",
                         "salmonella_ceftriaxone_resistant_ast")]
    print(f"   AMR/serotype cols present : {amr_cols or 'none in SAMPLE'}")

    # --- 2. NORS prior source --------------------------------------------
    section("2. CDC NORS workbook — foodborne outbreak prior (Step 4 source)")
    nors = ingest_local.load_nors(foodborne_only=True)
    print(f"   foodborne outbreak rows   : {len(nors)}")
    nors_salm = ingest_local.load_nors(foodborne_only=True, salmonella_only=True)
    print(f"   ...of which Salmonella     : {len(nors_salm)}")
    nors_poultry = ingest_local.load_nors(
        foodborne_only=True, salmonella_only=True, poultry_only=True
    )
    print(f"   ...Salmonella + poultry    : {len(nors_poultry)}")

    # --- 3. Credential readiness (NO values printed) ---------------------
    section("3. Credential readiness (values never shown)")
    for name, present in settings.status().items():
        print(f"   {name:<16}: {'LOADED' if present else 'MISSING'}")

    # --- 4. Live keyless CDC Socrata pull --------------------------------
    section("4. CDC Socrata (keyless) — live NORS query.csv pull")
    try:
        socrata_nors = ingest_api.fetch_cdc_socrata("nors")
        print(f"   NORS Socrata rows x cols  : {socrata_nors.shape}")
        print(f"   first columns             : {list(socrata_nors.columns[:6])}")
    except Exception as exc:  # noqa: BLE001 - report, don't crash the driver
        print(f"   [skipped/failed] {type(exc).__name__}: {exc}")

    # --- 5. Optional: exercise credentialed APIs -------------------------
    if live_credentialed:
        section("5. Live credentialed API probes (small calls)")
        _probe_credentialed()

    section("Step 1 complete")
    return 0


def _probe_credentialed() -> None:
    """Attempt one small real call per credentialed API. Best-effort."""
    # NOAA: list one day of summaries for a well-known station (GHCND:USW00013874 = Atlanta).
    try:
        noaa = ingest_api.NOAAClient()
        df = noaa.daily_summaries(
            station_id="GHCND:USW00013874",
            start="2019-07-01",
            end="2019-07-02",
            datatypes=("TMAX", "TMIN"),
        )
        print(f"   NOAA GHCND rows           : {len(df)}")
    except Exception as exc:  # noqa: BLE001
        print(f"   NOAA [skipped] {type(exc).__name__}: {exc}")

    # NASS: broilers production, one state, one year.
    try:
        df = ingest_api.fetch_nass_quickstats({
            "commodity_desc": "CHICKENS",
            "year": "2019",
            "agg_level_desc": "STATE",
            "state_alpha": "GA",
            "statisticcat_desc": "INVENTORY",
        })
        print(f"   NASS Quick Stats rows     : {len(df)}")
    except Exception as exc:  # noqa: BLE001
        print(f"   NASS [skipped] {type(exc).__name__}: {exc}")

    # Census: enumerate GA counties with population + FIPS (SVI join keys).
    try:
        df = ingest_api.fetch_census_counties(year="2022", state_fips="13")
        print(f"   Census GA counties rows   : {len(df)} "
              f"(cols incl FIPS: {'FIPS' in df.columns})")
    except Exception as exc:  # noqa: BLE001
        print(f"   Census [skipped] {type(exc).__name__}: {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 1 ingestion driver.")
    parser.add_argument(
        "--live-credentialed",
        action="store_true",
        help="Attempt small real calls against NOAA/NASS/Census (uses keys).",
    )
    args = parser.parse_args(argv)
    return run(live_credentialed=args.live_credentialed)


if __name__ == "__main__":
    sys.exit(main())
