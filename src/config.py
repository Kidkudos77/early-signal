"""Centralized, secure configuration for the Early Signal (SENS-D) pipeline.

Security guardrail (Step 1): API credentials are NEVER hardcoded. Every token
is pulled at runtime via os.getenv() after loading the local `.env` file with
python-dotenv. The `.env` file is gitignored; `.env.examples` documents the
expected variable names with empty values.

Usage:
    from src.config import settings
    token = settings.require("NOAA_TOKEN")
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project paths (safe, relative, OS-independent via pathlib / os.path)
# ---------------------------------------------------------------------------
# config.py lives in <root>/src/, so the project root is one level up.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
ENV_PATH: Path = NOTEBOOKS_DIR / ".env"

# Load credentials from notebooks/.env into the process environment.
# `override=False` so real environment variables (e.g. CI secrets) win.
load_dotenv(dotenv_path=ENV_PATH, override=False)


# ---------------------------------------------------------------------------
# Local dataset locations (Step 1: local FSIS RTE JSON + NORS Excel)
# ---------------------------------------------------------------------------
FSIS_RTE_JSON: Path = (
    DATA_DIR
    / "usda_fsis_data_product_establishment_specific_laboratory_sampling_rte_product_sample.json"
)
NORS_XLSX: Path = DATA_DIR / "NORS.xlsx"
NORS_DICTIONARY_XLSX: Path = DATA_DIR / "NORS-Flat-DataDictionary-508_JAN2025_CLEAN.xlsx"

# CDC/ATSDR SVI county files, keyed by vintage year. The schema differs by
# era (2000 uses USG* percentiles + split FIPS; 2010 uses R_PL_THEMES; 2014+
# use RPL_THEME1..4 + RPL_THEMES). src.ingest_local.load_svi normalizes them.
SVI_COUNTY_FILES: dict[int, Path] = {
    2000: DATA_DIR / "SVI_2000_US_COUNTY.csv",
    2010: DATA_DIR / "SVI_2010_US_COUNTY.csv",
    2014: DATA_DIR / "SVI_2014_US_COUNTY.csv",
    2016: DATA_DIR / "SVI_2016_US_COUNTY.csv",
    2018: DATA_DIR / "SVI_2018_US_COUNTY.csv",
    2020: DATA_DIR / "SVI_2020_US_COUNTY.csv",
    2022: DATA_DIR / "SVI_2022_US_COUNTY.csv",
}
# ATSDR missing-value sentinel used across SVI vintages.
SVI_MISSING_SENTINEL = -999

# U.S. Census 2023 Gazetteer county internal points (lat/lon per county FIPS).
# Public download, trimmed to FIPS/state/lat/lon. Used to route donations to
# real nearby counties instead of state centroids.
COUNTY_CENTROIDS_CSV: Path = DATA_DIR / "county_centroids_2023.csv"


# ---------------------------------------------------------------------------
# API endpoints (public URLs — not secrets)
# ---------------------------------------------------------------------------
NOAA_GHCND_BASE = "https://www.ncei.noaa.gov/cdo-web/api/v2"
NASS_QUICKSTATS_BASE = "https://quickstats.nass.usda.gov/api/api_GET/"
AMS_MARS_BASE = "https://marsapi.ams.usda.gov/services/v1.2"
CENSUS_BASE = "https://api.census.gov/data"

# CDC Socrata v3 query endpoints (BEAM Dashboard + NORS). Public, keyless.
CDC_SOCRATA_ENDPOINTS: dict[str, str] = {
    "beam_report_data": "https://data.cdc.gov/api/v3/views/jbhn-e8xn/query.csv",
    "beam_isolates_by_hhs_region": "https://data.cdc.gov/api/v3/views/khic-yj26/query.csv",
    "beam_top30_serotypes": "https://data.cdc.gov/api/v3/views/ch83-ush6/query.csv",
    "beam_serotypes_concern_illnesses_outbreaks": "https://data.cdc.gov/api/v3/views/fvm6-ic5r/query.csv",
    "beam_serotypes_concern_burden_trajectory": "https://data.cdc.gov/api/v3/views/8na9-qgz7/query.csv",
    "nors": "https://data.cdc.gov/api/v3/views/5xkq-dg7x/query.csv",
}


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Credential *values* are read lazily and never stored
    as attributes, so they cannot be accidentally logged or serialized."""

    env_names: tuple[str, ...] = field(
        default=(
            "NOAA_TOKEN",
            "CENSUS_API_KEY",
            "NASS_API_KEY",
            "AMS_API_KEY",
        )
    )

    def get(self, name: str, default: str | None = None) -> str | None:
        """Return a credential value or `default` if unset. Never logs the value."""
        return os.getenv(name, default)

    def require(self, name: str) -> str:
        """Return a credential value, raising a clear error if it is missing.

        The error message references the variable *name* only — never a value —
        and points the user to the `.env` setup.
        """
        value = os.getenv(name)
        if not value:
            raise RuntimeError(
                f"Required credential '{name}' is not set. "
                f"Copy notebooks/.env.examples to notebooks/.env and fill in '{name}'. "
                f"Credentials are loaded via os.getenv() and must never be hardcoded."
            )
        return value

    def status(self) -> dict[str, bool]:
        """Return which credentials are present WITHOUT revealing their values.

        Useful for a notebook sanity check: prints True/False per key only.
        """
        return {name: bool(os.getenv(name)) for name in self.env_names}


settings = Settings()
