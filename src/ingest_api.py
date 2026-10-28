"""Step 1 — Dynamic API ingestion.

Fetches the external signals the SENS-D model needs:
  * NOAA GHCND daily summaries  -> environmental cold-chain stress
  * USDA NASS Quick Stats        -> commodity production/transit volumes
  * USDA AMS Market News (MARS)   -> commodity movement/transit
  * CDC/ATSDR SVI via Census API  -> demographic equity by county
  * CDC Socrata v3 (BEAM + NORS)  -> serotype burden + outbreak ground truth

SECURITY GUARDRAIL (Step 1): No API token is ever hardcoded. Every credential
is read at call time via src.config.settings.require(<ENV_NAME>), which pulls
from the process environment populated by python-dotenv from notebooks/.env.
Token values are never logged, echoed, or embedded in URLs that get printed.

Only the Socrata endpoints (public, keyless) are fetched without a token.
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field

import pandas as pd
import requests

from . import config
from .config import settings

# ---------------------------------------------------------------------------
# HTTP session with a sane default timeout and retry-ish backoff.
# ---------------------------------------------------------------------------
DEFAULT_TIMEOUT = 60  # seconds
_session = requests.Session()
_session.headers.update({"User-Agent": "EarlySignal-SENSD/0.1 (research)"})


class APIError(RuntimeError):
    """Raised when an upstream API returns a non-success response."""


def _get(url: str, *, params: dict | None = None, headers: dict | None = None) -> requests.Response:
    """GET with timeout + status check. Never logs headers (may hold tokens)."""
    resp = _session.get(url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
    if resp.status_code != 200:
        # Report status + URL path only — not params/headers, which may carry
        # credentials. This keeps tokens out of tracebacks and logs.
        raise APIError(
            f"GET {url.split('?')[0]} failed with HTTP {resp.status_code}."
        )
    return resp


# ===========================================================================
# NOAA NCEI GHCND — Daily Summaries (cold-chain environmental stress)
# ===========================================================================
@dataclass
class NOAAClient:
    """Minimal NOAA Climate Data Online (CDO v2) GHCND client.

    Respects the documented limits (5 req/s, 10,000 req/day). The token is
    sent in the required `token` header and is read via settings.require, so
    it never appears in source.
    """

    base_url: str = config.NOAA_GHCND_BASE
    min_interval_s: float = 0.25  # <= 5 requests/sec (we stay well under)
    _last_call: float = field(default=0.0, repr=False)

    def _headers(self) -> dict[str, str]:
        # Pulled dynamically; value never printed.
        return {"token": settings.require("NOAA_TOKEN")}

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        self._last_call = time.monotonic()

    def daily_summaries(
        self,
        *,
        station_id: str,
        start: str,
        end: str,
        datatypes: tuple[str, ...] = ("TMIN", "TMAX", "TAVG", "PRCP"),
        units: str = "metric",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch GHCND daily summaries for one station over [start, end].

        Dates are ISO 'YYYY-MM-DD'. Returns a tidy DataFrame (one row per
        observation) with columns [date, datatype, station, value, ...].
        Handles CDO pagination via the `offset` parameter.
        """
        rows: list[dict] = []
        offset = 1
        while True:
            self._throttle()
            params = {
                "datasetid": "GHCND",
                "stationid": station_id,
                "startdate": start,
                "enddate": end,
                "units": units,
                "limit": limit,
                "offset": offset,
            }
            for dt in datatypes:
                # CDO accepts repeated datatypeid params; requests encodes a
                # list as repeated keys.
                params.setdefault("datatypeid", [])
                params["datatypeid"].append(dt)

            resp = _get(f"{self.base_url}/data", params=params, headers=self._headers())
            payload = resp.json()
            results = payload.get("results", [])
            rows.extend(results)

            meta = payload.get("metadata", {}).get("resultset", {})
            count = int(meta.get("count", len(results)))
            if offset + limit > count or not results:
                break
            offset += limit

        return pd.DataFrame(rows)


# ===========================================================================
# USDA NASS Quick Stats — commodity production / transit volumes
# ===========================================================================
def fetch_nass_quickstats(params: dict) -> pd.DataFrame:
    """Query USDA NASS Quick Stats (JSON) and return a DataFrame.

    `params` is the Quick Stats query (e.g. commodity_desc, year, agg_level_desc,
    state_alpha). The API key is injected from NASS_API_KEY at call time.
    """
    query = dict(params)
    query["key"] = settings.require("NASS_API_KEY")
    query.setdefault("format", "JSON")
    resp = _get(config.NASS_QUICKSTATS_BASE, params=query)
    data = resp.json().get("data", [])
    return pd.DataFrame(data)


# ===========================================================================
# USDA AMS Market News (MARS v1.2) — commodity movement / transit
# ===========================================================================
def fetch_ams_report(report_id: str, *, params: dict | None = None) -> pd.DataFrame:
    """Fetch a USDA AMS Market News report by ID.

    AMS MARS uses HTTP Basic auth with the API key as the username and an empty
    password. The key is read from AMS_API_KEY and passed via requests' `auth`
    (never embedded in the URL or logged).
    """
    api_key = settings.require("AMS_API_KEY")
    url = f"{config.AMS_MARS_BASE}/reports/{report_id}"
    resp = _session.get(
        url,
        params=params,
        auth=(api_key, ""),
        timeout=DEFAULT_TIMEOUT,
        headers={"Accept": "application/json"},
    )
    if resp.status_code != 200:
        raise APIError(f"AMS report {report_id} failed with HTTP {resp.status_code}.")
    body = resp.json()
    # MARS wraps rows under 'results' in most report shapes.
    rows = body.get("results", body if isinstance(body, list) else [])
    return pd.DataFrame(rows)


# ===========================================================================
# U.S. Census Data API — county enumeration, FIPS, population denominators
# ===========================================================================
# IMPORTANT: The CDC/ATSDR Social Vulnerability Index (SVI) is NOT an ACS
# variable — SVI fields like RPL_THEMES are published by ATSDR (svi.cdc.gov),
# not by the Census API. The Census API is used here for what it actually
# serves: county geographies, FIPS join keys, and population denominators that
# the equity analysis joins against the ATSDR SVI table. See load_svi_file()
# below for the SVI source itself.
def fetch_census_counties(
    *,
    year: str = "2022",
    variables: tuple[str, ...] = ("B01003_001E",),  # total population
    state_fips: str | None = None,
    dataset: str = "acs/acs5",
) -> pd.DataFrame:
    """Fetch county rows (NAME + requested ACS variables + FIPS) from Census.

    Default variable B01003_001E is total population, used as an equity
    denominator. `state_fips` limits to one state (e.g. '13' = Georgia);
    omit for all states. The Census key is injected from CENSUS_API_KEY and
    never appears in source or logs.
    """
    get_fields = ",".join(("NAME", *variables))
    params = {
        "get": get_fields,
        "for": "county:*",
        "key": settings.require("CENSUS_API_KEY"),
    }
    if state_fips:
        params["in"] = f"state:{state_fips}"

    url = f"{config.CENSUS_BASE}/{year}/{dataset}"
    resp = _get(url, params=params)
    payload = resp.json()
    if not payload:
        return pd.DataFrame()
    header, *records = payload
    df = pd.DataFrame(records, columns=header)
    # Build a 5-digit county FIPS for joining to ATSDR SVI (state+county).
    if {"state", "county"}.issubset(df.columns):
        df["FIPS"] = df["state"].str.zfill(2) + df["county"].str.zfill(3)
    return df


def load_svi_file(path, *, geography: str = "county") -> pd.DataFrame:
    """Load a CDC/ATSDR SVI table from a local file (CSV).

    SVI is distributed by ATSDR as downloads from svi.cdc.gov (county- or
    tract-level). Download the desired year/geography once, drop it under
    data/, and point this loader at it. The overall ranking column is
    'RPL_THEMES'; the county FIPS column is 'FIPS'. This keeps the equity
    signal grounded in the authoritative ATSDR source rather than mis-pulling
    it from the Census ACS API.

    Parameters
    ----------
    path:
        Path to the downloaded SVI CSV (str or pathlib.Path).
    geography:
        'county' or 'tract' — informational; the loader reads whatever is
        in the file.
    """
    from pathlib import Path as _Path

    p = _Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"SVI file not found at {p}. Download the {geography}-level SVI "
            f"from https://svi.cdc.gov/dataDownloads/data-download.html and "
            f"place it under data/."
        )
    svi = pd.read_csv(p)
    if "FIPS" in svi.columns:
        svi["FIPS"] = svi["FIPS"].astype("string").str.zfill(5)
    return svi


# ===========================================================================
# CDC Socrata v3 — BEAM Dashboard + NORS (public, keyless query.csv)
# ===========================================================================
def fetch_cdc_socrata(dataset: str, *, query: str | None = None) -> pd.DataFrame:
    """Fetch a CDC Socrata v3 dataset as CSV.

    `dataset` is a key of config.CDC_SOCRATA_ENDPOINTS (e.g. 'nors',
    'beam_top30_serotypes'). An optional SoQL `query` string is appended as the
    `query` parameter. These endpoints are public and require no credential.
    """
    if dataset not in config.CDC_SOCRATA_ENDPOINTS:
        valid = ", ".join(sorted(config.CDC_SOCRATA_ENDPOINTS))
        raise KeyError(f"Unknown CDC dataset '{dataset}'. Valid: {valid}")

    url = config.CDC_SOCRATA_ENDPOINTS[dataset]
    params = {"query": query} if query else None
    resp = _get(url, params=params)
    return pd.read_csv(io.StringIO(resp.text))


def fetch_beam_dashboard() -> dict[str, pd.DataFrame]:
    """Fetch all five BEAM Dashboard Socrata extracts, keyed by short name."""
    return {
        name: fetch_cdc_socrata(name)
        for name in config.CDC_SOCRATA_ENDPOINTS
        if name.startswith("beam_")
    }


__all__ = [
    "APIError",
    "NOAAClient",
    "fetch_nass_quickstats",
    "fetch_ams_report",
    "fetch_census_counties",
    "load_svi_file",
    "fetch_cdc_socrata",
    "fetch_beam_dashboard",
]
