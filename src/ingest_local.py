"""Step 1 — Local data ingestion.

Loads the local FSIS Ready-to-Eat (RTE) product-sampling JSON and the CDC NORS
Excel workbook using safe relative paths (via src.config, which builds paths
with pathlib from the project root).

Per the brief:
  * Load the FSIS RTE JSON primary + secondary pathogen tables.
  * Perform a FULL OUTER JOIN on `form_id` between them.
  * Filter explicitly for poultry products.

Schema notes (confirmed against the shipped files):
  * The FSIS JSON is a single-element list; element[0] has keys
    {metadata, data, appendix}. `data` is a dict with:
        - primary_table_data   : one row per (form_id, sample_number); holds
                                 establishment, product_name, collection_date
                                 and the Salmonella/Listeria screening flags.
        - secondary_table_data : keyed by form_id; holds salmonella_serotype
                                 and the full antimicrobial MIC / AST / WGS
                                 resistance panel (ciprofloxacin, ceftriaxone,
                                 azithromycin, trim-sulfa, meropenem, ...).
    The public extract is a SAMPLE (metadata.data_records == {primary:20,
    secondary:6}); this loader is schema-driven so it runs unchanged on the
    full export.
  * NORS.xlsx is wide CDC outbreak data (one row per outbreak, ~399 cols):
    pathogen in GenusName_1..N, poultry signal in Food_*_FoodName text,
    transmission in PrimaryMode, onset in DateFirstIll. NORS has no form_id;
    it is the seasonal/geographic PRIOR source for Step 4, not part of the
    form_id join.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from . import config

# ---------------------------------------------------------------------------
# Poultry identification
# ---------------------------------------------------------------------------
# Keywords used to flag poultry products/foods. Case-insensitive substring
# match. RTE poultry is the focus, but we keep the net wide (hen, fowl, etc.)
# and let downstream steps narrow further if needed.
POULTRY_KEYWORDS: tuple[str, ...] = (
    "chicken",
    "poultry",
    "turkey",
    "hen",
    "fowl",
    "duck",
    "broiler",
    "capon",
)

_POULTRY_PATTERN = "|".join(POULTRY_KEYWORDS)


# ---------------------------------------------------------------------------
# FSIS RTE JSON
# ---------------------------------------------------------------------------
@dataclass
class FSISTables:
    """Container for the two raw FSIS pathogen tables plus their join."""

    primary: pd.DataFrame
    secondary: pd.DataFrame
    joined: pd.DataFrame  # full outer join on form_id
    poultry: pd.DataFrame  # joined, filtered to poultry products


def _read_fsis_json() -> dict:
    """Return the inner object (element[0]) of the FSIS JSON list."""
    path = config.FSIS_RTE_JSON
    if not path.exists():
        raise FileNotFoundError(
            f"FSIS RTE JSON not found at {path}. Expected it under data/."
        )
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    # The asset is a single-element list wrapping {metadata, data, appendix}.
    obj = raw[0] if isinstance(raw, list) else raw
    if "data" not in obj:
        raise ValueError("FSIS JSON missing the 'data' section.")
    return obj


def load_fsis_primary_secondary() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the FSIS primary and secondary pathogen tables as DataFrames."""
    obj = _read_fsis_json()
    data = obj["data"]
    primary = pd.DataFrame(data.get("primary_table_data", []))
    secondary = pd.DataFrame(data.get("secondary_table_data", []))

    if "form_id" not in primary.columns and not primary.empty:
        raise ValueError("FSIS primary table missing required 'form_id' key.")
    if "form_id" not in secondary.columns and not secondary.empty:
        raise ValueError("FSIS secondary table missing required 'form_id' key.")

    # Normalize the join key to string so the outer join never fails on dtype
    # mismatch (form_id is 'character varying' per the data dictionary).
    for df in (primary, secondary):
        if "form_id" in df.columns:
            df["form_id"] = df["form_id"].astype("string")

    return primary, secondary


def join_on_form_id(
    primary: pd.DataFrame, secondary: pd.DataFrame
) -> pd.DataFrame:
    """Full OUTER join of primary and secondary pathogen tables on `form_id`.

    Overlapping non-key columns are suffixed to keep both sides. An indicator
    column `_merge` records whether each row came from the primary side only,
    the secondary side only, or both — useful for the 'orphan' secondary/
    tertiary tests the FSIS metadata warns about.
    """
    if primary.empty and secondary.empty:
        return pd.DataFrame()
    if secondary.empty:
        out = primary.copy()
        out["_merge"] = "left_only"
        return out
    if primary.empty:
        out = secondary.copy()
        out["_merge"] = "right_only"
        return out

    return primary.merge(
        secondary,
        on="form_id",
        how="outer",
        suffixes=("_primary", "_secondary"),
        indicator=True,
    )


def _poultry_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask selecting poultry rows from a joined FSIS frame.

    Scans product/sample text columns for poultry keywords. We check
    `product_name`, `sample_source_name`, and `project_name` because RTE
    poultry can be identified by any of them in the FSIS schema.
    """
    candidate_cols = [
        c
        for c in ("product_name", "sample_source_name", "project_name")
        if c in df.columns
    ]
    if not candidate_cols:
        # Nothing to filter on — return all-False so the caller sees an empty
        # poultry set rather than silently passing non-poultry rows through.
        return pd.Series(False, index=df.index)

    mask = pd.Series(False, index=df.index)
    for col in candidate_cols:
        mask |= (
            df[col]
            .astype("string")
            .str.contains(_POULTRY_PATTERN, case=False, na=False)
        )
    return mask


def filter_poultry(joined: pd.DataFrame) -> pd.DataFrame:
    """Return only poultry-product rows from a joined FSIS frame."""
    if joined.empty:
        return joined
    return joined.loc[_poultry_mask(joined)].reset_index(drop=True)


def load_fsis(require_poultry: bool = False) -> FSISTables:
    """End-to-end FSIS load: primary + secondary, full outer join, poultry filter.

    Parameters
    ----------
    require_poultry:
        If True, raise when the poultry filter yields zero rows (useful once
        the full dataset is in place; left False for the shipped SAMPLE, which
        may legitimately contain no poultry rows).
    """
    primary, secondary = load_fsis_primary_secondary()
    joined = join_on_form_id(primary, secondary)
    poultry = filter_poultry(joined)

    if require_poultry and poultry.empty:
        raise ValueError(
            "No poultry products found after filtering. Check that the full "
            "FSIS export (not the 20-row SAMPLE) is present in data/."
        )

    return FSISTables(
        primary=primary,
        secondary=secondary,
        joined=joined,
        poultry=poultry,
    )


# ---------------------------------------------------------------------------
# CDC NORS Excel
# ---------------------------------------------------------------------------
def load_nors(
    foodborne_only: bool = True,
    salmonella_only: bool = False,
    poultry_only: bool = False,
) -> pd.DataFrame:
    """Load the CDC NORS outbreak workbook.

    Parameters
    ----------
    foodborne_only:
        Keep only PrimaryMode == 'Food' outbreaks (the relevant transmission
        route for FSIS-regulated product).
    salmonella_only:
        Keep only outbreaks with Salmonella in any GenusName_* column.
    poultry_only:
        Keep only outbreaks whose any Food_*_FoodName mentions poultry.

    The NORS workbook is the seasonal/geographic PRIOR source for the Step 4
    Bayesian baseline; filters here are opt-in so callers control the subset.
    """
    path = config.NORS_XLSX
    if not path.exists():
        raise FileNotFoundError(f"NORS workbook not found at {path}.")

    df = pd.read_excel(path, sheet_name="Sheet1")

    if foodborne_only and "PrimaryMode" in df.columns:
        df = df[df["PrimaryMode"].astype("string").str.strip() == "Food"]

    if salmonella_only:
        df = df[_nors_genus_mask(df, "Salmonella")]

    if poultry_only:
        df = df[_nors_food_mask(df, _POULTRY_PATTERN)]

    return df.reset_index(drop=True)


def _nors_genus_mask(df: pd.DataFrame, genus: str) -> pd.Series:
    """Rows where `genus` appears in any GenusName_* column (case-insensitive)."""
    genus_cols = [c for c in df.columns if c.startswith("GenusName_")]
    if not genus_cols:
        return pd.Series(False, index=df.index)
    mask = pd.Series(False, index=df.index)
    for col in genus_cols:
        mask |= df[col].astype("string").str.contains(genus, case=False, na=False)
    return mask


def _nors_food_mask(df: pd.DataFrame, pattern: str) -> pd.Series:
    """Rows where any Food_*_FoodName matches `pattern` (case-insensitive regex)."""
    food_cols = [c for c in df.columns if c.endswith("_FoodName")]
    if not food_cols:
        return pd.Series(False, index=df.index)
    mask = pd.Series(False, index=df.index)
    for col in food_cols:
        mask |= df[col].astype("string").str.contains(pattern, case=False, na=False)
    return mask


def load_nors_dictionary() -> dict[str, pd.DataFrame]:
    """Load the NORS flat data dictionary (both sheets) for reference/lookup."""
    path = config.NORS_DICTIONARY_XLSX
    if not path.exists():
        raise FileNotFoundError(f"NORS data dictionary not found at {path}.")
    xls = pd.ExcelFile(path)
    return {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names}


# ---------------------------------------------------------------------------
# CDC / ATSDR Social Vulnerability Index (SVI), county level, 2000-2022
# ---------------------------------------------------------------------------
# Normalized output schema (one row per county per vintage):
#   svi_year      int   : vintage year
#   FIPS          str   : 5-digit county FIPS (zero-padded)
#   state         str   : state abbreviation (or name for 2000)
#   county        str   : county name
#   svi_overall   float : overall SVI percentile ranking in [0, 1]
#                         (RPL_THEMES / R_PL_THEMES; derived for 2000)
#   svi_theme1..4 float : the four theme percentile rankings where available
#
# The percentile ranking is the ATSDR summary used for equity comparison: a
# value near 1.0 = most vulnerable. Missing values (-999 sentinel) -> NaN.
SVI_NORMALIZED_COLUMNS = (
    "svi_year",
    "FIPS",
    "state",
    "county",
    "svi_overall",
    "svi_theme1",
    "svi_theme2",
    "svi_theme3",
    "svi_theme4",
)


def _clean_svi_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace ATSDR -999 missing sentinels with NaN (numeric columns only)."""
    return df.replace(config.SVI_MISSING_SENTINEL, pd.NA)


def _normalize_svi_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Map one vintage's native columns onto the common normalized schema."""
    cols = set(df.columns)
    out = pd.DataFrame()
    out["svi_year"] = [year] * len(df)

    # --- FIPS (5-digit string) -------------------------------------------
    if "FIPS" in cols:
        fips = df["FIPS"].astype("string")
    elif {"STATE_FIPS", "CNTY_FIPS"}.issubset(cols):
        # 2000: combine 2-digit state + 3-digit county.
        fips = (
            df["STATE_FIPS"].astype("string").str.zfill(2)
            + df["CNTY_FIPS"].astype("string").str.zfill(3)
        )
    elif "STCOFIPS" in cols:
        fips = df["STCOFIPS"].astype("string")
    elif "STCNTY" in cols:
        fips = df["STCNTY"].astype("string")
    else:
        fips = pd.Series([pd.NA] * len(df), dtype="string")
    out["FIPS"] = fips.str.replace(r"\.0$", "", regex=True).str.zfill(5)

    # --- state / county labels -------------------------------------------
    if "ST_ABBR" in cols:
        out["state"] = df["ST_ABBR"].astype("string")
    elif "STATE_ABBR" in cols:
        out["state"] = df["STATE_ABBR"].astype("string")
    elif "STATE" in cols:
        out["state"] = df["STATE"].astype("string")
    elif "STATE_NAME" in cols:
        out["state"] = df["STATE_NAME"].astype("string")
    else:
        out["state"] = pd.NA

    for name_col in ("COUNTY", "LOCATION"):
        if name_col in cols:
            out["county"] = df[name_col].astype("string")
            break
    else:
        out["county"] = pd.NA

    # --- overall + theme rankings ----------------------------------------
    # 2014+: RPL_THEMES / RPL_THEME1..4. 2010: R_PL_THEMES (no per-theme
    # standardized rankings in the same naming). 2000: no RPL at all.
    def _num(colname: str):
        return pd.to_numeric(df[colname], errors="coerce") if colname in cols else pd.NA

    if "RPL_THEMES" in cols:
        out["svi_overall"] = _num("RPL_THEMES")
    elif "R_PL_THEMES" in cols:
        out["svi_overall"] = _num("R_PL_THEMES")
    else:
        out["svi_overall"] = pd.NA  # 2000 handled below

    out["svi_theme1"] = _num("RPL_THEME1")
    out["svi_theme2"] = _num("RPL_THEME2")
    out["svi_theme3"] = _num("RPL_THEME3")
    out["svi_theme4"] = _num("RPL_THEME4")

    # 2000 special case: no precomputed overall percentile. ATSDR's 2000
    # product reports per-variable rankings (G*R) and USG* summaries rather
    # than a single 0-1 RPL_THEMES. Derive a comparable overall percentile by
    # percentile-ranking the count of flagged variables (USG*F fields), so the
    # 2000 vintage still yields a usable, monotonic vulnerability score. This
    # is an approximation and is flagged as such in the loader docstring.
    if year == 2000:
        flag_cols = [c for c in df.columns if c.upper().startswith("USG") and c.upper().endswith("F")]
        if flag_cols:
            flags = df[flag_cols].apply(pd.to_numeric, errors="coerce")
            total_flags = flags.sum(axis=1, min_count=1)
            out["svi_overall"] = total_flags.rank(pct=True)

    out = _clean_svi_sentinels(out)
    # Coerce ranking columns to float (NA-aware).
    for c in ("svi_overall", "svi_theme1", "svi_theme2", "svi_theme3", "svi_theme4"):
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Drop stray header/key rows: some ATSDR files (e.g. 2000) carry a second
    # label row whose FIPS is non-numeric ('STCOU'). A valid county FIPS is 5
    # digits; filter anything that isn't all-digits.
    valid_fips = out["FIPS"].str.fullmatch(r"\d{5}")
    out = out.loc[valid_fips.fillna(False)].reset_index(drop=True)
    return out[list(SVI_NORMALIZED_COLUMNS)]


def load_svi(year: int, *, normalized: bool = True) -> pd.DataFrame:
    """Load one ATSDR SVI county vintage.

    Parameters
    ----------
    year:
        One of the available vintages (2000, 2010, 2014, 2016, 2018, 2020, 2022).
    normalized:
        If True (default), return the common normalized schema
        (SVI_NORMALIZED_COLUMNS). If False, return the raw file as-is.

    Notes
    -----
    The 2000 vintage lacks a precomputed overall percentile (RPL_THEMES); its
    `svi_overall` is an approximation derived by percentile-ranking the number
    of flagged vulnerability variables. 2010+ use ATSDR's published rankings
    directly.
    """
    if year not in config.SVI_COUNTY_FILES:
        valid = ", ".join(str(y) for y in sorted(config.SVI_COUNTY_FILES))
        raise KeyError(f"No SVI file for year {year}. Available: {valid}")
    path = config.SVI_COUNTY_FILES[year]
    if not path.exists():
        raise FileNotFoundError(f"SVI {year} file not found at {path}.")

    raw = pd.read_csv(path, low_memory=False)
    if not normalized:
        return raw
    return _normalize_svi_year(raw, year)


def load_svi_all(normalized: bool = True) -> pd.DataFrame:
    """Load and vertically concatenate every available SVI vintage.

    Only meaningful with normalized=True (schemas differ across years). Returns
    a long frame with one row per county per vintage, carrying `svi_year`.
    """
    if not normalized:
        raise ValueError("load_svi_all requires normalized=True (schemas differ by year).")
    frames = [load_svi(y, normalized=True) for y in sorted(config.SVI_COUNTY_FILES)
              if config.SVI_COUNTY_FILES[y].exists()]
    if not frames:
        return pd.DataFrame(columns=list(SVI_NORMALIZED_COLUMNS))
    return pd.concat(frames, ignore_index=True)


__all__ = [
    "FSISTables",
    "POULTRY_KEYWORDS",
    "SVI_NORMALIZED_COLUMNS",
    "load_fsis",
    "load_fsis_primary_secondary",
    "join_on_form_id",
    "filter_poultry",
    "load_nors",
    "load_nors_dictionary",
    "load_svi",
    "load_svi_all",
]
