"""Geographic helpers — U.S. state name <-> 2-letter abbreviation.

NORS records states by full name ("Ohio"); FSIS records them by USPS
abbreviation ("OH"). The Bayesian model keys cells by abbreviation, so this
module normalizes both sources onto the same key.
"""
from __future__ import annotations

STATE_NAME_TO_ABBR: dict[str, str] = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN",
    "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
    "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    "Puerto Rico": "PR", "Guam": "GU", "Virgin Islands": "VI",
    "American Samoa": "AS", "Northern Mariana Islands": "MP",
}

ABBR_TO_STATE_NAME: dict[str, str] = {v: k for k, v in STATE_NAME_TO_ABBR.items()}


def to_abbr(value: str | None) -> str | None:
    """Normalize a state value (name or abbr) to its 2-letter abbreviation."""
    if value is None:
        return None
    v = str(value).strip()
    if not v:
        return None
    if len(v) == 2 and v.upper() in ABBR_TO_STATE_NAME:
        return v.upper()
    return STATE_NAME_TO_ABBR.get(v.title(), STATE_NAME_TO_ABBR.get(v))


def to_name(value: str | None) -> str | None:
    """Normalize a state value (abbr or name) to its full name."""
    if value is None:
        return None
    v = str(value).strip()
    if not v:
        return None
    if v.upper() in ABBR_TO_STATE_NAME:
        return ABBR_TO_STATE_NAME[v.upper()]
    return v.title() if v.title() in STATE_NAME_TO_ABBR else None


__all__ = ["STATE_NAME_TO_ABBR", "ABBR_TO_STATE_NAME", "to_abbr", "to_name"]
