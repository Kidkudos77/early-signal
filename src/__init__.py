"""Early Signal — proactive, time-aware Salmonella surge decision-support for
FSIS-regulated Ready-to-Eat poultry (NAFSI SENS-D, Track 3).

Step 1 of the pipeline (secure data ingestion & API integration) lives in:
    - src.config          : paths + credential loading (os.getenv + dotenv)
    - src.ingest_local    : FSIS RTE JSON + NORS Excel loaders and the
                            form_id full-outer-join / poultry filter
    - src.ingest_api      : NOAA GHCND, USDA NASS/AMS, CDC SVI (Census),
                            and CDC Socrata (BEAM + NORS) dynamic fetchers
"""

__all__ = ["config", "ingest_local", "ingest_api"]
__version__ = "0.1.0"
