# 🐔 Early Signal

**Proactive, time-aware decision support that predicts seasonal _Salmonella_
surges in FSIS-regulated Ready-to-Eat (RTE) poultry and recommends operational
interventions.**
NAFSI Student Data Challenge — Track 3 (SENS-D).

- **Static showcase (GitHub Pages):** pre-computed results + maps →
  `https://<your-username>.github.io/<repo>/`
- **Live interactive app (Streamlit Community Cloud):** full dashboard with
  live controls → _(URL added after deployment)_

---

## What it does

A seven-step pipeline, from raw public-health data to an operational dashboard:

| Step | What it does |
|------|--------------|
| 1 | **Secure ingestion** — FSIS RTE JSON (full outer join on `form_id`, poultry filter) + CDC NORS; live NOAA / USDA NASS / USDA AMS / Census / CDC BEAM Socrata APIs. All credentials via `os.getenv()` + `.env` — never hardcoded. |
| 2 | **Leakage-safe preprocessing** — scikit-learn `ColumnTransformer` (median-impute + scale / one-hot / custom domain transformers), fit on training data only. |
| 3 | **Individual-level tracking + explainability** — out-of-fold (OOF) predictions for every instance across three NORS targets; per-observation SHAP force plots. |
| 4 | **Sequential Bayesian risk** — Beta-Binomial conjugate model; NORS history is the prior `p(w)`, FSIS evidence updates it to the posterior `p(w│𝒟)`. Confirms a 2.5× summer surge. |
| 5 | **Expected loss + danger-weighted optimization** — asymmetric loss matrix `L_kj` (missing severe serotypes costs 20×), Bayes-optimal alert decisions, per-sample danger weights in the gradient. |
| 6 | **BEAM validation + NARMS weighting** — cross-references OOF hotspots against the live CDC BEAM dashboard; dynamic danger weights heaviest for MDR Infantis and ciprofloxacin/DSC resistance. |
| 7 | **Dashboard** — Streamlit app in three sections: Surge Alert Header, Trade-Off Matrix, Equity Impact Map (SVI + Donate-Cooked-Product routing vectors). |

## Repository layout

```
src/            pipeline modules (ingestion, preprocessing, modeling, bayes, loss, …)
app/dashboard.py   Streamlit dashboard (Step 7)
build_site.py      static-site generator for GitHub Pages → docs/
docs/           generated static site (served by GitHub Pages)
data/           public datasets (FSIS sample, NORS, ATSDR SVI 2000–2022)
artifacts/      generated model outputs (gitignored)
notebooks/      dev environment files (.env, full requirements)
requirements.txt   runtime deps for Streamlit Community Cloud
runtime.txt        pins Python 3.12 for Cloud
```

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (use source .venv/bin/activate on macOS/Linux)
pip install -r notebooks/requirements.txt

# copy the credential template and fill in your own API keys
copy notebooks\.env.examples notebooks\.env

# run any pipeline step, e.g.
python -m src.step4_bayes

# launch the dashboard
streamlit run app/dashboard.py
```

## Deploy

**Live app (Streamlit Community Cloud):** connect this repo at
[share.streamlit.io](https://share.streamlit.io), set the main file to
`app/dashboard.py`. Free, redeploys on every push.

**Static site (GitHub Pages):** Settings → Pages → Source: *Deploy from a
branch* → `main` / `/docs`. Regenerate the page with
`python build_site.py --live-app-url <your-streamlit-url>`.

## Security

API credentials live only in `notebooks/.env`, which is **gitignored** and
never committed. Code reads them via `os.getenv()`. See `notebooks/.env.examples`
for the required variable names.

## Data sources

CDC NORS · FSIS RTE Laboratory Sampling · CDC/ATSDR SVI · CDC BEAM Dashboard ·
FSIS NARMS Multi-Year Report · NOAA GHCND · USDA NASS/AMS · U.S. Census.

> **Note on scope.** The public FSIS RTE file is a 20-row *sample* with no
> serotype/AMR signal, so the modeling substrate is CDC NORS (which is rich);
> FSIS is the operational input the deployed system scores. Every step is
> written to run unchanged on the full FSIS export.
