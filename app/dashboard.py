"""Step 7 — Early Signal decision-support dashboard (Streamlit).

Run from the project root:
    .venv\\Scripts\\streamlit run app\\dashboard.py

Partitioned into EXACTLY three sections per the brief:
  1. Surge Alert Header   — Bayesian posterior probability of a spike (Step 4).
  2. Trade-Off Matrix     — the four exact routing scenarios.
  3. Equity Impact Map    — Folium map overlaying CDC SVI by county with the
                            "Donate Cooked Product" routing vectors.

All data is pulled from the pipeline modules (src.bayes, src.ingest_local,
src.routing). The app degrades gracefully if the live NORS/SVI loads are slow,
and never prints credential values.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when Streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import bayes, ingest_local, routing  # noqa: E402

st.set_page_config(
    page_title="Early Signal — RTE Poultry Salmonella Surge Decision Support",
    page_icon="🐔",
    layout="wide",
)

MONTH_NAMES = ["", "January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]


# ---------------------------------------------------------------------------
# Cached data loads (keep the UI responsive)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Building Bayesian priors from NORS...")
def load_priors():
    priors = bayes.build_priors()
    national = {m: priors[("__NATIONAL__", m)].prior_mean for m in range(1, 13)}
    return priors, national


@st.cache_data(show_spinner="Updating posteriors with FSIS evidence...")
def load_posteriors():
    # FSIS evidence is local-only (no live API), so this won't hang on a hosted
    # deployment. Guard anyway: a missing/partial FSIS sample yields an empty
    # dict and the app falls back to the national seasonal baseline.
    try:
        posts = bayes.posteriors_for_fsis(poultry_only=True)
    except Exception:
        return {}
    return {
        f"{st_}|{mo}": {
            "state": st_, "month": mo,
            "prior_mean": p.prior_mean, "posterior_mean": p.mean,
            "p_exceeds_20": p.prob_exceeds(0.20),
            "ci": p.credible_interval(0.95),
            "k": p.evidence_k, "n": p.evidence_n,
        }
        for (st_, mo), p in posts.items()
    }


@st.cache_data(show_spinner="Loading CDC/ATSDR SVI (2022)...")
def load_svi():
    return ingest_local.load_svi(2022)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.title("Early Signal")
st.sidebar.caption("Proactive Salmonella surge decision support for "
                   "FSIS-regulated RTE poultry (NAFSI SENS-D, Track 3).")

priors, national = load_priors()
posteriors = load_posteriors()

cell_keys = sorted(posteriors.keys())
cell_labels = {
    k: f"{posteriors[k]['state']} — {MONTH_NAMES[posteriors[k]['month']]}"
    for k in cell_keys
}
options = ["National seasonal baseline"] + [cell_labels[k] for k in cell_keys]
choice = st.sidebar.selectbox(
    "Supply-chain cell (state × month)", options,
    help="Pick an FSIS poultry evidence cell to see its Bayesian posterior, "
         "or the national seasonal baseline.",
)
alert_tau = st.sidebar.slider(
    "Alert threshold τ", min_value=0.05, max_value=0.50, value=0.20, step=0.01,
    help="Risk level above which a surge alert fires: shows P(w > τ).",
)

# Resolve the selected posterior (or national baseline).
if choice == "National seasonal baseline":
    sel_month = st.sidebar.slider("Month (baseline)", 1, 12, 7)
    sel = {
        "state": "US", "month": sel_month,
        "prior_mean": national[sel_month], "posterior_mean": national[sel_month],
        "p_exceeds_20": None, "ci": None, "k": 0, "n": 0, "is_baseline": True,
    }
else:
    key = next(k for k in cell_keys if cell_labels[k] == choice)
    sel = {**posteriors[key], "is_baseline": False}


# ===========================================================================
# SECTION 1 — Surge Alert Header
# ===========================================================================
st.title("Early Signal — Salmonella Surge Decision Support")

posterior_mean = sel["posterior_mean"]
# P(w > tau) from the live Beta posterior for the selected cell (recomputed at
# the chosen tau; the cached value is for tau=0.20).
if sel.get("is_baseline"):
    p_spike = None
else:
    st_, mo = sel["state"], sel["month"]
    prior = bayes.get_prior(priors, st_, mo)
    post = bayes.update_posterior(bayes.init_posterior(prior), sel["k"], sel["n"])
    p_spike = post.prob_exceeds(alert_tau)

st.header("1 · Surge Alert Header")
c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    loc = "National" if sel.get("is_baseline") else sel["state"]
    st.metric(
        label=f"Bayesian posterior surge probability — {loc}, {MONTH_NAMES[sel['month']]}",
        value=f"{posterior_mean:.1%}",
        delta=(None if sel.get("is_baseline")
               else f"{(posterior_mean - sel['prior_mean']) * 100:+.1f} pts vs prior"),
        help="Posterior mean E[w|D] of the Beta-Binomial model (Step 4).",
    )
with c2:
    if p_spike is not None:
        st.metric(label=f"P(risk > τ={alert_tau:.2f})", value=f"{p_spike:.1%}")
    else:
        st.metric(label="P(risk > τ)", value="n/a (baseline)")
with c3:
    if sel["n"] > 0:
        st.metric(label="FSIS evidence absorbed",
                  value=f"{sel['k']} pos / {sel['n']} tested")
    else:
        st.metric(label="FSIS evidence absorbed", value="none (prior only)")

# Alert banner driven by P(w > tau).
if p_spike is not None and p_spike >= 0.5:
    st.error(f"🔴 SURGE ALERT: posterior probability risk exceeds τ is "
             f"{p_spike:.0%}. Recommend protective routing (see Trade-Off Matrix).")
elif p_spike is not None and p_spike >= 0.2:
    st.warning(f"🟠 Elevated risk: P(risk > τ) = {p_spike:.0%}. Monitor and "
               f"consider low-cost interventions.")
else:
    season = "peak season" if sel["month"] in (6, 7, 8, 9) else "off-peak season"
    st.info(f"🟢 Baseline risk ({season}). Posterior surge probability "
            f"{posterior_mean:.1%}.")

with st.expander("How this number is computed"):
    st.markdown(
        "- Historical CDC NORS outbreaks set the **prior** `p(w)` = Beta(α₀, β₀) "
        "per (state, month), encoding the seasonal + geographic baseline.\n"
        "- New FSIS supply-chain samples provide **Binomial evidence** "
        "(k positive of n).\n"
        "- Bayes' theorem gives the **posterior** `p(w|𝒟)` = "
        "Beta(α₀+k, β₀+n−k); the headline is its mean E[w|𝒟]."
    )

# National seasonal curve for context.
st.caption("National seasonal baseline risk by month (NORS prior):")
season_df = pd.DataFrame({
    "month": [MONTH_NAMES[m] for m in range(1, 13)],
    "prior_risk": [national[m] for m in range(1, 13)],
}).set_index("month")
st.bar_chart(season_df, height=180)


# ===========================================================================
# SECTION 2 — Trade-Off Matrix
# ===========================================================================
st.header("2 · Trade-Off Matrix")
st.caption("Operational routing scenarios and their safety / protein / cost "
           "trade-offs.")

tdf = routing.trade_off_dataframe()
st.table(tdf)

with st.expander("Scenario rationale"):
    for s in routing.ROUTING_SCENARIOS:
        st.markdown(f"**{s.name}** — {s.rationale}")

# Highlight the recommended action given the current alert level.
if p_spike is not None and p_spike >= 0.5:
    rec = "Divert to Cooked Lines"  # high safety, keeps protein, moderate cost
elif p_spike is not None and p_spike >= 0.2:
    rec = "Trigger Enhanced Interventions"  # low cost, moderate safety
else:
    rec = "Adjust Transit Radii"  # low cost, baseline risk
st.success(f"Recommended action at current risk level: **{rec}**")


# ===========================================================================
# SECTION 3 — Equity Impact Map
# ===========================================================================
st.header("3 · Equity Impact Map")
st.caption("CDC/ATSDR Social Vulnerability Index (SVI) by county, overlaid "
           "with **Donate Cooked Product** routing vectors from highest-risk "
           "states to highest-vulnerability counties.")

svi = load_svi()

# Risk-by-state for routing sources: posterior means where we have FSIS cells,
# else the national month baseline for the selected month.
risk_by_state = {
    posteriors[k]["state"]: posteriors[k]["posterior_mean"] for k in cell_keys
}
if not risk_by_state:
    risk_by_state = {"CA": national[sel["month"]], "TX": national[sel["month"]]}

vectors = routing.donate_routing_vectors(risk_by_state, svi)

try:
    import folium
    from streamlit_folium import st_folium  # optional; fall back if absent
    _have_stfolium = True
except Exception:
    _have_stfolium = False

if _have_stfolium:
    import folium

    # OpenStreetMap tiles need no API key (CartoDB now requires one).
    m = folium.Map(location=[39.5, -98.35], zoom_start=4, tiles="OpenStreetMap")

    # County SVI choropleth via the CDC/ATSDR county GeoJSON (US counties).
    # We color by svi_overall using a circle-marker fallback keyed by state
    # centroid density to avoid shipping a 3k-polygon GeoJSON in-app; the
    # high-vulnerability counties are emphasized as markers.
    hv = routing.high_vulnerability_counties(svi).head(60)
    for row in hv.itertuples(index=False):
        lat, lon = routing.STATE_CENTROIDS.get(row.state, routing.US_CENTER)
        folium.CircleMarker(
            location=[lat, lon], radius=3 + 6 * float(row.svi_overall),
            color="#6a1b9a", fill=True, fill_opacity=0.5,
            tooltip=f"{row.county}, {row.state} — SVI {row.svi_overall:.2f}",
        ).add_to(m)

    # Donate Cooked Product routing vectors (source state -> target county).
    for v in vectors.itertuples(index=False):
        folium.PolyLine(
            locations=[[v.src_lat, v.src_lon], [v.tgt_lat, v.tgt_lon]],
            color="#d84315", weight=2 + 3 * float(v.source_risk), opacity=0.7,
            tooltip=(f"Donate: {v.source_state} (risk {v.source_risk:.2f}) → "
                     f"{v.target_county}, {v.target_state} (SVI {v.target_svi:.2f})"),
        ).add_to(m)
        folium.CircleMarker(
            location=[v.src_lat, v.src_lon], radius=6, color="#d84315",
            fill=True, fill_opacity=0.9,
            tooltip=f"Source: {v.source_state} surge risk {v.source_risk:.2f}",
        ).add_to(m)

    st_folium(m, width=1100, height=520)
else:
    # Plotly fallback (no streamlit-folium): choropleth + vector lines.
    import plotly.graph_objects as go
    fig = go.Figure()
    county_state = svi.groupby("state")["svi_overall"].mean().reset_index()
    fig.add_trace(go.Choropleth(
        locations=county_state["state"], locationmode="USA-states",
        z=county_state["svi_overall"], colorscale="Purples",
        colorbar=dict(title="mean SVI"),
    ))
    for v in vectors.itertuples(index=False):
        fig.add_trace(go.Scattergeo(
            lon=[v.src_lon, v.tgt_lon], lat=[v.src_lat, v.tgt_lat],
            mode="lines", line=dict(width=1 + 3 * float(v.source_risk),
                                    color="#d84315"),
            opacity=0.7, showlegend=False,
            hovertext=(f"Donate: {v.source_state} → {v.target_county}, "
                       f"{v.target_state}"),
        ))
    fig.update_layout(geo=dict(scope="usa"), height=520,
                      title="SVI by state with Donate Cooked Product vectors")
    st.plotly_chart(fig, width="stretch")
    st.info("Install `streamlit-folium` for the interactive county Folium map.")

with st.expander("Donate Cooked Product routing vectors (table)"):
    st.dataframe(
        vectors[["source_state", "source_risk", "target_county",
                 "target_state", "target_svi"]],
        width="stretch",
    )

st.divider()
st.caption("Early Signal pipeline: NORS/FSIS ingestion → leakage-safe "
           "preprocessing → OOF + SHAP → Bayesian risk → danger-weighted "
           "optimization → BEAM/NARMS validation → this dashboard.")
