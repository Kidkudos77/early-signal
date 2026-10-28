"""Early Signal — HCXAI command center (Streamlit).

Run from the project root:
    .venv\\Scripts\\streamlit run app\\dashboard.py

A human-centered decision-support tool for poultry plant managers. The raw
statistics (Bayesian posteriors, tau thresholds, log-odds, SHAP) are abstracted
into operational KPIs and plain-English insight at the top level; the technical
proof is tucked behind a "Detailed Diagnostics" expander (progressive
disclosure). The three brief-mandated sections — Surge Alert, Trade-Off Matrix,
Equity Impact Map — remain, reframed as a command center:

  * Top command bar: State / Month / Threshold, dynamically linked.
  * Executive KPI cards + natural-language insight.
  * Auto-recommended routing (backend crosses tau), color-coded matrix.
  * Equity map locked beside the logistics manifest (2:1 columns).
  * Decision record: Accept / Override + reason code + exportable audit log.
  * Detailed Diagnostics: the Bayesian math, SHAP, raw numbers (opt-in).

Honest notes:
  * Risk drivers are derived from what the model actually computes (seasonal +
    geographic effects, SHAP features) — no invented signals.
  * The audit log persists in-session (st.session_state) and is downloadable;
    a production deployment would persist it to a database.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import bayes, ingest_local, routing  # noqa: E402

st.set_page_config(
    page_title="Early Signal — Salmonella Surge Command Center",
    page_icon="🐔",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MONTH_NAMES = ["", "January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
PEAK_MONTHS = (6, 7, 8, 9)


# ---------------------------------------------------------------------------
# Scientific aesthetic — injected CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      /* Tighten the main container; enterprise density. */
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem;
                         max-width: 1280px; }
      /* Command-bar panel look. */
      .cmd-bar { background: linear-gradient(135deg,#1a1a2e,#2a1a3e);
                 border-radius: 12px; padding: 1.1rem 1.4rem; margin-bottom: 1rem; }
      /* KPI metric cards: stronger hierarchy. */
      div[data-testid="stMetric"] { background:#ffffff; border:1px solid #e6e1ea;
            border-radius:10px; padding:0.8rem 1rem; box-shadow:0 1px 3px rgba(0,0,0,.06); }
      div[data-testid="stMetricLabel"] { font-weight:600; color:#555; }
      /* Section headers. */
      h2 { color:#4a148c; border-bottom:2px solid #efe7f3; padding-bottom:.3rem; }
      h3 { color:#6a1b9a; }
      /* Risk banner. */
      .risk-banner { border-radius:10px; padding:1rem 1.3rem; color:#fff;
                     font-size:1.15rem; font-weight:600; margin:.4rem 0 1rem; }
      /* Recommended action chip. */
      .rec-chip { display:inline-block; background:#4a148c; color:#fff;
                  padding:.5rem 1rem; border-radius:999px; font-weight:700; }
      .stApp { background:#f7f5fa; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached backend
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading risk model…")
def load_backend():
    priors = bayes.build_priors()
    national = {m: priors[("__NATIONAL__", m)].prior_mean for m in range(1, 13)}
    national_overall = sum(national.values()) / 12.0
    try:
        posts = bayes.posteriors_for_fsis(poultry_only=True)
    except Exception:
        posts = {}
    posteriors = {
        (st_, mo): {
            "state": st_, "month": mo,
            "prior_mean": p.prior_mean, "posterior_mean": p.mean,
            "k": p.evidence_k, "n": p.evidence_n,
            "alpha": p.alpha, "beta": p.beta,
            "prior_alpha": p.prior_alpha, "prior_beta": p.prior_beta,
        }
        for (st_, mo), p in posts.items()
    }
    return priors, national, national_overall, posteriors


@st.cache_data(show_spinner="Loading CDC/ATSDR SVI…")
def load_svi():
    return ingest_local.load_svi(2022)


priors, national, national_overall, posteriors = load_backend()

# States/months that have real FSIS evidence cells.
evidence_states = sorted({k[0] for k in posteriors})
STATE_OPTIONS = ["National (seasonal baseline)"] + evidence_states


# ---------------------------------------------------------------------------
# Session state (audit log + dynamic filters)
# ---------------------------------------------------------------------------
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []
if "state_sel" not in st.session_state:
    st.session_state.state_sel = STATE_OPTIONS[0]


def months_for_state(state_label: str) -> list[int]:
    """Months available for the chosen state (dynamic linkage)."""
    if state_label.startswith("National"):
        return list(range(1, 13))
    months = sorted({mo for (st_, mo) in posteriors if st_ == state_label})
    return months or list(range(1, 13))


def _on_state_change():
    # Reset month to the first valid one for the new state.
    st.session_state.month_sel = months_for_state(st.session_state.state_sel)[0]


# ---------------------------------------------------------------------------
# COMMAND BAR — filters front and center
# ---------------------------------------------------------------------------
st.markdown('<div class="cmd-bar">', unsafe_allow_html=True)
st.markdown("### 🐔 Early Signal · Salmonella Surge Command Center")
st.caption("Proactive decision support for FSIS-regulated RTE poultry · "
           "NAFSI SENS-D, Track 3")

cbar = st.columns([1.1, 1.1, 1.3])
with cbar[0]:
    state_sel = st.selectbox(
        "📍 Facility / region", STATE_OPTIONS, key="state_sel",
        on_change=_on_state_change,
    )
with cbar[1]:
    avail_months = months_for_state(state_sel)
    if "month_sel" not in st.session_state or \
            st.session_state.month_sel not in avail_months:
        st.session_state.month_sel = avail_months[0] if state_sel.startswith("National") \
            else avail_months[0]
    month_sel = st.selectbox(
        "🗓️ Month", avail_months, key="month_sel",
        format_func=lambda m: MONTH_NAMES[m],
    )
with cbar[2]:
    tau = st.slider(
        "🎚️ Alert sensitivity", min_value=0.05, max_value=0.50, value=0.20,
        step=0.01,
        help="How cautious the system is. Lower = alerts sooner. "
             "(Technically: the risk threshold τ for P(risk > τ).)",
    )
st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Resolve the selected cell -> posterior + derived quantities
# ---------------------------------------------------------------------------
is_baseline = state_sel.startswith("National")
if is_baseline:
    pm = national[month_sel]
    prior_mean = pm
    cell = {"state": "US", "month": month_sel, "posterior_mean": pm,
            "prior_mean": pm, "k": 0, "n": 0,
            "alpha": None, "beta": None, "prior_alpha": None, "prior_beta": None}
    p_exceeds = None  # no Beta posterior to integrate for the national curve
else:
    cell = posteriors[(state_sel, month_sel)]
    pm = cell["posterior_mean"]
    prior_mean = cell["prior_mean"]
    post_obj = bayes.Posterior(
        state=cell["state"], month=cell["month"],
        alpha=cell["alpha"], beta=cell["beta"],
        prior_alpha=cell["prior_alpha"], prior_beta=cell["prior_beta"],
        evidence_k=cell["k"], evidence_n=cell["n"],
    )
    p_exceeds = post_obj.prob_exceeds(tau)

tier_label, tier_emoji, tier_color = routing.risk_tier(p_exceeds, pm)
rec_name, rec_reason = routing.recommend_scenario(p_exceeds, pm, tau)
is_peak = month_sel in PEAK_MONTHS
drivers = routing.risk_drivers(
    state=cell["state"], month=month_sel, posterior_mean=pm, prior_mean=prior_mean,
    national_month=national[month_sel], national_overall=national_overall,
    is_peak_season=is_peak,
)


# ===========================================================================
# EXECUTIVE KPIs + natural-language insight
# ===========================================================================
loc_name = "National" if is_baseline else cell["state"]
baseline_delta = (pm - national[month_sel]) * 100 if not is_baseline else None

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Risk level", f"{tier_emoji} {tier_label}")
with k2:
    st.metric("Surge probability", f"{pm:.0%}",
              delta=(f"{baseline_delta:+.0f} pts vs national"
                     if baseline_delta is not None else None))
with k3:
    if p_exceeds is not None:
        st.metric("Alert confidence", f"{p_exceeds:.0%}",
                  help="Chance the true risk exceeds your alert sensitivity.")
    else:
        st.metric("Alert confidence", "—")
with k4:
    st.metric("Location · Month", f"{loc_name} · {MONTH_NAMES[month_sel][:3]}")

# Risk banner (colored by tier) + natural-language insight.
season_word = "peak season" if is_peak else "off-peak season"
vs_base = pm - national[month_sel]
nl_insight = (
    f"<b>Risk Level: {tier_label.title()}.</b> "
    f"{loc_name} in {MONTH_NAMES[month_sel]} ({season_word}) — "
    f"surge probability {pm:.0%}"
)
if not is_baseline and abs(vs_base) > 0.005:
    nl_insight += (f", {abs(vs_base) * 100:.0f} pts "
                   f"{'above' if vs_base > 0 else 'below'} the national level "
                   f"for this month.")
else:
    nl_insight += "."
st.markdown(
    f'<div class="risk-banner" style="background:{tier_color};">{nl_insight}</div>',
    unsafe_allow_html=True,
)

# Layer 1 — plain-English risk drivers.
st.markdown("**Primary risk drivers**")
for d in drivers:
    st.markdown(f"- {d}")


# ===========================================================================
# SECTION 2 — Trade-Off Matrix (auto-recommended, color-coded)
# ===========================================================================
st.markdown("## Recommended Action")

rc1, rc2 = st.columns([1, 2])
with rc1:
    st.markdown(f'<div class="rec-chip">▶ {rec_name}</div>',
                unsafe_allow_html=True)
with rc2:
    st.markdown(f"_{rec_reason}_")

# Color-coded trade-off matrix via pandas Styler, recommended row flagged.
tdf = routing.trade_off_dataframe().copy()
tdf.insert(0, "", tdf["Scenario"].apply(lambda s: "✅" if s == rec_name else ""))

_cost_rank = {"Low": 0, "Moderate": 1, "High": 2, "Variable": 1}
_safety_rank = {"Moderate via Antimicrobial washes": 0, "Moderate": 0,
                "High via Heat kill-step": 1, "High": 1}


def _style_matrix(df: pd.DataFrame):
    sty = df.style
    # Numeric helper columns for gradients (hidden).
    cost_num = df["Cost"].map(_cost_rank).fillna(1)
    safe_num = df["Safety"].map(_safety_rank).fillna(0)

    def cost_color(val):
        r = _cost_rank.get(val, 1)
        c = ["#c8e6c9", "#fff9c4", "#ffcdd2"][int(r)]
        return f"background-color: {c};"

    def safety_color(val):
        r = _safety_rank.get(val, 0)
        c = ["#fff9c4", "#c8e6c9"][int(r)]
        return f"background-color: {c};"

    def protein_color(val):
        return ("background-color: #c8e6c9;" if str(val).strip() == "100%"
                else "background-color: #fff9c4;")

    sty = sty.map(cost_color, subset=["Cost"])
    sty = sty.map(safety_color, subset=["Safety"])
    sty = sty.map(protein_color, subset=["Protein Kept"])

    def highlight_row(row):
        if row["Scenario"] == rec_name:
            return ["font-weight:700; border-top:2px solid #4a148c; "
                    "border-bottom:2px solid #4a148c;"] * len(row)
        return [""] * len(row)

    sty = sty.apply(highlight_row, axis=1)
    return sty


st.markdown("### Trade-Off Matrix")
st.caption("Color legend: green = favorable, amber = moderate, red = costly. "
           "The ✅ row is the system's recommendation at your current settings.")
st.table(_style_matrix(tdf))


# ===========================================================================
# SECTION 3 — Equity Impact Map + logistics manifest (2:1)
# ===========================================================================
st.markdown("## Equity Impact Map")
st.caption("Where cooked donated product should flow: highest-risk sources → "
           "highest-vulnerability (CDC SVI) counties.")

svi = load_svi()
risk_by_state = {k[0]: posteriors[k]["posterior_mean"] for k in posteriors}
if not risk_by_state:
    risk_by_state = {"CA": national[month_sel], "TX": national[month_sel]}
vectors = routing.donate_routing_vectors(risk_by_state, svi)

map_col, manifest_col = st.columns([2, 1])

with map_col:
    try:
        import folium
        from streamlit_folium import st_folium
        m = folium.Map(location=[39.5, -98.35], zoom_start=4,
                       tiles="OpenStreetMap")
        hv = routing.high_vulnerability_counties(svi).head(60)
        for row in hv.itertuples(index=False):
            lat, lon = routing.STATE_CENTROIDS.get(row.state, routing.US_CENTER)
            folium.CircleMarker(
                location=[lat, lon], radius=3 + 6 * float(row.svi_overall),
                color="#6a1b9a", fill=True, fill_opacity=0.5,
                tooltip=f"{row.county}, {row.state} — SVI {row.svi_overall:.2f}",
            ).add_to(m)
        for v in vectors.itertuples(index=False):
            folium.PolyLine(
                locations=[[v.src_lat, v.src_lon], [v.tgt_lat, v.tgt_lon]],
                color="#d84315", weight=2 + 3 * float(v.source_risk), opacity=0.7,
                tooltip=(f"Donate: {v.source_state} → {v.target_county}, "
                         f"{v.target_state} (SVI {v.target_svi:.2f})"),
            ).add_to(m)
            folium.CircleMarker(
                location=[v.src_lat, v.src_lon], radius=6, color="#d84315",
                fill=True, fill_opacity=0.9,
                tooltip=f"Source: {v.source_state} risk {v.source_risk:.2f}",
            ).add_to(m)
        st_folium(m, height=460, width="100%")
    except Exception:
        import plotly.graph_objects as go
        fig = go.Figure()
        cs = svi.groupby("state")["svi_overall"].mean().reset_index()
        fig.add_trace(go.Choropleth(
            locations=cs["state"], locationmode="USA-states",
            z=cs["svi_overall"], colorscale="Purples",
            colorbar=dict(title="mean SVI")))
        for v in vectors.itertuples(index=False):
            fig.add_trace(go.Scattergeo(
                lon=[v.src_lon, v.tgt_lon], lat=[v.src_lat, v.tgt_lat],
                mode="lines", line=dict(width=2, color="#d84315"),
                opacity=0.7, showlegend=False))
        fig.update_layout(geo=dict(scope="usa"), height=460,
                          margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, width="stretch")

with manifest_col:
    st.markdown("**Logistics manifest**")
    st.caption("Donate Cooked Product targets (high-SVI counties)")
    manifest = vectors[["target_county", "target_state", "target_svi",
                        "source_state"]].copy()
    manifest.columns = ["Target county", "ST", "SVI", "From"]
    manifest["SVI"] = manifest["SVI"].map(lambda x: f"{x:.2f}")
    st.dataframe(manifest, height=420, hide_index=True, width="stretch")


# ===========================================================================
# DECISION RECORD — Accept / Override + audit log
# ===========================================================================
st.markdown("## Decision Record")
st.caption("Log the action taken. Overrides require a reason — this is the "
           "cheapest model evaluation your team will run.")

dr1, dr2, dr3 = st.columns([1, 1, 1.4])
with dr1:
    st.markdown(f"System recommends: **{rec_name}**")
    accept = st.button("✅ Accept recommendation", width="stretch")
with dr2:
    override_choice = st.selectbox(
        "Override with", [s.name for s in routing.ROUTING_SCENARIOS],
        index=[s.name for s in routing.ROUTING_SCENARIOS].index(rec_name),
    )
with dr3:
    reason = st.text_input("Reason code (required to override)",
                           placeholder="e.g. line capacity, supplier contract…")
    override = st.button("✏️ Record override", width="stretch")


def _log(action: str, chosen: str, reason_txt: str):
    st.session_state.audit_log.append({
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "location": loc_name, "month": MONTH_NAMES[month_sel],
        "tau_in_force": round(tau, 2),
        "surge_probability": round(pm, 4),
        "risk_tier": tier_label,
        "system_recommended": rec_name,
        "action": action, "human_choice": chosen,
        "reason": reason_txt,
    })


if accept:
    _log("accept", rec_name, "")
    st.success(f"Logged: accepted **{rec_name}**.")
if override:
    if override_choice == rec_name:
        st.warning("Override choice matches the recommendation — pick a "
                   "different scenario, or use Accept.")
    elif not reason.strip():
        st.error("A reason code is required to override the recommendation.")
    else:
        _log("override", override_choice, reason.strip())
        st.success(f"Logged: overrode to **{override_choice}** — {reason.strip()}")

if st.session_state.audit_log:
    log_df = pd.DataFrame(st.session_state.audit_log)
    with st.expander(f"📜 Decision audit log ({len(log_df)} entries)", expanded=False):
        st.dataframe(log_df, width="stretch", hide_index=True)
        st.download_button(
            "⬇ Download audit log (CSV)",
            data=log_df.to_csv(index=False).encode("utf-8"),
            file_name="early_signal_decision_log.csv", mime="text/csv",
        )
    st.caption("Note: this log lives in your browser session. A production "
               "deployment would persist it to a database.")


# ===========================================================================
# DETAILED DIAGNOSTICS — progressive disclosure (Layer 2)
# ===========================================================================
st.markdown("## ")
with st.expander("🔬 View Detailed Diagnostics (for data scientists & judges)"):
    st.markdown("### The statistics behind the recommendation")

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Bayesian risk model (Step 4)**")
        if not is_baseline:
            st.markdown(
                f"- Prior: `Beta(α₀={cell['prior_alpha']:.2f}, "
                f"β₀={cell['prior_beta']:.2f})` from NORS history\n"
                f"- FSIS evidence: `{cell['k']}` positive of `{cell['n']}` tested\n"
                f"- Posterior: `Beta(α={cell['alpha']:.2f}, β={cell['beta']:.2f})`\n"
                f"- Posterior mean E[w|𝒟] = **{pm:.4f}**\n"
                f"- P(w > τ={tau:.2f}) = **{p_exceeds:.4f}**"
            )
        else:
            st.markdown(
                f"- National seasonal prior mean for {MONTH_NAMES[month_sel]} = "
                f"**{pm:.4f}**\n- Select a facility/region cell to see the full "
                f"Bayesian update with FSIS evidence."
            )
        st.latex(r"p(w\mid\mathcal{D})=\frac{p(\mathcal{D}\mid w)\,p(w)}{p(\mathcal{D})}"
                 r"=\mathrm{Beta}(\alpha_0+k,\ \beta_0+n-k)")

    with d2:
        st.markdown("**Seasonal prior curve (NORS)**")
        season_df = pd.DataFrame({
            "month": [MONTH_NAMES[m][:3] for m in range(1, 13)],
            "prior risk": [national[m] for m in range(1, 13)],
        }).set_index("month")
        st.bar_chart(season_df, height=220)

    st.markdown("**Decision rule**")
    st.markdown(
        f"The recommended action is chosen automatically: the actionable signal "
        f"is P(risk > τ) = "
        f"{'%.3f' % p_exceeds if p_exceeds is not None else 'n/a (baseline, uses posterior mean)'}, "
        f"mapped to tiers (≥0.50 → Divert to Cooked Lines; ≥0.20 → Enhanced "
        f"Interventions; else → Adjust Transit Radii)."
    )

    # Link to the pre-rendered SHAP / math figures if present in docs/figures.
    figdir = PROJECT_ROOT / "docs" / "figures"
    shap_imgs = {
        "SHAP feature importance (is_salmonella)": "summary_is_salmonella.png"
        if (PROJECT_ROOT / "artifacts" / "shap" / "summary_is_salmonella.png").exists()
        else None,
    }
    st.markdown("**Explainability (SHAP, Step 3) & pipeline math**")
    shown = False
    for label, fname in [
        ("Bayesian prior → posterior update", "step4_bayes_update.png"),
        ("Expected-loss decision boundary (Step 5)", "step5_expected_loss_decision.png"),
        ("Severe-serotype recall improvement (Step 5)", "step5_severe_recall.png"),
        ("SHAP feature importance (Step 3)", "summary_is_salmonella.png"),
    ]:
        p = figdir / fname
        alt = (PROJECT_ROOT / "artifacts" / "shap" / fname)
        src = p if p.exists() else (alt if alt.exists() else None)
        if src is not None:
            st.image(str(src), caption=label, width="stretch")
            shown = True
    if not shown:
        st.info("Run `python -m src.figures` to generate the diagnostic figures.")


st.divider()
st.caption("Early Signal · NORS/FSIS ingestion → leakage-safe preprocessing → "
           "OOF + SHAP → Bayesian risk → danger-weighted optimization → "
           "BEAM/NARMS validation → this command center.")
