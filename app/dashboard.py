"""Early Signal — Salmonella surge command center (Streamlit).

Run from the project root:
    .venv\\Scripts\\streamlit run app\\dashboard.py

Answers the SENS-D question for a poultry plant manager: is a Salmonella surge
coming for my region this month, what should my lines do about it, and if
product has to be cooked, which vulnerable communities nearby should receive it?

Design notes:
  * Every state and every month is covered. Each region/month starts from its
    CDC NORS outbreak history (Bayesian prior) and is updated with FSIS plant
    samples where they exist.
  * The surge threshold is automatic: "surge" means risk above the normal
    yearly level from NORS history. No manual sensitivity control.
  * The equity section shows the chain from food safety to food security:
    detect the surge, cook the at-risk lots, deliver the safe protein to food
    banks in high-vulnerability (CDC SVI) counties within the service radius.
  * No raw HTML in the page body. Styling is injected with st.html (style-only
    content does not render as text), and all banners/cards are native
    Streamlit components.
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

from src import bayes, decision, geo, ingest_local, routing  # noqa: E402

st.set_page_config(
    page_title="Early Signal — Salmonella Surge Command Center",
    page_icon="🐔",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MONTHS = decision.MONTHS

# Style-only content: Streamlit applies it to the page without rendering text.
st.html(
    """
    <style>
      .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1320px; }
      div[data-testid="stMetric"] { background:#ffffff; border:1px solid #e6e1ea;
          border-radius:10px; padding:.75rem 1rem; box-shadow:0 1px 3px rgba(0,0,0,.05); }
      div[data-testid="stMetricLabel"] p { font-weight:600; color:#555; }
      h2 { color:#4a148c; }
      h3 { color:#6a1b9a; }
    </style>
    """
)


# ---------------------------------------------------------------------------
# Cached backend
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading risk model…")
def load_model():
    priors = bayes.build_priors()
    national = {m: priors[("__NATIONAL__", m)] for m in range(1, 13)}
    normal_level = sum(p.prior_mean for p in national.values()) / 12.0
    try:
        evidence = bayes.fsis_batches_by_cell(poultry_only=True)
    except Exception:
        evidence = {}
    states = sorted({k[0] for k in priors if k[0] != "__NATIONAL__"},
                    key=lambda s: geo.to_name(s) or s)
    return priors, national, normal_level, evidence, states


@st.cache_data(show_spinner="Loading CDC/ATSDR SVI and county locations…")
def load_vulnerability():
    return ingest_local.load_svi_vulnerability(2022)


priors, national, normal_level, evidence, states = load_model()
vuln = load_vulnerability()


def cell_posterior(state: str | None, month: int) -> tuple[bayes.Posterior, bool]:
    """Posterior for a region/month. Returns (posterior, has_local_history)."""
    if state is None:
        return bayes.init_posterior(national[month]), True
    has_local = (state, month) in priors
    prior = bayes.get_prior(priors, state, month)
    post = bayes.init_posterior(prior)
    if (state, month) in evidence:
        k, n = evidence[(state, month)]
        post = bayes.update_posterior(post, k, n)
    return post, has_local


@st.cache_data(show_spinner=False)
def need_share_by_state() -> dict[str, float]:
    return {s: decision.local_need(s, vuln).share_high for s in states}


@st.cache_data(show_spinner=False)
def scan_recommendations() -> pd.DataFrame:
    """Recommendation for every state x month (shown in diagnostics)."""
    shares = need_share_by_state()
    rows = []
    for s in states:
        for m in range(1, 13):
            post, _ = cell_posterior(s, m)
            p = post.prob_exceeds(normal_level)
            rows.append({"state": s, "month": m, "surge_probability": p,
                         "need_share": shares[s],
                         "action": decision.recommend(p, shares[s]).scenario})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Command bar
# ---------------------------------------------------------------------------
st.title("🐔 Early Signal · Salmonella Surge Command Center")
st.caption("Proactive decision support for FSIS-regulated ready-to-eat poultry · "
           "NAFSI SENS-D, Track 3")

NATIONAL = "United States (all regions)"
region_options = [NATIONAL] + states

with st.container(border=True):
    c1, c2, c3 = st.columns([1.3, 1, 1.7])
    with c1:
        region = st.selectbox(
            "📍 Facility region", region_options,
            format_func=lambda s: s if s == NATIONAL else f"{geo.to_name(s)} ({s})",
        )
    with c2:
        month = st.selectbox("🗓️ Month", list(range(1, 13)),
                             index=datetime.now().month - 1,
                             format_func=lambda m: MONTHS[m])
    with c3:
        st.markdown("**🎯 Surge threshold: automatic**")
        st.caption(f"A surge means Salmonella risk above the normal yearly level "
                   f"({normal_level:.0%}), set from CDC outbreak history. "
                   f"Nothing to tune.")

state = None if region == NATIONAL else region
loc_name = "the United States" if state is None else geo.to_name(state)

post, has_local = cell_posterior(state, month)
p_surge = post.prob_exceeds(normal_level)
risk = post.mean
tier = decision.risk_tier(p_surge)

need = decision.local_need(state, vuln) if state else None
rec = decision.recommend(p_surge, need.share_high if need else None)
fsis_k, fsis_n = evidence.get((state, month), (0, 0)) if state else (0, 0)


# ---------------------------------------------------------------------------
# KPIs + plain-English insight
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Risk level", f"{tier.emoji} {tier.label}")
k2.metric("Surge probability", f"{p_surge:.0%}",
          help="Chance that this region's Salmonella risk this month is above "
               "the normal yearly level.")
k3.metric("Expected risk", f"{risk:.0%}",
          delta=f"{(risk - normal_level) * 100:+.0f} pts vs normal",
          delta_color="inverse",
          help="The model's estimate of the share of foodborne outbreaks that are "
               "Salmonella for this region and month.")
k4.metric("Location · Month",
          f"{'US' if state is None else state} · {MONTHS[month][:3]}")

getattr(st, tier.level)(decision.headline(
    "The United States" if state is None else loc_name,
    month, tier, p_surge, normal_level))

st.markdown("**What is driving this**")
for d in decision.drivers(loc_name=loc_name, month=month, risk=risk,
                          national_month=national[month].prior_mean,
                          normal_level=normal_level, has_local_history=has_local,
                          fsis_k=fsis_k, fsis_n=fsis_n):
    st.markdown(f"- {d}")


# ---------------------------------------------------------------------------
# Recommended action + trade-off matrix
# ---------------------------------------------------------------------------
st.header("Recommended action")
with st.container(border=True):
    a1, a2 = st.columns([1, 2])
    a1.subheader(f"▶ {rec.scenario}")
    a2.markdown(rec.why)
    if state is None:
        a2.caption("Select a facility region to include local community need in "
                   "the recommendation.")
st.caption(
    f"How it is chosen: surge probability ≥ {decision.HIGH_SURGE:.0%} with "
    f"≥ {decision.HIGH_NEED_SHARE:.0%} of counties within "
    f"{decision.SERVICE_RADIUS_MI} miles high-vulnerability → Donate Cooked "
    f"Product · ≥ {decision.HIGH_SURGE:.0%} → Divert to Cooked Lines · "
    f"≥ {decision.ELEVATED_SURGE:.0%} → Trigger Enhanced Interventions · "
    f"otherwise → Adjust Transit Radii.")

tdf = routing.trade_off_dataframe().copy()
tdf.insert(0, "Pick", tdf["Scenario"].map(lambda s: "✅" if s == rec.scenario else ""))
_cost = {"Low": "#c8e6c9", "Moderate": "#fff9c4", "High": "#ffcdd2"}
_safety = {"High via Heat kill-step": "#c8e6c9", "High": "#c8e6c9",
           "Moderate": "#fff9c4", "Moderate via Antimicrobial washes": "#fff9c4"}


def _style(df: pd.DataFrame):
    sty = df.style
    sty = sty.map(lambda v: f"background-color:{_cost.get(v, '#fff9c4')};", subset=["Cost"])
    sty = sty.map(lambda v: f"background-color:{_safety.get(v, '#fff9c4')};", subset=["Safety"])
    sty = sty.map(lambda v: "background-color:#c8e6c9;" if v == "100%"
                  else "background-color:#fff9c4;", subset=["Protein Kept"])
    sty = sty.apply(lambda r: ["font-weight:700;"] * len(r)
                    if r["Scenario"] == rec.scenario else [""] * len(r), axis=1)
    return sty.hide(axis="index")


st.subheader("Trade-off matrix")
st.caption("Green = favorable · amber = moderate · red = costly. ✅ marks the "
           "recommendation for the region and month selected above.")
st.table(_style(tdf))


# ---------------------------------------------------------------------------
# Equity impact: food safety -> food security
# ---------------------------------------------------------------------------
st.header("Equity impact: from food safety to food security")
st.markdown(
    "When a surge is flagged, the at-risk raw poultry does not have to be thrown "
    "away. A validated heat kill-step makes it safe, and the cooked protein can go "
    "to food banks serving the communities most exposed to food insecurity and "
    "foodborne illness: older adults, children, people with disabilities or "
    "without health insurance, and households in poverty.")

donate_on = rec.scenario == "Donate Cooked Product"
cook_on = rec.scenario in ("Donate Cooked Product", "Divert to Cooked Lines")
s1, s2, s3 = st.columns(3)
with s1.container(border=True):
    st.markdown(f"**1 · Detect** {'✅' if tier.label != 'BASELINE' else '⏸️'}")
    st.caption(f"Surge probability for {loc_name} in {MONTHS[month]}: "
               f"{p_surge:.0%} ({tier.label.lower()}).")
with s2.container(border=True):
    st.markdown(f"**2 · Neutralize** {'✅' if cook_on else '⏸️'}")
    st.caption("Heat kill-step on the at-risk lots." if cook_on
               else "Not triggered at this risk level.")
with s3.container(border=True):
    st.markdown(f"**3 · Deliver** {'✅' if donate_on else '⏸️'}")
    st.caption("Route cooked protein to food banks in the counties listed below."
               if donate_on else
               "On standby. The counties below are where donations would go if a "
               "surge triggers.")

if state is None:
    st.info("Select a facility region above to see the high-vulnerability "
            "counties within its service radius and plan donation routes.")
else:
    targets = decision.donation_targets(need)
    n1, n2, n3, n4 = st.columns(4)
    n1.metric(f"High-vulnerability counties within {decision.SERVICE_RADIUS_MI} mi",
              f"{len(need.high_need)} of {len(need.nearby)}",
              help=f"Counties with overall CDC SVI ≥ {decision.HIGH_VULN_SVI}.")
    n2.metric("Residents in those counties", f"{need.population / 1e6:.1f} M")
    n3.metric("Adults 65+", f"{need.seniors / 1e6:.2f} M")
    n4.metric("Children 17 and under", f"{need.children / 1e6:.2f} M")

    if need.share_high >= decision.HIGH_NEED_SHARE:
        st.caption(f"Local need is high: {need.share_high:.0%} of nearby counties are "
                   f"high-vulnerability, so a surge here routes product to donation.")
    else:
        st.caption(f"Local need is moderate: {need.share_high:.0%} of nearby counties "
                   f"are high-vulnerability (donation triggers at "
                   f"{decision.HIGH_NEED_SHARE:.0%}), so a surge here diverts product "
                   f"to cooked lines.")

    map_col, list_col = st.columns([3, 2])
    with map_col:
        import folium
        from streamlit_folium import st_folium

        flat, flon = need.facility
        fmap = folium.Map(location=[flat, flon], zoom_start=6, tiles="OpenStreetMap")
        folium.Circle([flat, flon], radius=decision.SERVICE_RADIUS_MI * 1609.34,
                      color="#4a148c", weight=1, fill=False,
                      tooltip=f"{decision.SERVICE_RADIUS_MI}-mile service radius").add_to(fmap)

        def _svi_color(v: float) -> str:
            if v >= 0.75:
                return "#7b1fa2"
            if v >= 0.5:
                return "#ba68c8"
            if v >= 0.25:
                return "#e1bee7"
            return "#f3e5f5"

        for r in need.nearby.itertuples(index=False):
            folium.CircleMarker(
                [r.lat, r.lon], radius=4 if r.svi_overall < 0.75 else 6,
                color=_svi_color(r.svi_overall), fill=True, fill_opacity=0.8, weight=1,
                tooltip=f"{r.county}, {r.state} · SVI {r.svi_overall:.2f}",
            ).add_to(fmap)
        for r in targets.itertuples(index=False):
            folium.PolyLine(
                [[flat, flon], [r.lat, r.lon]],
                color="#d84315" if donate_on else "#9e9e9e",
                weight=3 if donate_on else 2,
                dash_array=None if donate_on else "6 6", opacity=0.85,
                tooltip=f"{'Donation route' if donate_on else 'Standby route'} → "
                        f"{r.county}, {r.state}",
            ).add_to(fmap)
        folium.Marker([flat, flon], tooltip=f"Facility region: {loc_name}",
                      icon=folium.Icon(color="red", icon="industry", prefix="fa")).add_to(fmap)
        st_folium(fmap, height=480, width="100%", returned_objects=[])
        st.caption("Dots: counties within the service radius, darker = more "
                   "vulnerable (CDC SVI). Lines: donation routes to the highest-need "
                   "counties — solid when donation is active, dashed on standby. "
                   "Facility shown at the state center.")

    with list_col:
        st.markdown("**Donation manifest — highest-need counties**")
        manifest = pd.DataFrame({
            "County": targets["county"] + ", " + targets["state"],
            "Miles": targets["distance_mi"].round(0),
            "SVI": targets["svi_overall"],
            "Adults 65+": targets["seniors_65"],
            "Children ≤17": targets["children_17"],
            "Below 150% poverty %": targets["pct_pov150"],
            "Disability %": targets["pct_disabled"],
            "Uninsured %": targets["pct_uninsured"],
        })
        st.dataframe(
            manifest, hide_index=True, width="stretch", height=420,
            column_config={
                "Miles": st.column_config.NumberColumn(format="%d"),
                "SVI": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                "Adults 65+": st.column_config.NumberColumn(format="%d"),
                "Children ≤17": st.column_config.NumberColumn(format="%d"),
                "Below 150% poverty %": st.column_config.NumberColumn(format="%.1f"),
                "Disability %": st.column_config.NumberColumn(format="%.1f"),
                "Uninsured %": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        st.caption("Source: CDC/ATSDR SVI 2022, Census county locations. SVI does "
                   "not measure immunocompromise or malnutrition directly; disability, "
                   "uninsured, and poverty rates are the closest published proxies.")


# ---------------------------------------------------------------------------
# Decision record
# ---------------------------------------------------------------------------
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

st.header("Decision record")
st.caption("Log what the plant did. Overrides need a reason, which gives the team "
           "a record of where people and the model disagree.")
scenario_names = [s.name for s in routing.ROUTING_SCENARIOS]
d1, d2, d3 = st.columns([1, 1, 1.4])
with d1:
    st.markdown(f"System recommends: **{rec.scenario}**")
    accept = st.button("✅ Accept recommendation", width="stretch")
with d2:
    override_choice = st.selectbox("Override with", scenario_names,
                                   index=scenario_names.index(rec.scenario))
with d3:
    reason = st.text_input("Reason (required to override)",
                           placeholder="e.g. line capacity, supplier contract")
    override = st.button("✏️ Record override", width="stretch")


def _log(action: str, chosen: str, why: str) -> None:
    st.session_state.audit_log.append({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "region": "US" if state is None else state, "month": MONTHS[month],
        "surge_threshold": round(normal_level, 4),
        "surge_probability": round(p_surge, 4), "risk_level": tier.label,
        "system_recommended": rec.scenario, "action": action,
        "final_choice": chosen, "reason": why,
    })


if accept:
    _log("accept", rec.scenario, "")
    st.success(f"Logged: accepted {rec.scenario}.")
if override:
    if override_choice == rec.scenario:
        st.warning("That matches the recommendation. Pick a different scenario or "
                   "use Accept.")
    elif not reason.strip():
        st.error("Enter a reason to record an override.")
    else:
        _log("override", override_choice, reason.strip())
        st.success(f"Logged: override to {override_choice} — {reason.strip()}")

if st.session_state.audit_log:
    log_df = pd.DataFrame(st.session_state.audit_log)
    with st.expander(f"📜 Decision log ({len(log_df)} entries)"):
        st.dataframe(log_df, width="stretch", hide_index=True)
        st.download_button("⬇ Download log (CSV)",
                           data=log_df.to_csv(index=False).encode("utf-8"),
                           file_name="early_signal_decision_log.csv", mime="text/csv")
    st.caption("The log is kept for this browser session. A production deployment "
               "would store it in a database.")


# ---------------------------------------------------------------------------
# Detailed diagnostics (opt-in)
# ---------------------------------------------------------------------------
with st.expander("🔬 Detailed diagnostics (for data scientists and judges)"):
    st.markdown("**Bayesian risk model**")
    st.latex(r"p(w\mid\mathcal{D})=\frac{p(\mathcal{D}\mid w)\,p(w)}{p(\mathcal{D})}"
             r"=\mathrm{Beta}(\alpha_0+k,\ \beta_0+n-k)")
    st.markdown(
        f"- Prior from NORS: Beta({post.prior_alpha:.2f}, {post.prior_beta:.2f})"
        f"{'' if has_local else ' (national seasonal prior; little local history)'}\n"
        f"- FSIS evidence: {fsis_k} positive of {fsis_n} tested\n"
        f"- Posterior: Beta({post.alpha:.2f}, {post.beta:.2f}), "
        f"mean E[w|D] = {risk:.4f}\n"
        f"- Surge threshold τ = normal yearly level = {normal_level:.4f} "
        f"(mean of the 12 national monthly priors)\n"
        f"- Surge probability P(w > τ | D) = {p_surge:.4f}")

    st.markdown("**Where each action is recommended (all regions × months)**")
    scan = scan_recommendations()
    counts = scan["action"].value_counts().reindex(scenario_names, fill_value=0)
    st.dataframe(counts.rename("region-months").to_frame(), width="content")
    donate_cells = scan[scan["action"] == "Donate Cooked Product"].copy()
    if not donate_cells.empty:
        donate_cells["region-month"] = donate_cells.apply(
            lambda r: f"{r['state']} {MONTHS[int(r['month'])][:3]}", axis=1)
        st.caption("Donate Cooked Product is recommended in: " +
                   ", ".join(donate_cells.sort_values(["month", "state"])["region-month"]))

    st.markdown("**Seasonal prior (national)**")
    st.bar_chart(pd.DataFrame({"prior risk": [national[m].prior_mean for m in range(1, 13)]},
                              index=[MONTHS[m][:3] for m in range(1, 13)]), height=200)

    figdir = PROJECT_ROOT / "docs" / "figures"
    for label, fname in [
        ("Bayesian prior → posterior update", "step4_bayes_update.png"),
        ("Expected-loss decision boundary", "step5_expected_loss_decision.png"),
        ("Severe-serotype recall", "step5_severe_recall.png"),
        ("OOF calibration (is_salmonella)", "step3_oof_calibration_is_salmonella.png"),
    ]:
        if (figdir / fname).exists():
            st.image(str(figdir / fname), caption=label, width="stretch")

st.divider()
st.caption("Early Signal · NORS/FSIS ingestion → leakage-safe preprocessing → "
           "OOF + SHAP → Bayesian risk → danger-weighted optimization → "
           "BEAM/NARMS validation → this command center.")
