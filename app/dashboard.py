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
        from folium.plugins import AntPath, PolyLineTextPath
        from streamlit_folium import st_folium

        flat, flon = need.facility
        route_color = "#d84315" if donate_on else "#757575"

        # --- Legend (dynamic: counts and route status follow the selection) ---
        class_counts = {label: 0 for _, label, _ in decision.SVI_CLASSES}
        for v in need.nearby["svi_overall"]:
            class_counts[decision.svi_class(v)[0]] += 1
        swatches = "".join(
            f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0;">'
            f'<span style="width:12px;height:12px;border-radius:50%;background:{color};'
            f'border:1px solid #888;display:inline-block;"></span>'
            f'<span>{label} — {class_counts[label]} counties</span></div>'
            for _, label, color in decision.SVI_CLASSES
        )
        route_line = (
            f'<span style="display:inline-block;width:28px;border-top:3px solid {route_color};'
            f'{"" if donate_on else "border-top-style:dashed;"}vertical-align:middle;"></span>')
        status = ("Donation <b>active</b> — moving dashes show product flowing out "
                  "to each county." if donate_on else
                  "Donation on <b>standby</b> — routes are planned but not running. "
                  "They switch on when the recommendation is Donate Cooked Product.")
        st.html(f"""
        <div style="font-family:sans-serif;font-size:13px;border:1px solid #e0d7e6;
                    border-radius:8px;padding:8px 12px;background:#fff;">
          <div style="display:flex;flex-wrap:wrap;gap:22px;">
            <div>
              <div style="font-weight:700;margin-bottom:3px;">County dots · CDC SVI vulnerability</div>
              {swatches}
              <div style="color:#666;font-size:12px;">Only counties within
              {decision.SERVICE_RADIUS_MI} miles of the facility are shown.</div>
            </div>
            <div>
              <div style="font-weight:700;margin-bottom:3px;">Logistics</div>
              <div style="margin:2px 0;">🏭 Facility region ({loc_name}, at the state center)</div>
              <div style="margin:2px 0;"><span style="display:inline-block;width:18px;height:18px;
                   border-radius:50%;background:{route_color};color:#fff;text-align:center;
                   font-size:11px;line-height:18px;font-weight:700;">1</span>
                   Receiving county, numbered by delivery priority (matches the manifest)</div>
              <div style="margin:2px 0;">{route_line} ➤ Route: runs <b>from</b> the facility
                   <b>to</b> the county; arrows point the way product travels. Thicker = larger
                   share of product.</div>
              <div style="margin:2px 0;">◯ Dashed purple ring: {decision.SERVICE_RADIUS_MI}-mile service radius</div>
              <div style="margin-top:4px;">{status}</div>
            </div>
          </div>
        </div>""")

        # --- Map ---
        fmap = folium.Map(location=[flat, flon], zoom_start=6, tiles="OpenStreetMap")
        folium.Circle([flat, flon], radius=decision.SERVICE_RADIUS_MI * 1609.34,
                      color="#4a148c", weight=1.5, dash_array="6 6", fill=False,
                      tooltip=f"{decision.SERVICE_RADIUS_MI}-mile service radius").add_to(fmap)

        target_fips = set(targets["FIPS"])
        for r in need.nearby.itertuples(index=False):
            if r.FIPS in target_fips:
                continue  # drawn below as numbered stops
            label, color = decision.svi_class(r.svi_overall)
            folium.CircleMarker(
                [r.lat, r.lon], radius=5 if r.svi_overall >= decision.HIGH_VULN_SVI else 3.5,
                color="#666", weight=0.5, fill=True, fill_color=color, fill_opacity=0.9,
                tooltip=(f"{r.county}, {r.state} · vulnerability percentile "
                         f"{r.svi_overall * 100:.1f} ({label}) · "
                         f"{r.distance_mi:.0f} mi · click for profile"),
            ).add_to(fmap)

        for r in targets.itertuples(index=False):
            weight = 2 + 8 * float(r.allocation)
            path = [[flat, flon], [r.lat, r.lon]]
            tip = (f"#{r.priority} {r.county}, {r.state} · {r.allocation:.0%} of product · "
                   f"{r.distance_mi:.0f} mi")
            if donate_on:
                line = AntPath(path, color=route_color, weight=weight, opacity=0.9,
                               delay=900, dash_array=[12, 18], pulse_color="#ffffff",
                               tooltip=tip)
            else:
                line = folium.PolyLine(path, color=route_color, weight=weight,
                                       opacity=0.7, dash_array="8 8", tooltip=tip)
            line.add_to(fmap)
            PolyLineTextPath(line, "      ➤      ", repeat=True, offset=5,
                             attributes={"fill": route_color, "font-size": "14",
                                         "font-weight": "bold"}).add_to(fmap)
            folium.Marker(
                [r.lat, r.lon], tooltip=tip + " · click for profile",
                icon=folium.DivIcon(
                    icon_size=(24, 24), icon_anchor=(12, 12),
                    html=(f'<div style="width:24px;height:24px;border-radius:50%;'
                          f'background:{route_color};color:#fff;border:2px solid #fff;'
                          f'box-shadow:0 0 3px rgba(0,0,0,.5);text-align:center;'
                          f'line-height:20px;font:700 12px sans-serif;">{r.priority}</div>')),
            ).add_to(fmap)

        folium.Marker([flat, flon], tooltip=f"Facility region: {loc_name}",
                      icon=folium.Icon(color="red", icon="industry", prefix="fa")).add_to(fmap)

        clicked = st_folium(fmap, height=500, width="100%", key=f"equity_{state}",
                            returned_objects=["last_object_clicked"])

        # --- Click a county for its profile ---
        pick = (clicked or {}).get("last_object_clicked")
        if pick and pick.get("lat") is not None:
            d = decision.haversine_mi(pick["lat"], pick["lng"],
                                      need.nearby["lat"].to_numpy(),
                                      need.nearby["lon"].to_numpy())
            i = int(d.argmin())
            if d[i] <= 15:
                c = need.nearby.iloc[i]
                tier_label, _ = decision.svi_class(c["svi_overall"])
                with st.container(border=True):
                    st.markdown(f"**{c['county']}, {c['state']}** · {c['distance_mi']:.0f} mi "
                                f"from facility · {tier_label} vulnerability "
                                f"(percentile {c['svi_overall'] * 100:.1f})")
                    p1, p2, p3, p4, p5 = st.columns(5)
                    p1.metric("Residents", f"{c['population']:,.0f}")
                    p2.metric("Adults 65+", f"{c['pct_age65']:.1f}%")
                    p3.metric("Children ≤17", f"{c['pct_age17']:.1f}%")
                    p4.metric("Below 150% poverty", f"{c['pct_pov150']:.1f}%")
                    p5.metric("Disability", f"{c['pct_disabled']:.1f}%")
        else:
            st.caption("Click any county on the map to see its population profile.")

    with list_col:
        st.markdown("**Donation manifest — highest-need counties**")
        st.caption(
            "**Vulnerability percentile** ranks each county against all ~3,100 U.S. "
            "counties on the CDC Social Vulnerability Index. 99.5 means more "
            "vulnerable than 99.5% of counties; 100 is the most vulnerable. 75 and "
            "above counts as high. **Share of product** splits the donation by the "
            "number of residents below 150% of the poverty line.")
        manifest = pd.DataFrame({
            "#": targets["priority"],
            "County": targets["county"] + ", " + targets["state"],
            "Miles": targets["distance_mi"].round(0),
            "Vulnerability percentile": targets["svi_percentile"],
            "Share of product": targets["allocation"] * 100,
            "Residents below 150% poverty": targets["people_in_need"],
            "Adults 65+": targets["seniors_65"],
            "Children ≤17": targets["children_17"],
            "Disability %": targets["pct_disabled"],
            "Uninsured %": targets["pct_uninsured"],
        })
        st.dataframe(
            manifest, hide_index=True, width="stretch", height=420,
            column_config={
                "#": st.column_config.NumberColumn(help="Delivery priority; matches the "
                                                        "numbered stops on the map.", width="small"),
                "Miles": st.column_config.NumberColumn(format="%d",
                                                       help="Straight-line distance from the facility region."),
                "Vulnerability percentile": st.column_config.ProgressColumn(
                    min_value=0, max_value=100, format="%.1f",
                    help="CDC SVI national percentile rank. 100 = most vulnerable county "
                         "in the U.S.; 75+ = high vulnerability."),
                "Share of product": st.column_config.ProgressColumn(
                    min_value=0, max_value=100, format="%.0f%%",
                    help="Portion of the donated cooked product routed to this county."),
                "Residents below 150% poverty": st.column_config.NumberColumn(format="%d"),
                "Adults 65+": st.column_config.NumberColumn(format="%d"),
                "Children ≤17": st.column_config.NumberColumn(format="%d"),
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
