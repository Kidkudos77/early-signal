"""Static site generator for GitHub Pages.

Pre-computes the Early Signal results and writes a self-contained static site to
docs/ (GitHub Pages serves from /docs on the main branch). The site is a
SNAPSHOT: it embeds the pre-built Surge Alert numbers, the Trade-Off Matrix, and
the interactive Folium/Plotly maps as static HTML/JS — no Python server needed.
A prominent banner links to the live, fully-interactive Streamlit app.

Run:
    python build_site.py --live-app-url https://your-app.streamlit.app

Outputs:
    docs/index.html                 landing page (all three sections)
    docs/equity_map.html            standalone SVI + Donate-vectors Folium map
    docs/beam_validation.html       BEAM ground-truth validation (if available)
    docs/.nojekyll                  tell Pages not to run Jekyll
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd

from src import bayes, ingest_local, routing

PROJECT_ROOT = Path(__file__).resolve().parent
DOCS = PROJECT_ROOT / "docs"
MONTHS = ["", "January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def assemble():
    priors = bayes.build_priors()
    national = {m: priors[("__NATIONAL__", m)].prior_mean for m in range(1, 13)}
    posts = bayes.posteriors_for_fsis(poultry_only=True)

    cells = []
    for (st_, mo), p in sorted(posts.items()):
        cells.append({
            "state": st_, "month": mo, "month_name": MONTHS[mo],
            "prior": p.prior_mean, "posterior": p.mean,
            "p20": p.prob_exceeds(0.20),
            "k": p.evidence_k, "n": p.evidence_n,
        })

    svi = ingest_local.load_svi(2022)
    risk_by_state = {c["state"]: c["posterior"] for c in cells} or {
        "CA": national[7], "TX": national[7]}
    vectors = routing.donate_routing_vectors(risk_by_state, svi)
    return national, cells, svi, vectors


# ---------------------------------------------------------------------------
# Folium equity map (standalone HTML)
# ---------------------------------------------------------------------------
def build_equity_map(svi, vectors) -> str:
    import folium

    m = folium.Map(location=[39.5, -98.35], zoom_start=4, tiles="OpenStreetMap")
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
    out = DOCS / "equity_map.html"
    m.save(str(out))
    return out.name


# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------
def build_index(national, cells, equity_map_name, live_app_url) -> None:
    # Trade-off matrix rows.
    tdf = routing.trade_off_dataframe()
    rows = "\n".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>"
        for r in tdf.itertuples(index=False)
    )
    header_cells = "".join(f"<th>{html.escape(c)}</th>" for c in tdf.columns)

    # Surge cells.
    if cells:
        surge_rows = "\n".join(
            f"<tr><td>{c['state']}</td><td>{c['month_name']}</td>"
            f"<td>{c['prior']:.1%}</td><td><b>{c['posterior']:.1%}</b></td>"
            f"<td>{c['p20']:.1%}</td><td>{c['k']}/{c['n']}</td></tr>"
            for c in cells
        )
    else:
        surge_rows = "<tr><td colspan=6>No FSIS evidence cells.</td></tr>"

    peak_m = max(national, key=national.get)
    peak_v = national[peak_m]

    # Seasonal bars (simple inline CSS bar chart).
    max_r = max(national.values())
    season_bars = "\n".join(
        f'<div class="bar"><span class="lbl">{MONTHS[m][:3]}</span>'
        f'<span class="fill" style="width:{national[m]/max_r*100:.0f}%"></span>'
        f'<span class="val">{national[m]:.0%}</span></div>'
        for m in range(1, 13)
    )

    live_banner = (
        f'<a class="cta" href="{html.escape(live_app_url)}" target="_blank" '
        f'rel="noopener">▶ Open the live interactive dashboard</a>'
        if live_app_url else
        '<span class="cta disabled">Live app URL not set — '
        'deploy on Streamlit Community Cloud and rebuild</span>'
    )

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Early Signal — RTE Poultry Salmonella Surge Decision Support</title>
<style>
  :root {{ --accent:#d84315; --purple:#6a1b9a; --bg:#faf7fb; --ink:#1a1a1a; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:system-ui,Segoe UI,Roboto,sans-serif;
          color:var(--ink); background:var(--bg); line-height:1.5; }}
  header {{ background:linear-gradient(135deg,#d84315,#6a1b9a); color:#fff;
            padding:2.5rem 1.5rem; }}
  header h1 {{ margin:0 0 .3rem; font-size:1.9rem; }}
  header p {{ margin:0; opacity:.92; max-width:60ch; }}
  main {{ max-width:1000px; margin:0 auto; padding:1.5rem; }}
  section {{ background:#fff; border-radius:12px; padding:1.4rem 1.6rem;
             margin:1.2rem 0; box-shadow:0 1px 4px rgba(0,0,0,.08); }}
  h2 {{ margin-top:0; color:var(--accent); font-size:1.3rem; }}
  .cta {{ display:inline-block; background:#fff; color:var(--accent);
          font-weight:700; padding:.7rem 1.2rem; border-radius:8px;
          text-decoration:none; margin-top:1rem; }}
  .cta.disabled {{ background:rgba(255,255,255,.2); color:#fff; }}
  table {{ border-collapse:collapse; width:100%; margin-top:.6rem; }}
  th,td {{ text-align:left; padding:.55rem .7rem; border-bottom:1px solid #eee; }}
  th {{ background:#f5f0f6; }}
  .metric {{ display:inline-block; margin:.3rem 1.4rem .3rem 0; }}
  .metric .big {{ font-size:2rem; font-weight:800; color:var(--accent); }}
  .metric .cap {{ font-size:.85rem; color:#666; }}
  .bar {{ display:flex; align-items:center; gap:.5rem; margin:.15rem 0; }}
  .bar .lbl {{ width:2.4rem; font-size:.8rem; color:#555; }}
  .bar .fill {{ height:14px; background:var(--accent); border-radius:3px; }}
  .bar .val {{ font-size:.78rem; color:#555; }}
  iframe {{ width:100%; height:520px; border:1px solid #e3dbe6; border-radius:8px; }}
  .note {{ font-size:.85rem; color:#666; }}
  footer {{ text-align:center; color:#888; font-size:.8rem; padding:2rem; }}
  code {{ background:#f0ebf2; padding:.1rem .35rem; border-radius:4px; }}
</style>
</head>
<body>
<header>
  <h1>🐔 Early Signal</h1>
  <p>Proactive, time-aware decision support that predicts seasonal
  <i>Salmonella</i> surges in FSIS-regulated Ready-to-Eat poultry and
  recommends operational interventions. NAFSI SENS-D, Track 3.</p>
  {live_banner}
  <p class="note" style="margin-top:1rem;color:#ffe;">This page is a static
  snapshot of pre-computed results. The button above opens the full
  interactive app (live sidebar controls, adjustable alert threshold).</p>
</header>
<main>

  <section>
    <h2>1 · Surge Alert Header</h2>
    <p>Bayesian posterior probability of a Salmonella spike, per (state ×
    month) cell. Historical CDC NORS outbreaks form the prior; incoming FSIS
    supply-chain samples update it via Bayes' theorem.</p>
    <div class="metric"><div class="big">{peak_v:.0%}</div>
      <div class="cap">National peak-season risk ({MONTHS[peak_m]})</div></div>
    <h3 style="font-size:1rem;">National seasonal baseline risk</h3>
    {season_bars}
    <h3 style="font-size:1rem;margin-top:1rem;">FSIS poultry evidence cells
    (posterior)</h3>
    <table>
      <tr><th>State</th><th>Month</th><th>Prior</th><th>Posterior</th>
          <th>P(risk&gt;0.20)</th><th>FSIS k/n</th></tr>
      {surge_rows}
    </table>
  </section>

  <section>
    <h2>2 · Trade-Off Matrix</h2>
    <p>Operational routing scenarios and their safety / protein / cost
    trade-offs.</p>
    <table><tr>{header_cells}</tr>{rows}</table>
  </section>

  <section>
    <h2>3 · Equity Impact Map</h2>
    <p>CDC/ATSDR Social Vulnerability Index (SVI) by county, overlaid with
    <b>Donate Cooked Product</b> routing vectors from highest-risk states to
    highest-vulnerability counties.</p>
    <iframe src="{equity_map_name}" title="Equity Impact Map"></iframe>
  </section>

  <section>
    <h2>Run it yourself</h2>
    <p>Full interactive dashboard (live controls, API refresh):</p>
    <p><code>streamlit run app/dashboard.py</code> locally, or use the live
    app button at the top of this page.</p>
    <p class="note">Pipeline: NORS/FSIS ingestion → leakage-safe preprocessing
    → out-of-fold predictions + SHAP → sequential Bayesian risk → danger-weighted
    optimization → BEAM/NARMS validation → this dashboard.</p>
  </section>

</main>
<footer>Early Signal · static snapshot generated by <code>build_site.py</code></footer>
</body>
</html>
"""
    (DOCS / "index.html").write_text(doc, encoding="utf-8")


def copy_existing_artifacts() -> list[str]:
    """Copy pre-rendered artifact maps into docs/ if they exist."""
    copied = []
    src_map = PROJECT_ROOT / "artifacts" / "beam_narms" / "hotspot_validation_map.html"
    if src_map.exists():
        (DOCS / "beam_validation.html").write_text(
            src_map.read_text(encoding="utf-8"), encoding="utf-8")
        copied.append("beam_validation.html")
    return copied


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the static Pages site.")
    parser.add_argument("--live-app-url", default="",
                        help="URL of the deployed Streamlit app (optional).")
    args = parser.parse_args(argv)

    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")

    national, cells, svi, vectors = assemble()
    equity_name = build_equity_map(svi, vectors)
    build_index(national, cells, equity_name, args.live_app_url)
    extra = copy_existing_artifacts()

    print("Static site written to docs/")
    print("  index.html, equity_map.html, .nojekyll" +
          ("".join(f", {e}" for e in extra)))
    print(f"  live app url: {args.live_app_url or '(not set)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
