"""Step 4 — Sequential Bayesian risk updating.

Establishes a baseline geographical + seasonal Salmonella-surge risk from
historical CDC NORS data (the prior) and updates it as new FSIS supply-chain
evidence arrives, via Bayes' theorem:

        p(w | D) = p(D | w) p(w) / p(D)

Model
-----
The latent quantity w is the Salmonella-surge risk of a (state, month) cell:
the probability that a drawn signal in that cell is Salmonella-positive.

We use the Beta-Binomial conjugate pair, which makes Bayes' theorem exact and
closed-form (p(D) is handled analytically):

  * Prior      w ~ Beta(alpha0, beta0), with alpha0/beta0 derived from NORS
               Salmonella-vs-not outbreak counts in that (state, month) cell,
               blended with a national seasonal baseline so sparse cells are
               regularized (empirical-Bayes style).
  * Likelihood A batch of FSIS samples contributes Binomial evidence: k
               Salmonella-positive of n tested -> p(D|w) = C(n,k) w^k (1-w)^(n-k).
  * Posterior  w | D ~ Beta(alpha0 + k, beta0 + (n - k)).  Conjugacy.

Sequential updating is just accumulation: each new batch adds its (k, n-k) to
the cell's Beta parameters, so today's posterior is tomorrow's prior.

The posterior mean E[w|D] = (alpha+k)/(alpha+beta+n) is the "surge probability"
surfaced on the Step 7 dashboard. We also expose the full posterior (for
credible intervals) and the probability that risk exceeds a threshold
(P(w > tau)), which is the actionable alert signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from . import geo, ingest_local, targets

# Strength of the prior: how many "pseudo-observations" the NORS prior is worth
# before FSIS evidence. Higher = stronger prior, slower to move. This is the
# concentration of the Beta prior built from NORS rates.
DEFAULT_PRIOR_STRENGTH = 20.0

# Blend weight toward the national seasonal baseline for sparse cells
# (empirical Bayes shrinkage). 0 = pure cell rate, 1 = pure national-month rate.
# We set this adaptively per cell by sample size, but this caps the floor.
MIN_CELL_OUTBREAKS = 5  # below this, lean on the national-month baseline


@dataclass
class CellPrior:
    """Beta prior parameters for one (state, month) cell."""

    state: str
    month: int
    alpha: float
    beta: float
    nors_total: int        # NORS outbreaks observed in the cell
    nors_salmonella: int    # of which Salmonella

    @property
    def prior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


@dataclass
class Posterior:
    """Posterior Beta for a cell after absorbing FSIS evidence."""

    state: str
    month: int
    alpha: float
    beta: float
    prior_alpha: float
    prior_beta: float
    evidence_k: int = 0     # Salmonella-positive FSIS samples absorbed
    evidence_n: int = 0     # total FSIS samples absorbed
    history: list[tuple[int, int]] = field(default_factory=list)  # (k,n) per batch

    @property
    def mean(self) -> float:
        """Posterior mean E[w|D] — the surge probability."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def prior_mean(self) -> float:
        return self.prior_alpha / (self.prior_alpha + self.prior_beta)

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Equal-tailed Beta credible interval for w."""
        lo = (1 - level) / 2
        hi = 1 - lo
        return (float(stats.beta.ppf(lo, self.alpha, self.beta)),
                float(stats.beta.ppf(hi, self.alpha, self.beta)))

    def prob_exceeds(self, tau: float) -> float:
        """P(w > tau | D): probability risk exceeds threshold tau (the alert)."""
        return float(stats.beta.sf(tau, self.alpha, self.beta))

    def pdf(self, grid: np.ndarray) -> np.ndarray:
        """Posterior density over a [0,1] grid (for plotting the distribution)."""
        return stats.beta.pdf(grid, self.alpha, self.beta)


# ---------------------------------------------------------------------------
# Prior construction from NORS
# ---------------------------------------------------------------------------
def build_priors(
    nors: pd.DataFrame | None = None,
    *,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
) -> dict[tuple[str, int], CellPrior]:
    """Build Beta priors per (state_abbr, month) from NORS outbreak history.

    For each cell we estimate a prior risk rate by shrinking the cell's own
    Salmonella rate toward the national rate for that MONTH (which carries the
    seasonal signal) when the cell has few observations. The Beta prior is then
    Beta(prior_strength * rate, prior_strength * (1 - rate)).
    """
    if nors is None:
        nors = ingest_local.load_nors(foodborne_only=True)

    dt = pd.to_datetime(nors["DateFirstIll"], errors="coerce")
    salm = targets._salmonella_mask(nors).astype(int)
    df = pd.DataFrame({
        "state": nors["ExposureState"].map(geo.to_abbr),
        "month": dt.dt.month,
        "salm": salm.to_numpy(),
    }).dropna(subset=["state", "month"])
    df["month"] = df["month"].astype(int)

    # National per-month Salmonella rate (seasonal baseline).
    national_month_rate = df.groupby("month")["salm"].mean().to_dict()
    national_rate = float(df["salm"].mean())

    priors: dict[tuple[str, int], CellPrior] = {}
    grp = df.groupby(["state", "month"])["salm"].agg(["size", "sum"])
    for (state, month), row in grp.iterrows():
        total = int(row["size"])
        salm_n = int(row["sum"])
        cell_rate = salm_n / total if total else national_rate
        season_rate = national_month_rate.get(month, national_rate)

        # Empirical-Bayes shrinkage toward the seasonal baseline: weight the
        # cell rate by its sample size relative to MIN_CELL_OUTBREAKS.
        w_cell = total / (total + MIN_CELL_OUTBREAKS)
        rate = w_cell * cell_rate + (1 - w_cell) * season_rate
        rate = min(max(rate, 1e-3), 1 - 1e-3)  # keep strictly in (0,1)

        alpha = prior_strength * rate
        beta = prior_strength * (1 - rate)
        priors[(state, month)] = CellPrior(
            state=state, month=month, alpha=alpha, beta=beta,
            nors_total=total, nors_salmonella=salm_n,
        )

    # Also store a fallback prior per month for unseen cells.
    for month, season_rate in national_month_rate.items():
        key = ("__NATIONAL__", int(month))
        r = min(max(season_rate, 1e-3), 1 - 1e-3)
        priors[key] = CellPrior(
            state="__NATIONAL__", month=int(month),
            alpha=prior_strength * r, beta=prior_strength * (1 - r),
            nors_total=0, nors_salmonella=0,
        )
    return priors


def get_prior(
    priors: dict[tuple[str, int], CellPrior], state: str, month: int
) -> CellPrior:
    """Look up a cell prior, falling back to the national-month baseline."""
    key = (state, month)
    if key in priors:
        return priors[key]
    nat = priors.get(("__NATIONAL__", month))
    if nat is not None:
        # Return a copy keyed to the requested cell for clarity.
        return CellPrior(state=state, month=month, alpha=nat.alpha,
                         beta=nat.beta, nors_total=0, nors_salmonella=0)
    # Absolute fallback: weak uniform-ish prior.
    return CellPrior(state=state, month=month, alpha=1.0, beta=1.0,
                     nors_total=0, nors_salmonella=0)


# ---------------------------------------------------------------------------
# Sequential Bayesian updating
# ---------------------------------------------------------------------------
def init_posterior(prior: CellPrior) -> Posterior:
    """Start a posterior equal to the prior (no evidence absorbed yet)."""
    return Posterior(
        state=prior.state, month=prior.month,
        alpha=prior.alpha, beta=prior.beta,
        prior_alpha=prior.alpha, prior_beta=prior.beta,
    )


def update_posterior(post: Posterior, k: int, n: int) -> Posterior:
    """Absorb one FSIS batch: k Salmonella-positive of n tested (Binomial).

    Conjugate update: alpha += k, beta += (n - k). Returns a NEW Posterior so
    the sequence of updates is inspectable; mutating in place is avoided.
    """
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"Invalid batch counts: k={k}, n={n} (need 0<=k<=n).")
    new = Posterior(
        state=post.state, month=post.month,
        alpha=post.alpha + k, beta=post.beta + (n - k),
        prior_alpha=post.prior_alpha, prior_beta=post.prior_beta,
        evidence_k=post.evidence_k + k, evidence_n=post.evidence_n + n,
        history=post.history + [(k, n)],
    )
    return new


def sequential_update(
    prior: CellPrior, batches: list[tuple[int, int]]
) -> Posterior:
    """Apply a sequence of (k, n) FSIS batches to a cell prior, in order.

    Demonstrates the 'today's posterior is tomorrow's prior' property: the
    result is identical whether batches are applied one-by-one or pooled,
    which is a property of the Beta-Binomial conjugate model.
    """
    post = init_posterior(prior)
    for k, n in batches:
        post = update_posterior(post, k, n)
    return post


# ---------------------------------------------------------------------------
# FSIS evidence extraction
# ---------------------------------------------------------------------------
def fsis_batches_by_cell(
    fsis_joined: pd.DataFrame | None = None,
    *,
    poultry_only: bool = True,
) -> dict[tuple[str, int], tuple[int, int]]:
    """Aggregate FSIS samples into (state, month) -> (k_positive, n_total).

    k = count of Salmonella-positive results, n = total analyzed samples in the
    cell. Uses salmonella_sp_analysis (Positive/Negative); 'NULL'/missing and
    not-analyzed rows are excluded from n.
    """
    if fsis_joined is None:
        tables = ingest_local.load_fsis()
        fsis_joined = tables.poultry if poultry_only else tables.joined

    df = fsis_joined
    if df.empty:
        return {}

    state = df["establishment_state"].map(geo.to_abbr)
    month = pd.to_datetime(df["collection_date"], errors="coerce").dt.month
    result = df["salmonella_sp_analysis"].astype("string").str.strip().str.lower()

    analyzed = result.isin(["positive", "negative"])
    positive = result.eq("positive")

    work = pd.DataFrame({
        "state": state, "month": month,
        "analyzed": analyzed.astype(int), "positive": positive.astype(int),
    }).dropna(subset=["state", "month"])
    work = work[work["analyzed"] == 1]
    work["month"] = work["month"].astype(int)

    out: dict[tuple[str, int], tuple[int, int]] = {}
    for (st, mo), g in work.groupby(["state", "month"]):
        n = int(len(g))
        k = int(g["positive"].sum())
        out[(st, mo)] = (k, n)
    return out


def posteriors_for_fsis(
    *,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    poultry_only: bool = True,
) -> dict[tuple[str, int], Posterior]:
    """End-to-end: NORS priors updated by FSIS evidence, per observed cell.

    Returns a posterior for every (state, month) cell that appears in the FSIS
    evidence, each built from that cell's NORS prior and updated with the FSIS
    batch counts.
    """
    priors = build_priors(prior_strength=prior_strength)
    batches = fsis_batches_by_cell(poultry_only=poultry_only)
    out: dict[tuple[str, int], Posterior] = {}
    for (state, month), (k, n) in batches.items():
        prior = get_prior(priors, state, month)
        out[(state, month)] = update_posterior(init_posterior(prior), k, n)
    return out


__all__ = [
    "CellPrior",
    "Posterior",
    "DEFAULT_PRIOR_STRENGTH",
    "build_priors",
    "get_prior",
    "init_posterior",
    "update_posterior",
    "sequential_update",
    "fsis_batches_by_cell",
    "posteriors_for_fsis",
]
