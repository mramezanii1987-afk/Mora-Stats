"""Effect size estimators and their confidence intervals.

Conventions used throughout, chosen for small-sample honesty:

* Hedges' g rather than Cohen's d, with a noncentral t interval.
* Omega squared rather than eta squared for one-way ANOVA, because eta
  squared is biased upward at the sample sizes researchers actually have.
* Rank-biserial correlation for Mann-Whitney, epsilon squared for
  Kruskal-Wallis, both with a seeded percentile bootstrap.
* Fisher z intervals for Pearson and Spearman, with the Bonett-Wright
  variance correction for Spearman.
* Cramer's V from a noncentral chi-square interval on the noncentrality,
  which avoids a bootstrap for contingency tables.

Every bootstrap is seeded with BOOTSTRAP_SEED so the engine returns the same
interval for the same data on every run, on every machine.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, stats

BOOTSTRAP_SEED = 20260914
BOOTSTRAP_DRAWS = 4000


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def hedges_correction(df: float) -> float:
    """Small-sample correction factor J. Exactly 1 in the limit."""
    if df <= 1:
        return 1.0
    return 1.0 - 3.0 / (4.0 * df - 1.0)


def ncp_interval(t_obs: float, df: float, level: float = 0.95
                 ) -> Tuple[Optional[float], Optional[float]]:
    """Confidence interval for the noncentrality of a t statistic.

    Solves for the noncentralities that place the observed t at the upper
    and lower tails of their noncentral t distributions.
    """
    alpha = 1.0 - level
    lo_target = 1.0 - alpha / 2.0
    hi_target = alpha / 2.0

    def stable_cdf(ncp: float) -> float:
        value = float(stats.nct.cdf(t_obs, df, ncp))
        if not np.isfinite(value):
            # Standard normal approximation to the noncentral t, used only
            # where the exact computation returns NaN.
            denom = np.sqrt(1.0 + t_obs ** 2 / (2.0 * df))
            value = float(stats.norm.cdf(
                (t_obs * (1.0 - 1.0 / (4.0 * df)) - ncp) / denom))
        return value

    def solve(target: float) -> Optional[float]:
        def f(ncp: float) -> float:
            return stable_cdf(ncp) - target

        span = max(10.0, abs(t_obs) * 4.0 + 10.0)
        lo, hi = t_obs - span, t_obs + span
        try:
            if f(lo) * f(hi) > 0:
                return None
            return float(optimize.brentq(f, lo, hi, xtol=1e-8, maxiter=200))
        except (ValueError, RuntimeError):
            return None

    return solve(lo_target), solve(hi_target)


def ncp_interval_f(f_obs: float, df1: float, df2: float, level: float = 0.95
                   ) -> Tuple[Optional[float], Optional[float]]:
    """Confidence interval for the noncentrality of an F statistic."""
    alpha = 1.0 - level

    def solve(target: float) -> float:
        def g(lam: float) -> float:
            return stats.ncf.cdf(f_obs, df1, df2, lam) - target

        hi = 10.0
        for _ in range(60):
            if g(hi) < 0:
                break
            hi *= 2.0
        # g already subtracts the target, so the test for "no noncentrality
        # is consistent with this statistic" compares against zero.
        if g(0.0) < 0:
            return 0.0
        try:
            return float(optimize.brentq(g, 0.0, hi, xtol=1e-8, maxiter=200))
        except (ValueError, RuntimeError):
            return 0.0

    return solve(1.0 - alpha / 2.0), solve(alpha / 2.0)


def ncp_interval_chi2(x_obs: float, df: float, level: float = 0.95
                      ) -> Tuple[float, float]:
    alpha = 1.0 - level

    def solve(target: float) -> float:
        def g(lam: float) -> float:
            return stats.ncx2.cdf(x_obs, df, lam) - target

        if g(1e-9) < 0:
            return 0.0
        hi = 10.0
        for _ in range(60):
            if g(hi) < 0:
                break
            hi *= 2.0
        try:
            return float(optimize.brentq(g, 1e-9, hi, xtol=1e-8, maxiter=200))
        except (ValueError, RuntimeError):
            return 0.0

    return solve(1.0 - alpha / 2.0), solve(alpha / 2.0)


def bootstrap_ci(statistic: Callable[..., float],
                 samples: Sequence[np.ndarray],
                 level: float = 0.95,
                 draws: int = BOOTSTRAP_DRAWS,
                 seed: int = BOOTSTRAP_SEED
                 ) -> Tuple[Optional[float], Optional[float]]:
    """Seeded percentile bootstrap over one or more independent samples."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        resampled = [rng.choice(s, size=len(s), replace=True) for s in samples]
        try:
            values.append(statistic(*resampled))
        except Exception:
            continue
    if len(values) < draws * 0.5:
        return None, None
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 10:
        return None, None
    alpha = 1.0 - level
    return (float(np.quantile(arr, alpha / 2.0)),
            float(np.quantile(arr, 1.0 - alpha / 2.0)))


# --------------------------------------------------------------------------
# two independent groups
# --------------------------------------------------------------------------

def pooled_sd(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    return float(np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)))


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    sp = pooled_sd(a, b)
    if sp == 0:
        return 0.0
    d = (float(np.mean(a)) - float(np.mean(b))) / sp
    return d * hedges_correction(len(a) + len(b) - 2)


def hedges_g_ci(a: np.ndarray, b: np.ndarray, level: float = 0.95
                ) -> Tuple[Optional[float], Optional[float]]:
    n1, n2 = len(a), len(b)
    df = n1 + n2 - 2
    sp = pooled_sd(a, b)
    if sp == 0:
        return None, None
    t_obs = (float(np.mean(a)) - float(np.mean(b))) / (sp * np.sqrt(1 / n1 + 1 / n2))
    lo, hi = ncp_interval(t_obs, df, level)
    scale = np.sqrt(1 / n1 + 1 / n2) * hedges_correction(df)
    return (None if lo is None else lo * scale,
            None if hi is None else hi * scale)


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Rank-biserial correlation, positive when a tends to exceed b.

    Equals the probability that a randomly drawn value from a exceeds one
    from b, minus the reverse probability.
    """
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 0.0
    u1 = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2.0 * u1 / (n1 * n2) - 1.0)


# --------------------------------------------------------------------------
# k groups
# --------------------------------------------------------------------------

def anova_effects(groups: Sequence[np.ndarray]) -> dict:
    """Sums of squares, eta squared and omega squared for one-way ANOVA."""
    all_values = np.concatenate(groups)
    grand = float(np.mean(all_values))
    n_total = len(all_values)
    k = len(groups)
    ss_between = float(sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups))
    ss_within = float(sum(((g - np.mean(g)) ** 2).sum() for g in groups))
    ss_total = ss_between + ss_within
    df1, df2 = k - 1, n_total - k
    ms_within = ss_within / df2 if df2 > 0 else np.nan
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
    denom = ss_total + ms_within
    omega_sq = (ss_between - df1 * ms_within) / denom if denom > 0 else 0.0
    return {
        "ss_between": ss_between,
        "ss_within": ss_within,
        "ss_total": ss_total,
        "df1": df1,
        "df2": df2,
        "ms_within": ms_within,
        "eta_squared": eta_sq,
        "omega_squared": float(max(omega_sq, 0.0)),
        "omega_squared_raw": float(omega_sq),
        "n_total": n_total,
    }


def omega_squared_ci(f_obs: float, df1: int, df2: int, n_total: int,
                     level: float = 0.95) -> Tuple[float, float]:
    """Interval for omega squared, carried over from the noncentrality.

    With lambda the noncentrality, f squared is lambda / N and omega squared
    is f squared over one plus f squared.
    """
    lo, hi = ncp_interval_f(f_obs, df1, df2, level)

    def to_omega(lam: Optional[float]) -> float:
        if lam is None or lam <= 0:
            return 0.0
        f2 = lam / n_total
        return float(f2 / (1.0 + f2))

    return to_omega(lo), to_omega(hi)


def epsilon_squared(h: float, n: int) -> float:
    """Epsilon squared for Kruskal-Wallis."""
    if n < 2:
        return 0.0
    return float(h * (n + 1.0) / (n * n - 1.0))


# --------------------------------------------------------------------------
# association
# --------------------------------------------------------------------------

def fisher_ci(r: float, n: int, level: float = 0.95,
              spearman: bool = False) -> Tuple[Optional[float], Optional[float]]:
    if n < 4 or abs(r) >= 1.0:
        return None, None
    z = np.arctanh(r)
    # Bonett and Wright's variance correction for Spearman's rho.
    se = np.sqrt(1.06 / (n - 3)) if spearman else np.sqrt(1.0 / (n - 3))
    crit = stats.norm.ppf(1.0 - (1.0 - level) / 2.0)
    return float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se))


def cramers_v(chi2: float, n: int, rows: int, cols: int) -> float:
    k = min(rows - 1, cols - 1)
    if n <= 0 or k <= 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * k)))


def cramers_v_ci(chi2: float, df: int, n: int, rows: int, cols: int,
                 level: float = 0.95) -> Tuple[float, float]:
    k = min(rows - 1, cols - 1)
    lo, hi = ncp_interval_chi2(chi2, df, level)

    def to_v(lam: float) -> float:
        if lam <= 0 or k <= 0:
            return 0.0
        return float(np.sqrt((lam / n) / k))

    return to_v(lo), to_v(hi)


def interpret_magnitude(value: float, kind: str) -> str:
    """A coarse verbal label, used only as a secondary clause.

    The engine leads with the number in the units of the measure. This label
    exists because readers expect one, and the benchmarks are stated as
    conventions rather than as facts about the world.
    """
    v = abs(value)
    if kind in ("g", "d"):
        cuts = [(0.2, "very small"), (0.5, "small"), (0.8, "medium")]
    elif kind in ("r", "rho", "rank_biserial"):
        cuts = [(0.1, "very small"), (0.3, "small"), (0.5, "medium")]
    elif kind in ("omega_squared", "eta_squared", "epsilon_squared"):
        cuts = [(0.01, "very small"), (0.06, "small"), (0.14, "medium")]
    elif kind == "cramers_v":
        cuts = [(0.1, "very small"), (0.3, "small"), (0.5, "medium")]
    else:
        return "unclassified"
    for threshold, label in cuts:
        if v < threshold:
            return label
    return "large"
