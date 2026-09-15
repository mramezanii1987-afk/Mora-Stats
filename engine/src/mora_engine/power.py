"""Power, minimum detectable effect, and the exaggeration ratio.

Two deliberate choices are worth recording here, because both depart from
what SPSS output trains people to expect.

First, a non-significant result never gets a shrug. The engine reports the
smallest effect the design could have detected with 80% power at the
observed n, which is a property of the design rather than of the data, and
it says in plain words that this is not evidence of no effect.

Second, the inflation flag for an underpowered significant result is not
based on observed power. Power computed from the observed effect at the
observed n is a one-to-one function of p, so a warning driven by it merely
restates the p value. The engine instead reports a Type M ratio: taking the
observed effect as if it were the true one, it computes the average
magnitude of the estimates that would clear significance in a design of this
size, and divides by the true value. When the design is well powered the
ratio sits close to one. When it is not, the ratio states how much the
published estimate would overstate the truth.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from scipy import optimize, stats

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80


# --------------------------------------------------------------------------
# power functions
# --------------------------------------------------------------------------

def _nct_power(crit: float, df: float, ncp: float) -> float:
    """Two-sided power from a noncentral t, with a stable fallback.

    SciPy's noncentral t returns NaN for some large noncentralities, and a
    NaN in the middle of a root search is what silently turns a minimum
    detectable effect into a missing sentence. The normal approximation is
    accurate to several decimals in exactly the region where the exact
    computation breaks down, so it takes over there.
    """
    value = float(stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp))
    if not np.isfinite(value):
        value = float(stats.norm.cdf(abs(ncp) - crit)
                      + stats.norm.cdf(-abs(ncp) - crit))
    return float(min(max(value, 0.0), 1.0))


def _ncf_power(crit: float, df1: float, df2: float, lam: float) -> float:
    value = float(stats.ncf.sf(crit, df1, df2, lam))
    if not np.isfinite(value):
        value = 1.0 if lam > df1 * crit else 0.0
    return float(min(max(value, 0.0), 1.0))


def _ncx2_power(crit: float, df: float, lam: float) -> float:
    value = float(stats.ncx2.sf(crit, df, lam))
    if not np.isfinite(value):
        value = 1.0 if lam + df > crit else 0.0
    return float(min(max(value, 0.0), 1.0))


def power_two_sample_t(d: float, n1: int, n2: int,
                       alpha: float = DEFAULT_ALPHA) -> float:
    df = n1 + n2 - 2
    if df <= 0:
        return float("nan")
    ncp = d / np.sqrt(1.0 / n1 + 1.0 / n2)
    crit = stats.t.ppf(1.0 - alpha / 2.0, df)
    return _nct_power(crit, df, ncp)


def power_anova(f: float, k: int, n_total: int,
                alpha: float = DEFAULT_ALPHA) -> float:
    df1, df2 = k - 1, n_total - k
    if df1 <= 0 or df2 <= 0:
        return float("nan")
    lam = f * f * n_total
    crit = stats.f.ppf(1.0 - alpha, df1, df2)
    return _ncf_power(crit, df1, df2, lam)


def power_correlation(r: float, n: int, alpha: float = DEFAULT_ALPHA) -> float:
    if n < 4 or abs(r) >= 1:
        return float("nan")
    df = n - 2
    ncp = r * np.sqrt(df) / np.sqrt(1 - r * r)
    crit = stats.t.ppf(1.0 - alpha / 2.0, df)
    return _nct_power(crit, df, ncp)


def power_chi2(w: float, n: int, df: int, alpha: float = DEFAULT_ALPHA) -> float:
    if df <= 0:
        return float("nan")
    lam = w * w * n
    crit = stats.chi2.ppf(1.0 - alpha, df)
    return _ncx2_power(crit, df, lam)


# --------------------------------------------------------------------------
# minimum detectable effect
# --------------------------------------------------------------------------

def _solve_mde(raw_power_fn: Callable[[float], float], target: float,
               hi: float = 10.0) -> Optional[float]:
    def power_fn(x: float) -> float:
        value = raw_power_fn(x)
        return 0.0 if not np.isfinite(value) else value

    lo = 1e-6
    try:
        if power_fn(lo) >= target:
            return lo
        top = hi
        for _ in range(40):
            if power_fn(top) >= target:
                break
            top *= 1.5
        else:
            return None
        return float(optimize.brentq(lambda x: power_fn(x) - target, lo, top,
                                     xtol=1e-6, maxiter=200))
    except (ValueError, RuntimeError):
        return None


def mde_two_sample_t(n1: int, n2: int, alpha: float = DEFAULT_ALPHA,
                     target: float = DEFAULT_POWER) -> Optional[float]:
    return _solve_mde(lambda d: power_two_sample_t(d, n1, n2, alpha), target)


def mde_anova_omega(k: int, n_total: int, alpha: float = DEFAULT_ALPHA,
                    target: float = DEFAULT_POWER) -> Optional[float]:
    f = _solve_mde(lambda x: power_anova(x, k, n_total, alpha), target)
    if f is None:
        return None
    f2 = f * f
    return float(f2 / (1.0 + f2))


def mde_correlation(n: int, alpha: float = DEFAULT_ALPHA,
                    target: float = DEFAULT_POWER) -> Optional[float]:
    return _solve_mde(lambda r: power_correlation(min(r, 0.999), n, alpha),
                      target, hi=0.999)


def mde_cramers_v(n: int, df: int, rows: int, cols: int,
                  alpha: float = DEFAULT_ALPHA,
                  target: float = DEFAULT_POWER) -> Optional[float]:
    w = _solve_mde(lambda x: power_chi2(x, n, df, alpha), target)
    if w is None:
        return None
    k = min(rows - 1, cols - 1)
    if k <= 0:
        return None
    return float(w / np.sqrt(k))


# --------------------------------------------------------------------------
# exaggeration (Type M)
# --------------------------------------------------------------------------

def _conditional_mean_above(sf, crit: float, upper: float,
                            points: int = 3000) -> Optional[float]:
    """E[X | X > crit], computed from the survival function.

    Integrating the survival function rather than the density avoids the
    singularities that make SciPy's noncentral densities return NaN, and the
    identity is exact for a non-negative variable:
    E[X 1{X>c}] = c P(X>c) + the integral of P(X>t) from c upward.
    """
    tail = sf(crit)
    if tail is None or not np.isfinite(tail) or tail <= 0:
        return None
    grid = np.linspace(crit, upper, points)
    values = np.array([sf(float(t)) for t in grid])
    if not np.isfinite(values).all():
        values = np.nan_to_num(values, nan=0.0)
    area = float(np.trapezoid(values, grid))
    return float((crit * tail + area) / tail)


def _sf_abs_t(df: float, ncp: float):
    def sf(t: float) -> float:
        value = float(stats.nct.sf(t, df, ncp) + stats.nct.cdf(-t, df, ncp))
        if not np.isfinite(value):
            value = float(stats.norm.sf(t - ncp) + stats.norm.cdf(-t - ncp))
        return min(max(value, 0.0), 1.0)

    return sf


def exaggeration_t(d_obs: float, n1: int, n2: int,
                   alpha: float = DEFAULT_ALPHA) -> Optional[float]:
    """Average overstatement among significant estimates at this design.

    Takes the observed effect as if it were the true one, then asks what the
    estimates that clear significance in a design of this size average. A
    well powered design returns a ratio near one. An underpowered one
    returns the factor by which a published estimate would overstate the
    truth, which is the number worth printing.
    """
    if d_obs == 0:
        return None
    df = n1 + n2 - 2
    if df <= 0:
        return None
    ncp = abs(d_obs) / np.sqrt(1.0 / n1 + 1.0 / n2)
    crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    mean = _conditional_mean_above(_sf_abs_t(df, ncp), crit,
                                   crit + ncp + 40.0)
    if mean is None or ncp <= 0:
        return None
    return float(mean / ncp)


def exaggeration_f(f_obs: float, df1: int, df2: int, n_total: int,
                   alpha: float = DEFAULT_ALPHA) -> Optional[float]:
    lam = max(f_obs * df1 - df1, 0.0)
    if lam <= 0 or df1 <= 0 or df2 <= 0:
        return None
    crit = float(stats.f.ppf(1.0 - alpha, df1, df2))

    def sf(x: float) -> float:
        value = float(stats.ncf.sf(x, df1, df2, lam))
        return 0.0 if not np.isfinite(value) else min(max(value, 0.0), 1.0)

    mean_f = _conditional_mean_above(sf, crit, crit + (lam + 40.0) / df1 * 4)
    if mean_f is None:
        return None
    mean_lambda = mean_f * df1 - df1
    if mean_lambda <= 0:
        return None
    return float(mean_lambda / lam)


def exaggeration_chi2(chi2_obs: float, df: int,
                      alpha: float = DEFAULT_ALPHA) -> Optional[float]:
    lam = max(chi2_obs - df, 0.0)
    if lam <= 0 or df <= 0:
        return None
    crit = float(stats.chi2.ppf(1.0 - alpha, df))

    def sf(x: float) -> float:
        value = float(stats.ncx2.sf(x, df, lam))
        return 0.0 if not np.isfinite(value) else min(max(value, 0.0), 1.0)

    mean_chi = _conditional_mean_above(sf, crit, crit + lam + 60.0 + 8 * df)
    if mean_chi is None:
        return None
    mean_lambda = mean_chi - df
    if mean_lambda <= 0:
        return None
    return float(mean_lambda / lam)


# --------------------------------------------------------------------------
# rank based tests
# --------------------------------------------------------------------------
# There is no exact power function for Mann-Whitney or Kruskal-Wallis. The
# engine reports the parametric analogue's minimum detectable effect,
# adjusted by the asymptotic relative efficiency of the rank test under
# normality, and always labels the number as approximate.

ARE_WILCOXON = 3.0 / np.pi   # about 0.955
ARE_KRUSKAL = 3.0 / np.pi


def mde_mann_whitney(n1: int, n2: int, alpha: float = DEFAULT_ALPHA,
                     target: float = DEFAULT_POWER) -> Optional[float]:
    d = mde_two_sample_t(n1, n2, alpha, target)
    if d is None:
        return None
    return float(d / np.sqrt(ARE_WILCOXON))


def mde_kruskal(k: int, n_total: int, alpha: float = DEFAULT_ALPHA,
                target: float = DEFAULT_POWER) -> Optional[float]:
    omega = mde_anova_omega(k, n_total, alpha, target)
    if omega is None:
        return None
    return float(min(omega / ARE_KRUSKAL, 0.999))


def d_to_rank_biserial(d: float) -> float:
    """Map a standardised mean difference to a rank-biserial value.

    Under normal shift, the probability of superiority is the normal CDF of
    d over root two, and rank-biserial is twice that probability minus one.
    """
    ps = float(stats.norm.cdf(d / np.sqrt(2.0)))
    return 2.0 * ps - 1.0
