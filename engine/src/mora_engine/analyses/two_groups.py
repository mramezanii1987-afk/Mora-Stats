"""Independent two group comparisons."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .. import diagnostics as dx
from .. import effects, power
from ..models import (AnalysisResult, Design, Estimate, PowerSummary,
                      VarType, Variable)
from .common import complete_cases, describe, split_groups


def _prepare(frame: pd.DataFrame, outcome: Variable, group: Variable):
    kept, dropped = complete_cases(frame, [outcome.name, group.name])
    groups = split_groups(kept, outcome.name, group.name)
    if len(groups) != 2:
        raise ValueError(
            f"{group.display} has {len(groups)} groups with data. This test "
            "compares exactly two, so either filter the rows or use one-way "
            "ANOVA or Kruskal-Wallis."
        )
    labels = list(groups.keys())
    return kept, dropped, groups, labels


def _mean_difference_estimate(a: np.ndarray, b: np.ndarray, welch: bool,
                              unit: str, level: float = 0.95) -> Estimate:
    n1, n2 = len(a), len(b)
    diff = float(np.mean(a) - np.mean(b))
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    if welch:
        se = float(np.sqrt(v1 / n1 + v2 / n2))
        df = (v1 / n1 + v2 / n2) ** 2 / (
            (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    else:
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        se = float(np.sqrt(sp2 * (1 / n1 + 1 / n2)))
        df = n1 + n2 - 2
    crit = stats.t.ppf(1 - (1 - level) / 2, df)
    return Estimate(
        name=f"mean difference ({unit})",
        value=diff,
        ci_low=diff - crit * se,
        ci_high=diff + crit * se,
        method="Welch" if welch else "pooled",
    )


def _common_threats(kept, groups, labels, outcome, dropped, effect_fn,
                    observed_effect):
    threats = []
    threats += dx.missing_note(sum(len(v) for v in groups.values()), dropped)
    threats += dx.check_cell_sizes(groups)
    threats += dx.check_variance_ratio(groups)
    for label in labels:
        threats += dx.check_bounds(groups[label], outcome,
                                   where=f" in {label}")
    threats += dx.check_normality(groups, outcome)
    index = np.arange(len(kept))
    threats += dx.leave_one_out_influence(
        effect_fn, index, observed_effect,
        row_labels=[f"row {i}" for i in kept.index.to_list()])
    return threats


def t_test(frame: pd.DataFrame, outcome: Variable, group: Variable,
           design: Design, welch: bool = True, alpha: float = 0.05
           ) -> AnalysisResult:
    kept, dropped, groups, labels = _prepare(frame, outcome, group)
    a, b = groups[labels[0]], groups[labels[1]]
    n1, n2 = len(a), len(b)

    t_stat, p = stats.ttest_ind(a, b, equal_var=not welch)
    if welch:
        v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
        df = (v1 / n1 + v2 / n2) ** 2 / (
            (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    else:
        df = n1 + n2 - 2

    g = effects.hedges_g(a, b)
    if welch:
        # The noncentral t interval assumes the pooled denominator, so for
        # Welch the engine bootstraps the same estimator instead of pretending.
        g_lo, g_hi = effects.bootstrap_ci(effects.hedges_g, [a, b])
        g_method = "seeded percentile bootstrap"
    else:
        g_lo, g_hi = effects.hedges_g_ci(a, b)
        g_method = "noncentral t"

    outcome_values = kept[outcome.name].to_numpy(dtype=float)
    group_values = kept[group.name].astype(str).to_numpy()

    def loo_effect(keep_idx: np.ndarray) -> float:
        sub_out = outcome_values[keep_idx]
        sub_grp = group_values[keep_idx]
        aa = sub_out[sub_grp == labels[0]]
        bb = sub_out[sub_grp == labels[1]]
        if len(aa) < 2 or len(bb) < 2:
            return float("nan")
        return effects.hedges_g(aa, bb)

    threats = _common_threats(kept, groups, labels, outcome, dropped,
                              loo_effect, g)

    mde = power.mde_two_sample_t(n1, n2, alpha)
    sp = effects.pooled_sd(a, b)
    summary = PowerSummary(
        alpha=alpha,
        mde=mde,
        mde_label="Hedges' g",
        mde_raw=None if mde is None else mde * sp,
        observed_power=power.power_two_sample_t(g, n1, n2, alpha),
        exaggeration=power.exaggeration_t(g, n1, n2, alpha),
    )

    label = "Welch's t-test" if welch else "Student's t-test"
    return AnalysisResult(
        analysis="welch_t" if welch else "student_t",
        label=label,
        design=design,
        statistic_name="t",
        statistic=float(t_stat),
        df=float(df),
        p=float(p),
        alpha=alpha,
        n=n1 + n2,
        n_missing=dropped,
        effect=Estimate(name="Hedges' g", value=g, ci_low=g_lo, ci_high=g_hi,
                        method=g_method),
        raw_effect=_mean_difference_estimate(a, b, welch, outcome.units),
        groups=[describe(labels[0], a), describe(labels[1], b)],
        variables=[outcome.name, group.name],
        outcome=outcome,
        predictor=group,
        power=summary,
        threats=threats,
        extra={"family": "two_group_mean", "labels": labels,
               "pooled_sd": sp},
    )


def mann_whitney(frame: pd.DataFrame, outcome: Variable, group: Variable,
                 design: Design, alpha: float = 0.05) -> AnalysisResult:
    kept, dropped, groups, labels = _prepare(frame, outcome, group)
    a, b = groups[labels[0]], groups[labels[1]]
    n1, n2 = len(a), len(b)

    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    rb = effects.rank_biserial(a, b)
    rb_lo, rb_hi = effects.bootstrap_ci(effects.rank_biserial, [a, b])

    outcome_values = kept[outcome.name].to_numpy(dtype=float)
    group_values = kept[group.name].astype(str).to_numpy()

    def loo_effect(keep_idx: np.ndarray) -> float:
        sub_out = outcome_values[keep_idx]
        sub_grp = group_values[keep_idx]
        aa = sub_out[sub_grp == labels[0]]
        bb = sub_out[sub_grp == labels[1]]
        if len(aa) < 2 or len(bb) < 2:
            return float("nan")
        return effects.rank_biserial(aa, bb)

    threats = []
    threats += dx.missing_note(n1 + n2, dropped)
    threats += dx.check_cell_sizes(groups)
    for lbl in labels:
        threats += dx.check_bounds(groups[lbl], outcome, where=f" in {lbl}")
    threats += dx.leave_one_out_influence(
        loo_effect, np.arange(len(kept)), rb,
        row_labels=[f"row {i}" for i in kept.index.to_list()])

    mde_d = power.mde_mann_whitney(n1, n2, alpha)
    mde_rb = None if mde_d is None else power.d_to_rank_biserial(mde_d)
    summary = PowerSummary(
        alpha=alpha,
        mde=mde_rb,
        mde_label="rank-biserial r",
        observed_power=None,
        approximate=True,
        note=("Approximate. Rank tests have no exact power function, so this "
              "comes from the equivalent t-test adjusted for the efficiency "
              "of rank methods."),
    )

    median_diff = float(np.median(a) - np.median(b))
    return AnalysisResult(
        analysis="mann_whitney",
        label="Mann-Whitney U test",
        design=design,
        statistic_name="U",
        statistic=float(res.statistic),
        df=None,
        p=float(res.pvalue),
        alpha=alpha,
        n=n1 + n2,
        n_missing=dropped,
        effect=Estimate(name="rank-biserial r", value=rb, ci_low=rb_lo,
                        ci_high=rb_hi, method="seeded percentile bootstrap"),
        raw_effect=Estimate(name=f"median difference ({outcome.units})",
                            value=median_diff, method="descriptive"),
        groups=[describe(labels[0], a), describe(labels[1], b)],
        variables=[outcome.name, group.name],
        outcome=outcome,
        predictor=group,
        power=summary,
        threats=threats,
        extra={"family": "two_group_mean", "labels": labels,
               "prob_superiority": (rb + 1) / 2},
    )
