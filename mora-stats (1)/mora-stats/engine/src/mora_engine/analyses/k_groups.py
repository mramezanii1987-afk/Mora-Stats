"""One-way ANOVA and Kruskal-Wallis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .. import diagnostics as dx
from .. import effects, power
from ..models import (AnalysisResult, Design, Estimate, PowerSummary, Variable)
from .common import complete_cases, describe, split_groups


def _prepare(frame: pd.DataFrame, outcome: Variable, group: Variable):
    kept, dropped = complete_cases(frame, [outcome.name, group.name])
    groups = split_groups(kept, outcome.name, group.name)
    if len(groups) < 3:
        raise ValueError(
            f"{group.display} has {len(groups)} groups with data. With two "
            "groups use a t-test or Mann-Whitney, which answer the same "
            "question more directly."
        )
    return kept, dropped, groups, list(groups.keys())


def one_way_anova(frame: pd.DataFrame, outcome: Variable, group: Variable,
                  design: Design, alpha: float = 0.05) -> AnalysisResult:
    kept, dropped, groups, labels = _prepare(frame, outcome, group)
    arrays = [groups[label] for label in labels]
    f_stat, p = stats.f_oneway(*arrays)
    parts = effects.anova_effects(arrays)
    omega = parts["omega_squared"]
    lo, hi = effects.omega_squared_ci(float(f_stat), parts["df1"],
                                      parts["df2"], parts["n_total"])

    means = {label: float(np.mean(groups[label])) for label in labels}
    high = max(means, key=lambda k: means[k])
    low = min(means, key=lambda k: means[k])
    spread = means[high] - means[low]

    threats = []
    threats += dx.missing_note(parts["n_total"], dropped)
    threats += dx.check_cell_sizes(groups)
    threats += dx.check_variance_ratio(groups)
    for label in labels:
        threats += dx.check_bounds(groups[label], outcome, where=f" in {label}")
    threats += dx.check_normality(groups, outcome)

    mde = power.mde_anova_omega(len(labels), parts["n_total"], alpha)
    summary = PowerSummary(
        alpha=alpha,
        mde=mde,
        mde_label="omega squared",
        observed_power=power.power_anova(
            np.sqrt(omega / (1 - omega)) if 0 < omega < 1 else 0.0,
            len(labels), parts["n_total"], alpha),
        exaggeration=power.exaggeration_f(float(f_stat), parts["df1"],
                                          parts["df2"], parts["n_total"],
                                          alpha),
    )

    return AnalysisResult(
        analysis="one_way_anova",
        label="One-way ANOVA",
        design=design,
        statistic_name="F",
        statistic=float(f_stat),
        df=float(parts["df1"]),
        df2=float(parts["df2"]),
        p=float(p),
        alpha=alpha,
        n=parts["n_total"],
        n_missing=dropped,
        effect=Estimate(name="omega squared", value=omega, ci_low=lo,
                        ci_high=hi, method="noncentral F"),
        raw_effect=Estimate(
            name=f"spread between the highest and lowest group mean "
                 f"({outcome.units})",
            value=spread, method="descriptive"),
        groups=[describe(label, groups[label]) for label in labels],
        variables=[outcome.name, group.name],
        outcome=outcome,
        predictor=group,
        power=summary,
        threats=threats,
        extra={"family": "k_group_mean", "labels": labels, "means": means,
               "highest": high, "lowest": low,
               "eta_squared": parts["eta_squared"]},
    )


def kruskal_wallis(frame: pd.DataFrame, outcome: Variable, group: Variable,
                   design: Design, alpha: float = 0.05) -> AnalysisResult:
    kept, dropped, groups, labels = _prepare(frame, outcome, group)
    arrays = [groups[label] for label in labels]
    h_stat, p = stats.kruskal(*arrays)
    n_total = int(sum(len(a) for a in arrays))
    eps = effects.epsilon_squared(float(h_stat), n_total)

    def eps_stat(*resampled):
        h, _ = stats.kruskal(*resampled)
        return effects.epsilon_squared(float(h), sum(len(r) for r in resampled))

    lo, hi = effects.bootstrap_ci(eps_stat, arrays)

    medians = {label: float(np.median(groups[label])) for label in labels}
    high = max(medians, key=lambda k: medians[k])
    low = min(medians, key=lambda k: medians[k])

    threats = []
    threats += dx.missing_note(n_total, dropped)
    threats += dx.check_cell_sizes(groups)
    for label in labels:
        threats += dx.check_bounds(groups[label], outcome, where=f" in {label}")

    mde = power.mde_kruskal(len(labels), n_total, alpha)
    summary = PowerSummary(
        alpha=alpha,
        mde=mde,
        mde_label="epsilon squared",
        approximate=True,
        note=("Approximate. Rank tests have no exact power function, so this "
              "comes from the equivalent ANOVA adjusted for the efficiency "
              "of rank methods."),
    )

    return AnalysisResult(
        analysis="kruskal_wallis",
        label="Kruskal-Wallis test",
        design=design,
        statistic_name="H",
        statistic=float(h_stat),
        df=float(len(labels) - 1),
        p=float(p),
        alpha=alpha,
        n=n_total,
        n_missing=dropped,
        effect=Estimate(name="epsilon squared", value=eps, ci_low=lo,
                        ci_high=hi, method="seeded percentile bootstrap"),
        raw_effect=Estimate(
            name=f"spread between the highest and lowest group median "
                 f"({outcome.units})",
            value=medians[high] - medians[low], method="descriptive"),
        groups=[describe(label, groups[label]) for label in labels],
        variables=[outcome.name, group.name],
        outcome=outcome,
        predictor=group,
        power=summary,
        threats=threats,
        extra={"family": "k_group_mean", "labels": labels,
               "medians": medians, "highest": high, "lowest": low},
    )
