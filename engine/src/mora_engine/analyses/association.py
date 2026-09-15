"""Correlation and chi-square test of independence."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .. import diagnostics as dx
from .. import effects, power
from ..models import (AnalysisResult, Design, Estimate, GroupDescriptives,
                      PowerSummary, Variable)
from .common import complete_cases


def correlation(frame: pd.DataFrame, x: Variable, y: Variable, design: Design,
                method: str = "pearson", alpha: float = 0.05) -> AnalysisResult:
    kept, dropped = complete_cases(frame, [x.name, y.name])
    xv = kept[x.name].to_numpy(dtype=float)
    yv = kept[y.name].to_numpy(dtype=float)
    n = len(xv)
    if n < 4:
        raise ValueError("A correlation needs at least four complete cases.")

    if method == "spearman":
        r, p = stats.spearmanr(xv, yv)
        name, label = "Spearman's rho", "Spearman rank correlation"
        lo, hi = effects.fisher_ci(float(r), n, spearman=True)
        stat_name, stat = "rho", float(r)
        df = None
    else:
        r, p = stats.pearsonr(xv, yv)
        name, label = "Pearson's r", "Pearson correlation"
        lo, hi = effects.fisher_ci(float(r), n, spearman=False)
        df = n - 2
        stat_name = "t"
        stat = float(r) * np.sqrt(df) / np.sqrt(1 - float(r) ** 2) if abs(r) < 1 else float("inf")

    slope = float(np.polyfit(xv, yv, 1)[0]) if np.std(xv) > 0 else float("nan")

    def loo_effect(keep_idx: np.ndarray) -> float:
        sx, sy = xv[keep_idx], yv[keep_idx]
        if len(sx) < 4 or np.std(sx) == 0 or np.std(sy) == 0:
            return float("nan")
        if method == "spearman":
            return float(stats.spearmanr(sx, sy).statistic)
        return float(np.corrcoef(sx, sy)[0, 1])

    threats = []
    threats += dx.missing_note(n, dropped)
    threats += dx.check_bounds(xv, x)
    threats += dx.check_bounds(yv, y)
    if method == "pearson":
        threats += dx.check_normality({x.display: xv, y.display: yv})
    threats += dx.leave_one_out_influence(
        loo_effect, np.arange(n), float(r),
        row_labels=[f"row {i}" for i in kept.index.to_list()])
    if n < 50:
        threats.append(dx.Threat(
            code="small_n_correlation",
            severity="warn",
            message=(f"With n = {n}, a correlation estimate is unstable. The "
                     "interval on this one runs from "
                     f"{lo:.2f} to {hi:.2f}." if lo is not None else
                     f"With n = {n}, a correlation estimate is unstable."),
            action=("Report the interval rather than the point estimate, and "
                    "treat the size of the correlation as unresolved until "
                    "n is closer to 150."),
            values={"n": n},
        ))

    mde = power.mde_correlation(n, alpha)
    summary = PowerSummary(
        alpha=alpha,
        mde=mde,
        mde_label="r",
        observed_power=power.power_correlation(float(r), n, alpha),
        exaggeration=(power.exaggeration_t(2 * float(r) / np.sqrt(1 - float(r) ** 2),
                                           n // 2, n - n // 2, alpha)
                      if abs(r) < 0.999 else None),
    )

    return AnalysisResult(
        analysis=f"{method}_correlation",
        label=label,
        design=design,
        statistic_name=stat_name,
        statistic=stat,
        df=None if df is None else float(df),
        p=float(p),
        alpha=alpha,
        n=n,
        n_missing=dropped,
        effect=Estimate(name=name, value=float(r), ci_low=lo, ci_high=hi,
                        method="Fisher z"),
        raw_effect=Estimate(
            name=f"slope ({y.units} of {y.display} per unit of {x.display})",
            value=slope, method="least squares"),
        groups=[GroupDescriptives(label=x.display, n=n, mean=float(np.mean(xv)),
                                  sd=float(np.std(xv, ddof=1)),
                                  median=float(np.median(xv))),
                GroupDescriptives(label=y.display, n=n, mean=float(np.mean(yv)),
                                  sd=float(np.std(yv, ddof=1)),
                                  median=float(np.median(yv)))],
        variables=[x.name, y.name],
        outcome=y,
        predictor=x,
        power=summary,
        threats=threats,
        extra={"family": "association", "r_squared": float(r) ** 2,
               "method": method},
    )


def chi_square(frame: pd.DataFrame, row_var: Variable, col_var: Variable,
               design: Design, alpha: float = 0.05) -> AnalysisResult:
    kept, dropped = complete_cases(frame, [row_var.name, col_var.name])
    table = pd.crosstab(kept[row_var.name], kept[col_var.name])
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("A chi-square test needs at least two rows and two "
                         "columns with data.")
    chi2, p, df, expected = stats.chi2_contingency(table.to_numpy(),
                                                   correction=False)
    n = int(table.to_numpy().sum())
    rows, cols = table.shape
    v = effects.cramers_v(float(chi2), n, rows, cols)
    lo, hi = effects.cramers_v_ci(float(chi2), int(df), n, rows, cols)

    observed = table.to_numpy(dtype=float)
    residuals = (observed - expected) / np.sqrt(expected)
    flat_idx = int(np.argmax(np.abs(residuals)))
    ri, cij = divmod(flat_idx, cols)
    cell_label = f"{table.index[ri]} / {table.columns[cij]}"
    obs_share = observed[ri, cij] / observed[ri].sum()
    exp_share = expected[ri, cij] / expected[ri].sum()

    threats = []
    threats += dx.missing_note(n, dropped)
    threats += dx.check_expected_counts(expected)

    mde = power.mde_cramers_v(n, int(df), rows, cols, alpha)
    summary = PowerSummary(
        alpha=alpha,
        mde=mde,
        mde_label="Cramer's V",
        observed_power=power.power_chi2(
            v * np.sqrt(min(rows - 1, cols - 1)), n, int(df), alpha),
        exaggeration=power.exaggeration_chi2(float(chi2), int(df), alpha),
    )

    groups = [GroupDescriptives(label=str(idx), n=int(observed[i].sum()))
              for i, idx in enumerate(table.index)]

    return AnalysisResult(
        analysis="chi_square",
        label="Chi-square test of independence",
        design=design,
        statistic_name="chi2",
        statistic=float(chi2),
        df=float(df),
        p=float(p),
        alpha=alpha,
        n=n,
        n_missing=dropped,
        effect=Estimate(name="Cramer's V", value=v, ci_low=lo, ci_high=hi,
                        method="noncentral chi-square"),
        raw_effect=Estimate(
            name=f"percentage points, {cell_label} above what independence "
                 "would give",
            value=float((obs_share - exp_share) * 100), method="descriptive"),
        groups=groups,
        variables=[row_var.name, col_var.name],
        outcome=col_var,
        predictor=row_var,
        power=summary,
        threats=threats,
        extra={"family": "association", "table": table.to_dict(),
               "expected": expected.tolist(),
               "residuals": residuals.tolist(),
               "largest_cell": cell_label,
               "observed_share": float(obs_share),
               "expected_share": float(exp_share),
               "shape": [rows, cols]},
    )
