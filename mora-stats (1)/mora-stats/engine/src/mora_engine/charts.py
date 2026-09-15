"""Chart data, built by the engine and drawn by the interface.

The engine decides whether a chart earns its place, which views of the data
are honest at this sample size, and what every number in it is. The frontend
draws and animates. That split keeps the picture reproducible: the same data
gives the same coordinates on every machine, including the jitter, which is
seeded rather than random.

A chart is offered only when it shows something the table cannot. Two means
are a table. The same two means with every observation behind them is a
picture worth having.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from scipy import stats

from .models import AnalysisResult

MAX_PLOTTED = 1200      # beyond this, points are thinned for drawing only
JITTER_SEED = 20260914


def _thin(values: np.ndarray) -> tuple[np.ndarray, bool]:
    if len(values) <= MAX_PLOTTED:
        return values, False
    stride = int(np.ceil(len(values) / MAX_PLOTTED))
    return values[::stride], True


def _jitter(n: int, spread: float = 0.33) -> List[float]:
    """Seeded offsets, so a strip of points never moves between runs."""
    rng = np.random.default_rng(JITTER_SEED + n)
    return [float(v) for v in rng.uniform(-spread, spread, n)]


def chart_for(result: AnalysisResult, frame=None) -> Dict[str, Any]:
    a = result.analysis
    if a in ("student_t", "welch_t", "one_way_anova"):
        return _groups(result, frame, parametric=True)
    if a in ("mann_whitney", "kruskal_wallis"):
        return _groups(result, frame, parametric=False)
    if a.endswith("correlation"):
        return _scatter(result, frame)
    if a == "chi_square":
        return _mosaic(result)
    return {"kind": "none",
            "reason": "This analysis is read from the table above."}


# --------------------------------------------------------------------------
# groups
# --------------------------------------------------------------------------

def _groups(result: AnalysisResult, frame, parametric: bool) -> Dict[str, Any]:
    if frame is None or len(result.variables) < 2:
        return {"kind": "none", "reason": "No data was available to draw."}
    outcome, group = result.variables
    kept = frame.dropna(subset=[outcome, group])
    groups: List[Dict[str, Any]] = []
    thinned = False

    for descriptive in result.groups:
        values = kept.loc[kept[group].astype(str) == descriptive.label,
                          outcome].to_numpy(dtype=float)
        if values.size == 0:
            continue
        shown, was_thinned = _thin(np.sort(values))
        thinned = thinned or was_thinned
        q1, median, q3 = (float(v) for v in np.percentile(values, [25, 50, 75]))
        iqr = q3 - q1
        inside = values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]
        mean = float(np.mean(values))
        n = int(values.size)
        if n > 1:
            se = float(np.std(values, ddof=1)) / np.sqrt(n)
            crit = float(stats.t.ppf(0.975, n - 1))
            ci_low, ci_high = mean - crit * se, mean + crit * se
        else:
            ci_low = ci_high = mean
        groups.append({
            "label": descriptive.label,
            "n": n,
            "mean": mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "median": median,
            "q1": q1,
            "q3": q3,
            "whisker_low": float(inside.min()) if inside.size else float(values.min()),
            "whisker_high": float(inside.max()) if inside.size else float(values.max()),
            "values": [float(v) for v in shown],
            "jitter": _jitter(int(shown.size)),
        })

    if len(groups) < 2:
        return {"kind": "none",
                "reason": "Only one group has data, so there is nothing to "
                          "compare."}

    smallest = min(g["n"] for g in groups)
    if smallest < 8:
        # A box promises a shape that eight numbers cannot support. The raw
        # values make no such promise, so they lead instead.
        views = ["points", "intervals"]
        default = "points"
    else:
        views = ["intervals", "distribution", "points"]
        default = "intervals" if parametric else "distribution"

    return {
        "kind": "groups",
        "title": f"{result.outcome.display} by {result.predictor.display}",
        "unit": result.outcome.units if result.outcome else "",
        "y_label": result.outcome.display if result.outcome else outcome,
        "x_label": result.predictor.display if result.predictor else group,
        "views": views,
        "default_view": default,
        "groups": groups,
        "thinned": thinned,
    }


# --------------------------------------------------------------------------
# scatter
# --------------------------------------------------------------------------

def _scatter(result: AnalysisResult, frame) -> Dict[str, Any]:
    if frame is None:
        return {"kind": "none", "reason": "No data was available to draw."}
    x_name, y_name = result.variables
    kept = frame.dropna(subset=[x_name, y_name])
    xs = kept[x_name].to_numpy(dtype=float)
    ys = kept[y_name].to_numpy(dtype=float)
    if len(xs) < 3:
        return {"kind": "none",
                "reason": "Too few complete cases to plot."}

    order = np.argsort(xs)
    xs_sorted, ys_sorted = xs[order], ys[order]
    shown_x, thinned = _thin(xs_sorted)
    shown_y, _ = _thin(ys_sorted)

    fit = None
    band: List[Dict[str, float]] = []
    if np.std(xs) > 0 and len(xs) > 3:
        slope, intercept = (float(v) for v in np.polyfit(xs, ys, 1))
        residuals = ys - (slope * xs + intercept)
        s = float(np.sqrt(np.sum(residuals ** 2) / (len(xs) - 2)))
        mean_x = float(np.mean(xs))
        sxx = float(np.sum((xs - mean_x) ** 2))
        crit = float(stats.t.ppf(0.975, len(xs) - 2))
        for point in np.linspace(float(xs.min()), float(xs.max()), 40):
            centre = slope * point + intercept
            se = s * np.sqrt(1 / len(xs) + (point - mean_x) ** 2 / sxx)
            band.append({"x": float(point), "y": float(centre),
                         "low": float(centre - crit * se),
                         "high": float(centre + crit * se)})
        fit = {"slope": slope, "intercept": intercept,
               "x1": float(xs.min()),
               "y1": float(slope * float(xs.min()) + intercept),
               "x2": float(xs.max()),
               "y2": float(slope * float(xs.max()) + intercept)}

    return {
        "kind": "scatter",
        "title": f"{result.outcome.display} against {result.predictor.display}",
        "x_label": result.predictor.display if result.predictor else x_name,
        "y_label": result.outcome.display if result.outcome else y_name,
        "views": ["fit", "points"] if fit else ["points"],
        "default_view": "fit" if fit else "points",
        "points": [{"x": float(a), "y": float(b)}
                   for a, b in zip(shown_x, shown_y)],
        "fit": fit,
        "band": band,
        "thinned": thinned,
    }


# --------------------------------------------------------------------------
# contingency
# --------------------------------------------------------------------------

def _mosaic(result: AnalysisResult) -> Dict[str, Any]:
    table = result.extra.get("table") or {}
    expected = result.extra.get("expected") or []
    residuals = result.extra.get("residuals") or []
    column_keys = list(table.keys())
    if not column_keys:
        return {"kind": "none", "reason": "The table was empty."}
    row_keys = list(table[column_keys[0]].keys())

    cells = []
    row_totals = []
    for i, rk in enumerate(row_keys):
        total = sum(float(table[ck][rk]) for ck in column_keys)
        row_totals.append(total)
        for j, ck in enumerate(column_keys):
            count = float(table[ck][rk])
            cells.append({
                "row": str(rk),
                "column": str(ck),
                "count": count,
                "expected": (float(expected[i][j])
                             if i < len(expected) and j < len(expected[i])
                             else None),
                "residual": (float(residuals[i][j])
                             if i < len(residuals) and j < len(residuals[i])
                             else 0.0),
                "share": count / total if total else 0.0,
            })

    return {
        "kind": "mosaic",
        "title": f"{result.outcome.display} within {result.predictor.display}",
        "x_label": result.outcome.display if result.outcome else "",
        "y_label": result.predictor.display if result.predictor else "",
        "views": ["share", "residual"],
        "default_view": "share",
        "rows": [str(r) for r in row_keys],
        "columns": [str(c) for c in column_keys],
        "row_totals": row_totals,
        "cells": cells,
        "thinned": False,
    }
