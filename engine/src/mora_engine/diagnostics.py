"""Threat checks. Every one is computed from this dataset, never guessed.

Thresholds are constants at the top of the file so that a reviewer can read
them in one place and a user can be told exactly why a warning fired.

On normality, the engine deliberately keeps Shapiro-Wilk out of the decision
path. That test is underpowered at the small n where non-normality actually
threatens a t-test, and it is close to certain to reject at the large n
where the same test is robust, which is the wrong way round. The decision
rule uses skewness and kurtosis conditioned on cell size instead, and
Shapiro-Wilk is still reported in the assumptions detail for people who
expect to see it.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats

from .models import Threat, Variable

MIN_CELL_N = 20
MAX_CELL_RATIO = 1.5
BOUND_SHARE = 0.15              # floor or ceiling if more than 15% sit at a bound
INFLUENCE_SHIFT = 0.10          # absolute shift in the standardised effect
INFLUENCE_RELATIVE = 0.10       # and at least this share of the effect itself
INFLUENCE_LARGE = 0.30          # any shift this big is flagged on its own
INFLUENCE_MAX_N = 2000
SKEW_SEVERE = 2.0
KURTOSIS_SEVERE = 7.0
SKEW_MODERATE = 1.0
SMALL_CELL_FOR_SKEW = 25
MIN_EXPECTED_COUNT = 5
VARIANCE_RATIO_FLAG = 2.0


def check_cell_sizes(groups: Dict[str, np.ndarray]) -> List[Threat]:
    threats: List[Threat] = []
    sizes = {k: len(v) for k, v in groups.items()}
    if not sizes:
        return threats
    smallest = min(sizes, key=lambda k: sizes[k])
    largest = max(sizes, key=lambda k: sizes[k])
    if sizes[smallest] < MIN_CELL_N:
        threats.append(Threat(
            code="small_cell",
            severity="warn",
            message=(f"The smallest group has {sizes[smallest]} cases "
                     f"({smallest}), below the {MIN_CELL_N} that the interval "
                     "around this effect assumes to behave well."),
            action=("Read the confidence interval as the result rather than "
                    "the p value, and collect more cases in this group before "
                    "treating the estimate as settled."),
            values={"sizes": sizes, "smallest": smallest},
        ))
    if sizes[smallest] > 0:
        ratio = sizes[largest] / sizes[smallest]
        if ratio > MAX_CELL_RATIO:
            threats.append(Threat(
                code="unbalanced",
                severity="note",
                message=(f"Group sizes are unbalanced at {ratio:.2f} to 1 "
                         f"({largest} = {sizes[largest]}, {smallest} = "
                         f"{sizes[smallest]})."),
                action=("Unequal groups make the test sensitive to unequal "
                        "variances. Use the Welch version, which does not "
                        "assume equal variances, and report it as such."),
                values={"ratio": ratio, "sizes": sizes},
            ))
    return threats


def check_variance_ratio(groups: Dict[str, np.ndarray]) -> List[Threat]:
    variances = {k: float(np.var(v, ddof=1)) for k, v in groups.items()
                 if len(v) > 1}
    if len(variances) < 2:
        return []
    lo = min(variances.values())
    hi = max(variances.values())
    if lo <= 0:
        return []
    ratio = hi / lo
    if ratio <= VARIANCE_RATIO_FLAG:
        return []
    return [Threat(
        code="variance_ratio",
        severity="warn",
        message=(f"The largest group variance is {ratio:.1f} times the "
                 "smallest, so the groups are not equally spread."),
        action=("Report the Welch version of this test, which corrects the "
                "degrees of freedom for unequal variances and costs almost "
                "nothing when variances are in fact equal."),
        values={"ratio": ratio, "variances": variances},
    )]


def check_bounds(values: np.ndarray, variable: Optional[Variable],
                 where: str = "") -> List[Threat]:
    """Floor and ceiling effects against the declared scale bounds.

    Declared bounds matter. A one to seven scale on which nobody chose seven
    cannot be distinguished from a one to six scale by looking at the data,
    so when bounds are missing the engine says so instead of guessing.
    """
    if variable is None or len(values) == 0:
        return []
    lower, upper = variable.lower_bound, variable.upper_bound
    if lower is None and upper is None:
        distinct = np.unique(values)
        if len(distinct) <= 12 and np.allclose(distinct, np.round(distinct)):
            return [Threat(
                code="bounds_unknown",
                severity="note",
                message=(f"{variable.display} looks like a rating scale with "
                         f"{len(distinct)} points, but its scale limits are "
                         "not declared, so floor and ceiling effects cannot "
                         "be checked."),
                action=(f"Set the scale minimum and maximum for "
                        f"{variable.display} in the variable list and this "
                        "check will run automatically."),
                values={"distinct": int(len(distinct))},
            )]
        return []

    threats: List[Threat] = []
    n = len(values)
    if lower is not None:
        share = float(np.mean(values <= lower))
        if share > BOUND_SHARE:
            threats.append(Threat(
                code="floor",
                severity="warn",
                message=(f"{share * 100:.0f}% of {variable.display} scores "
                         f"sit at the scale minimum of {lower:g}{where}, so "
                         "the measure cannot show anyone doing worse."),
                action=("A floor effect compresses real differences toward "
                        "zero, so treat this effect as a lower bound. Use a "
                        "measure with more room at the bottom, or analyse "
                        "the ranks rather than the scores."),
                values={"share": share, "bound": lower, "n": n},
            ))
    if upper is not None:
        share = float(np.mean(values >= upper))
        if share > BOUND_SHARE:
            threats.append(Threat(
                code="ceiling",
                severity="warn",
                message=(f"{share * 100:.0f}% of {variable.display} scores "
                         f"sit at the scale maximum of {upper:g}{where}, so "
                         "the measure cannot show anyone doing better."),
                action=("A ceiling effect compresses real differences toward "
                        "zero, so treat this effect as a lower bound. Use a "
                        "harder version of the measure, or analyse the ranks "
                        "rather than the scores."),
                values={"share": share, "bound": upper, "n": n},
            ))
    return threats


def normality_report(values: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {
        "n": int(len(values)),
        "skew": float(stats.skew(values, bias=False)) if len(values) > 2 else float("nan"),
        "kurtosis": float(stats.kurtosis(values, bias=False)) if len(values) > 3 else float("nan"),
    }
    if 3 <= len(values) <= 5000:
        w, p = stats.shapiro(values)
        out["shapiro_w"] = float(w)
        out["shapiro_p"] = float(p)
    return out


def check_normality(groups: Dict[str, np.ndarray],
                    outcome: Optional[Variable] = None) -> List[Threat]:
    threats: List[Threat] = []
    label = outcome.display if outcome else "the outcome"
    for name, values in groups.items():
        if len(values) < 4:
            continue
        rep = normality_report(values)
        skew, kurt, n = rep["skew"], rep["kurtosis"], rep["n"]
        severe = abs(skew) > SKEW_SEVERE or abs(kurt) > KURTOSIS_SEVERE
        moderate_small = n < SMALL_CELL_FOR_SKEW and abs(skew) > SKEW_MODERATE
        if severe or moderate_small:
            tail = ("which is far enough from a normal shape to bend the "
                    "result" if severe else
                    "which is a mild departure from a normal shape, and at "
                    "this sample size that is enough to be worth checking")
            threats.append(Threat(
                code="non_normal",
                severity="warn" if severe else "note",
                message=(f"{label} in {name} is skewed at {skew:.2f} with "
                         f"kurtosis {kurt:.2f} at n = {n}, {tail}."),
                action=("Run the rank-based version of this test, which does "
                        "not assume a normal distribution, and report that "
                        "one instead. The app has it selected in the list of "
                        "suitable tests."),
                values=rep,
            ))
    return threats


def leave_one_out_influence(effect_fn: Callable[[np.ndarray], float],
                            index: np.ndarray,
                            observed: float,
                            row_labels: Optional[Sequence] = None
                            ) -> List[Threat]:
    """Recompute the standardised effect without each case in turn.

    Cook's distance is the familiar name for this idea, and it belongs to
    regression. For a two-group comparison or a correlation the honest
    analogue is to drop one case at a time and look at how far the effect
    moves, which is what this does.
    """
    n = len(index)
    if n > INFLUENCE_MAX_N or n < 5:
        return []
    shifts = np.empty(n)
    for i in range(n):
        keep = np.delete(index, i)
        try:
            shifts[i] = effect_fn(keep)
        except Exception:
            shifts[i] = np.nan
    deltas = shifts - observed
    if not np.any(np.isfinite(deltas)):
        return []
    worst = int(np.nanargmax(np.abs(deltas)))
    delta = float(deltas[worst])
    relative = abs(delta) >= INFLUENCE_RELATIVE * abs(observed) if observed else True
    if abs(delta) < INFLUENCE_LARGE and not (
            abs(delta) >= INFLUENCE_SHIFT and relative):
        return []
    label = str(row_labels[worst]) if row_labels is not None else f"row {worst + 1}"
    direction = "down" if delta < 0 else "up"
    return [Threat(
        code="influential_case",
        severity="warn",
        message=(f"One case ({label}) is carrying more of this result than "
                 f"the rest. Dropping it alone moves the effect from "
                 f"{observed:.2f} to {observed + delta:.2f}, a shift of "
                 f"{abs(delta):.2f}."),
        action=("Check that case for a data entry error or a genuine reason "
                "to exclude it, and report the analysis both with and "
                "without it. Do not drop it only because it is inconvenient."),
        values={"row": label, "delta": delta, "without": observed + delta},
    )]


def check_expected_counts(expected: np.ndarray) -> List[Threat]:
    flat = expected.flatten()
    below = int(np.sum(flat < MIN_EXPECTED_COUNT))
    if below == 0:
        return []
    share = below / flat.size
    return [Threat(
        code="expected_counts",
        severity="serious" if share > 0.2 else "warn",
        message=(f"{below} of {flat.size} cells "
                 f"{'has' if below == 1 else 'have'} an expected count below "
                 f"{MIN_EXPECTED_COUNT} (the smallest is "
                 f"{flat.min():.2f}), which the chi-square approximation "
                 "does not handle."),
        action=("Use Fisher's exact test for a two by two table, or combine "
                "categories that belong together on substantive grounds "
                "before rerunning."),
        values={"below": below, "cells": int(flat.size),
                "min_expected": float(flat.min())},
    )]


def missing_note(n_used: int, n_missing: int) -> List[Threat]:
    total = n_used + n_missing
    if total == 0 or n_missing == 0:
        return []
    share = n_missing / total
    if share < 0.05:
        return []
    return [Threat(
        code="missing",
        severity="warn" if share > 0.10 else "note",
        message=(f"{n_missing} of {total} cases ({share * 100:.0f}%) were "
                 "dropped because a value used by this analysis was missing."),
        action=("Open the missing data panel to see whether the dropped cases "
                "differ from the rest. If they do, listwise deletion is "
                "biasing this estimate and you need a different approach."),
        values={"n_missing": n_missing, "n_total": total, "share": share},
    )]
