"""Which tests are available for the variables the user picked, and why not.

The app never presents a menu of tests that will fail. It presents the ones
that fit, marks one as the default, and gives a single line of reason for
each one that is greyed out. The reason is always about this data.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .diagnostics import (KURTOSIS_SEVERE, MIN_CELL_N, MIN_EXPECTED_COUNT,
                          SKEW_MODERATE, SKEW_SEVERE, SMALL_CELL_FOR_SKEW)
from .models import Suggestion, VarType, Variable


def _group_arrays(frame: pd.DataFrame, outcome: Variable,
                  group: Variable) -> Dict[str, np.ndarray]:
    kept = frame.dropna(subset=[outcome.name, group.name])
    out: Dict[str, np.ndarray] = {}
    for label, chunk in kept.groupby(group.name, sort=True, observed=True):
        out[str(label)] = chunk[outcome.name].to_numpy(dtype=float)
    return out


def _badly_shaped(values: np.ndarray) -> bool:
    if len(values) < 4:
        return True
    skew = float(stats.skew(values, bias=False))
    kurt = float(stats.kurtosis(values, bias=False))
    if abs(skew) > SKEW_SEVERE or abs(kurt) > KURTOSIS_SEVERE:
        return True
    return len(values) < SMALL_CELL_FOR_SKEW and abs(skew) > SKEW_MODERATE


def suggest_tests(frame: pd.DataFrame, variables: Sequence[Variable]
                  ) -> List[Suggestion]:
    """Given one or two selected variables, list the tests that fit."""
    if len(variables) != 2:
        return [Suggestion("descriptives", "Descriptives", True,
                           "Always available.", recommended=True)]

    first, second = variables
    types = {first.type, second.type}

    # continuous or ordinal outcome, categorical predictor
    if VarType.NOMINAL in types and VarType.NOMINAL not in (
            {first.type} if second.type is VarType.NOMINAL else {second.type}):
        group = first if first.type is VarType.NOMINAL else second
        outcome = second if first.type is VarType.NOMINAL else first
        return _numeric_by_group(frame, outcome, group)

    if types == {VarType.NOMINAL}:
        return _two_nominal(frame, first, second)

    return _two_numeric(frame, first, second)


def _numeric_by_group(frame: pd.DataFrame, outcome: Variable,
                      group: Variable) -> List[Suggestion]:
    groups = _group_arrays(frame, outcome, group)
    k = len(groups)
    sizes = {label: len(values) for label, values in groups.items()}
    shape_problem = any(_badly_shaped(v) for v in groups.values())
    tiny = any(n < MIN_CELL_N for n in sizes.values())
    ordinal_outcome = outcome.type is VarType.ORDINAL

    out: List[Suggestion] = []
    two_ok = k == 2
    two_reason = ("" if two_ok else
                  f"{group.display} has {k} groups with data and this test "
                  "compares exactly two.")

    prefer_rank = shape_problem or tiny or ordinal_outcome
    rank_reason = []
    if ordinal_outcome:
        rank_reason.append(f"{outcome.display} is ordinal")
    if shape_problem:
        rank_reason.append("at least one group is far from a normal shape")
    if tiny:
        rank_reason.append("at least one group is small")

    out.append(Suggestion(
        "welch_t", "Welch's t-test", two_ok,
        two_reason or ("Compares two means without assuming the groups are "
                       "equally spread."),
        recommended=two_ok and not prefer_rank))
    out.append(Suggestion(
        "student_t", "Student's t-test", two_ok,
        two_reason or ("Assumes the two groups are equally spread. Welch "
                       "costs almost nothing and drops that assumption.")))
    out.append(Suggestion(
        "mann_whitney", "Mann-Whitney U", two_ok,
        two_reason or ("Compares the distributions by rank, which needs no "
                       "assumption about their shape."),
        recommended=two_ok and prefer_rank))

    k_ok = k >= 3
    k_reason = ("" if k_ok else
                f"{group.display} has {k} groups with data and this test "
                "needs three or more. With two, use a t-test.")
    out.append(Suggestion(
        "one_way_anova", "One-way ANOVA", k_ok,
        k_reason or "Compares three or more means at once.",
        recommended=k_ok and not prefer_rank))
    out.append(Suggestion(
        "kruskal_wallis", "Kruskal-Wallis", k_ok,
        k_reason or "Compares three or more groups by rank.",
        recommended=k_ok and prefer_rank))

    if prefer_rank and (two_ok or k_ok):
        note = " and ".join(rank_reason)
        for s in out:
            if s.test in ("mann_whitney", "kruskal_wallis") and s.recommended:
                s.reason = f"Suggested because {note}."
    out.append(Suggestion("descriptives", "Descriptives", True,
                          "Always available."))
    return out


def _two_numeric(frame: pd.DataFrame, x: Variable,
                 y: Variable) -> List[Suggestion]:
    kept = frame.dropna(subset=[x.name, y.name])
    xv = kept[x.name].to_numpy(dtype=float)
    yv = kept[y.name].to_numpy(dtype=float)
    n = len(xv)
    ordinal = VarType.ORDINAL in (x.type, y.type)
    shape_problem = _badly_shaped(xv) or _badly_shaped(yv)
    prefer_rank = ordinal or shape_problem
    enough = n >= 4
    reason = "" if enough else f"Only {n} complete cases, which is too few."

    out = [
        Suggestion("pearson_correlation", "Pearson correlation", enough,
                   reason or ("Measures how closely the two move together in "
                              "a straight line."),
                   recommended=enough and not prefer_rank),
        Suggestion("spearman_correlation", "Spearman correlation", enough,
                   reason or ("Measures how closely their ranks move "
                              "together, with no assumption of a straight "
                              "line or a normal shape."),
                   recommended=enough and prefer_rank),
        Suggestion("descriptives", "Descriptives", True, "Always available."),
    ]
    return out


def _two_nominal(frame: pd.DataFrame, a: Variable,
                 b: Variable) -> List[Suggestion]:
    kept = frame.dropna(subset=[a.name, b.name])
    table = pd.crosstab(kept[a.name], kept[b.name])
    ok = table.shape[0] >= 2 and table.shape[1] >= 2
    reason = "" if ok else "Both variables need at least two categories with data."
    small_expected = False
    if ok:
        _, _, _, expected = stats.chi2_contingency(table.to_numpy(),
                                                   correction=False)
        small_expected = bool((expected < MIN_EXPECTED_COUNT).any())
    text = ("Tests whether the two classifications go together.")
    if small_expected:
        text = ("Runs, but some expected counts are below "
                f"{MIN_EXPECTED_COUNT}, so the result will carry a warning "
                "and Fisher's exact test would be safer for a two by two "
                "table.")
    return [
        Suggestion("chi_square", "Chi-square test of independence", ok,
                   reason or text, recommended=ok),
        Suggestion("descriptives", "Descriptives", True, "Always available."),
    ]


def assumption_report(frame: pd.DataFrame, outcome: Variable,
                      group: Optional[Variable] = None) -> Dict[str, dict]:
    """Plain-language assumption summary for the analysis pane."""
    from .diagnostics import normality_report

    report: Dict[str, dict] = {}
    if group is None:
        values = frame[outcome.name].dropna().to_numpy(dtype=float)
        report[outcome.display] = normality_report(values)
        return report
    for label, values in _group_arrays(frame, outcome, group).items():
        report[label] = normality_report(values)
    return report
