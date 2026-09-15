"""MoRa Stats engine.

Deterministic statistics and rule-based interpretation. No model, no network,
no API key. The same data produces the same words on every machine, and a
reviewer can trace every sentence back to a number in ``provenance``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from . import analyses, dataio, diagnostics, effects, power, report, suggest
from .interpret import BOUNDARY_NOTE, interpret, language_violations
from .models import (AnalysisResult, Design, Estimate, Interpretation,
                     Suggestion, Threat, VarType, Variable)
from .session import SessionLedger

__version__ = "0.1.0"

ANALYSES = {
    "descriptives": "Descriptives",
    "student_t": "Student's t-test",
    "welch_t": "Welch's t-test",
    "mann_whitney": "Mann-Whitney U",
    "one_way_anova": "One-way ANOVA",
    "kruskal_wallis": "Kruskal-Wallis",
    "pearson_correlation": "Pearson correlation",
    "spearman_correlation": "Spearman correlation",
    "chi_square": "Chi-square test of independence",
}


def run(frame: pd.DataFrame, test: str, variables: Sequence[Variable],
        design: Design, alpha: float = 0.05,
        ledger: Optional[SessionLedger] = None,
        filter_key: str = "") -> Tuple[AnalysisResult, Interpretation]:
    """Run one analysis and interpret it.

    ``variables`` is ordered. For group comparisons it is (outcome, group).
    For correlation it is (x, y). For chi-square it is (rows, columns).
    """
    if test == "descriptives":
        result = analyses.descriptives(frame, list(variables), design)
    elif test in ("welch_t", "student_t"):
        result = analyses.t_test(frame, variables[0], variables[1], design,
                                 welch=(test == "welch_t"), alpha=alpha)
    elif test == "mann_whitney":
        result = analyses.mann_whitney(frame, variables[0], variables[1],
                                       design, alpha=alpha)
    elif test == "one_way_anova":
        result = analyses.one_way_anova(frame, variables[0], variables[1],
                                        design, alpha=alpha)
    elif test == "kruskal_wallis":
        result = analyses.kruskal_wallis(frame, variables[0], variables[1],
                                         design, alpha=alpha)
    elif test in ("pearson_correlation", "spearman_correlation"):
        result = analyses.correlation(
            frame, variables[0], variables[1], design,
            method="spearman" if test.startswith("spearman") else "pearson",
            alpha=alpha)
    elif test == "chi_square":
        result = analyses.chi_square(frame, variables[0], variables[1],
                                     design, alpha=alpha)
    else:
        raise ValueError(f"Unknown analysis '{test}'. Available: "
                         f"{', '.join(sorted(ANALYSES))}.")

    return result, interpret(result, ledger=ledger, filter_key=filter_key)


__all__ = [
    "ANALYSES", "AnalysisResult", "BOUNDARY_NOTE", "Design", "Estimate",
    "Interpretation", "SessionLedger", "Suggestion", "Threat", "VarType",
    "Variable", "analyses", "dataio", "diagnostics", "effects", "interpret",
    "language_violations", "power", "report", "run", "suggest", "__version__",
]
