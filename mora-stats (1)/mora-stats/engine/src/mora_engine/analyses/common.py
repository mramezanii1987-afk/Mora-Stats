"""Shared preparation for every analysis.

Missing data is handled by listwise deletion per analysis in this milestone,
and the number of cases dropped is always carried into the result so that it
can be stated rather than buried.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..models import GroupDescriptives, Variable


def complete_cases(frame: pd.DataFrame, columns: List[str]
                   ) -> Tuple[pd.DataFrame, int]:
    before = len(frame)
    kept = frame.dropna(subset=columns)
    return kept, before - len(kept)


def split_groups(frame: pd.DataFrame, outcome: str, group: str
                 ) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for label, chunk in frame.groupby(group, sort=True, observed=True):
        out[str(label)] = chunk[outcome].to_numpy(dtype=float)
    return out


def describe(label: str, values: np.ndarray, missing: int = 0
             ) -> GroupDescriptives:
    if len(values) == 0:
        return GroupDescriptives(label=label, n=0, missing=missing)
    q1, q3 = np.percentile(values, [25, 75])
    return GroupDescriptives(
        label=label,
        n=int(len(values)),
        mean=float(np.mean(values)),
        sd=float(np.std(values, ddof=1)) if len(values) > 1 else None,
        median=float(np.median(values)),
        iqr=float(q3 - q1),
        missing=missing,
    )


def as_variable(spec, fallback_type) -> Variable:
    if isinstance(spec, Variable):
        return spec
    return Variable(name=str(spec), type=fallback_type)


def row_labels(frame: pd.DataFrame) -> List[str]:
    return [f"row {i}" for i in frame.index.to_list()]
