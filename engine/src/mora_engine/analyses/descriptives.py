"""Descriptives. Not an inferential test, so it never touches the counter."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .. import diagnostics as dx
from ..models import (AnalysisResult, Design, GroupDescriptives, VarType,
                      Variable)


def descriptives(frame: pd.DataFrame, variables: List[Variable],
                 design: Design, by: Optional[Variable] = None
                 ) -> AnalysisResult:
    rows: List[GroupDescriptives] = []
    threats = []
    detail = {}

    for var in variables:
        series = frame[var.name]
        missing = int(series.isna().sum())
        values = series.dropna()
        if var.type is VarType.NOMINAL:
            counts = values.astype(str).value_counts().to_dict()
            detail[var.name] = {"counts": counts, "missing": missing,
                                "type": var.type.value}
            rows.append(GroupDescriptives(label=var.display,
                                          n=int(len(values)), missing=missing))
            continue
        arr = values.to_numpy(dtype=float)
        if len(arr) == 0:
            rows.append(GroupDescriptives(label=var.display, n=0,
                                          missing=missing))
            continue
        q1, q3 = np.percentile(arr, [25, 75])
        rows.append(GroupDescriptives(
            label=var.display, n=int(len(arr)), mean=float(np.mean(arr)),
            sd=float(np.std(arr, ddof=1)) if len(arr) > 1 else None,
            median=float(np.median(arr)), iqr=float(q3 - q1), missing=missing))
        detail[var.name] = {
            "min": float(np.min(arr)), "max": float(np.max(arr)),
            "skew": float(stats.skew(arr, bias=False)) if len(arr) > 2 else None,
            "kurtosis": float(stats.kurtosis(arr, bias=False)) if len(arr) > 3 else None,
            "missing": missing, "type": var.type.value,
        }
        threats += dx.check_bounds(arr, var)

    return AnalysisResult(
        analysis="descriptives",
        label="Descriptives",
        design=design,
        n=int(len(frame)),
        n_missing=int(sum(r.missing for r in rows)),
        groups=rows,
        variables=[v.name for v in variables],
        threats=threats,
        extra={"family": "descriptives", "detail": detail},
        inferential=False,
    )
