"""Import and type detection.

The imported dataset is immutable. Nothing in the engine writes back to the
frame it is given, and every transform in the app is a visible step that can
be undone, so the file on disk is never the thing that changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .models import VarType, Variable

ORDINAL_MAX_LEVELS = 10
NOMINAL_MAX_LEVELS = 30


def load(path: str | Path, header_rows: int = 1,
         sheet: Optional[str] = None) -> pd.DataFrame:
    """Read a .csv or .xlsx file.

    ``header_rows`` above one flattens a multi-row header by joining the
    levels, which is what spreadsheets exported from survey tools usually
    need.
    """
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        header = list(range(header_rows)) if header_rows > 1 else 0
        frame = pd.read_excel(path, sheet_name=sheet or 0, header=header)
    else:
        header = list(range(header_rows)) if header_rows > 1 else 0
        frame = pd.read_csv(path, header=header)

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [
            " / ".join(str(p) for p in col
                       if str(p) and not str(p).startswith("Unnamed"))
            or f"column_{i}"
            for i, col in enumerate(frame.columns)
        ]
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def detect_type(series: pd.Series) -> VarType:
    values = series.dropna()
    if values.empty:
        return VarType.NOMINAL
    if pd.api.types.is_bool_dtype(values):
        return VarType.NOMINAL
    if pd.api.types.is_numeric_dtype(values):
        distinct = values.unique()
        whole = np.allclose(distinct, np.round(distinct))
        if whole and 2 <= len(distinct) <= ORDINAL_MAX_LEVELS:
            return VarType.ORDINAL
        return VarType.CONTINUOUS
    coerced = pd.to_numeric(values, errors="coerce")
    if coerced.notna().mean() > 0.95:
        return detect_type(coerced.dropna())
    return VarType.NOMINAL


def suggest_bounds(series: pd.Series, var_type: VarType
                   ) -> Tuple[Optional[float], Optional[float]]:
    """Candidate scale limits, offered to the user rather than assumed."""
    if var_type is not VarType.ORDINAL:
        return None, None
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None, None
    return float(values.min()), float(values.max())


def profile(frame: pd.DataFrame, declare_bounds: bool = False
            ) -> List[Variable]:
    """Build the variable list shown in the data pane."""
    out: List[Variable] = []
    for name in frame.columns:
        series = frame[name]
        var_type = detect_type(series)
        lower, upper = (suggest_bounds(series, var_type)
                        if declare_bounds else (None, None))
        out.append(Variable(name=name, type=var_type, lower_bound=lower,
                            upper_bound=upper))
    return out


def variable_summary(frame: pd.DataFrame, variables: List[Variable]
                     ) -> List[Dict]:
    rows = []
    for var in variables:
        series = frame[var.name]
        row: Dict = {
            "name": var.name,
            "type": var.type.value,
            "missing": int(series.isna().sum()),
            "distinct": int(series.nunique(dropna=True)),
        }
        if var.type is not VarType.NOMINAL:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if not numeric.empty:
                row.update({
                    "mean": float(numeric.mean()),
                    "sd": float(numeric.std(ddof=1)) if len(numeric) > 1 else None,
                    "min": float(numeric.min()),
                    "max": float(numeric.max()),
                    "sparkline": _sparkline(numeric.to_numpy()),
                })
        else:
            row["top"] = series.astype(str).value_counts().head(5).to_dict()
        rows.append(row)
    return rows


_BLOCKS = "▁▂▃▄▅▆▇█"


def _sparkline(values: np.ndarray, bins: int = 12) -> str:
    if len(values) == 0:
        return ""
    counts, _ = np.histogram(values, bins=bins)
    if counts.max() == 0:
        return ""
    scaled = (counts / counts.max() * (len(_BLOCKS) - 1)).round().astype(int)
    return "".join(_BLOCKS[i] for i in scaled)
