"""Import, type detection and the variable list."""

import pandas as pd
import pytest

from mora_engine import dataio
from mora_engine.models import VarType


def test_type_detection_rules(tmp_path):
    frame = pd.DataFrame({
        "height": [172.4, 165.1, 180.9, 159.2, 175.0],
        "likert": [1, 3, 5, 4, 2],
        "city": ["Melbourne", "Perth", "Melbourne", "Hobart", "Perth"],
        "flag": [True, False, True, True, False],
        "numeric_text": ["1.5", "2.5", "3.5", "4.5", "5.5"],
    })
    variables = {v.name: v for v in dataio.profile(frame)}
    assert variables["height"].type is VarType.CONTINUOUS
    assert variables["likert"].type is VarType.ORDINAL
    assert variables["city"].type is VarType.NOMINAL
    assert variables["flag"].type is VarType.NOMINAL
    assert variables["numeric_text"].type is VarType.CONTINUOUS


def test_bounds_are_suggested_not_declared(tmp_path):
    frame = pd.DataFrame({"likert": [1, 2, 3, 4, 5, 5, 4]})
    plain = dataio.profile(frame)[0]
    assert plain.lower_bound is None
    offered = dataio.profile(frame, declare_bounds=True)[0]
    assert (offered.lower_bound, offered.upper_bound) == (1.0, 5.0)


def test_csv_round_trip(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1.5, 2.5], "b": ["x", "y"]}).to_csv(path, index=False)
    frame = dataio.load(path)
    assert list(frame.columns) == ["a", "b"]
    assert len(frame) == 2


def test_multi_row_headers_are_flattened(tmp_path):
    path = tmp_path / "messy.xlsx"
    frame = pd.DataFrame([
        ["Baseline", "Baseline", "Follow up"],
        ["score", "time", "score"],
        [10, 1, 12],
        [11, 2, 13],
    ])
    frame.to_excel(path, index=False, header=False)
    loaded = dataio.load(path, header_rows=2)
    assert loaded.columns[0] == "Baseline / score"
    assert loaded.columns[2] == "Follow up / score"
    assert len(loaded) == 2


def test_variable_summary_reports_missing_and_sparkline():
    frame = pd.DataFrame({"x": [1.0, 2.0, None, 4.0, 5.0, 6.0]})
    variables = dataio.profile(frame)
    summary = dataio.variable_summary(frame, variables)[0]
    assert summary["missing"] == 1
    assert summary["sparkline"]
