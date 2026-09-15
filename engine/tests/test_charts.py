"""A chart is offered when it adds something, and never moves between runs."""

import numpy as np
import pandas as pd
import pytest

import mora_engine as me
from mora_engine import charts
from mora_engine.models import Design, VarType, Variable

SCORE = Variable("score", VarType.CONTINUOUS, unit="points")
ARM = Variable("arm", VarType.NOMINAL)
SITE = Variable("site", VarType.NOMINAL)


def test_group_chart_carries_both_summaries_and_raw_values(big_effect):
    result, _ = me.run(big_effect, "welch_t", [SCORE, ARM],
                       Design.EXPERIMENTAL)
    spec = charts.chart_for(result, big_effect)
    assert spec["kind"] == "groups"
    assert len(spec["groups"]) == 2
    first = spec["groups"][0]
    assert first["ci_low"] < first["mean"] < first["ci_high"]
    assert first["q1"] <= first["median"] <= first["q3"]
    assert len(first["values"]) == first["n"]
    assert len(first["jitter"]) == len(first["values"])


def test_a_chart_is_withheld_when_it_would_add_nothing(big_effect):
    result, _ = me.run(big_effect, "descriptives", [SCORE],
                       Design.EXPERIMENTAL)
    spec = charts.chart_for(result, big_effect)
    assert spec["kind"] == "none"
    assert spec["reason"]


def test_missing_data_is_a_reason_rather_than_a_crash(big_effect):
    result, _ = me.run(big_effect, "welch_t", [SCORE, ARM],
                       Design.EXPERIMENTAL)
    spec = charts.chart_for(result, None)
    assert spec["kind"] == "none"
    assert spec["reason"]


def test_small_groups_lead_with_points_not_a_box():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({
        "score": np.r_[rng.normal(50, 8, 6), rng.normal(58, 8, 6)],
        "arm": ["a"] * 6 + ["b"] * 6})
    result, _ = me.run(frame, "welch_t", [SCORE, ARM], Design.EXPERIMENTAL)
    spec = charts.chart_for(result, frame)
    assert spec["default_view"] == "points"
    assert "distribution" not in spec["views"]


def test_rank_tests_default_to_the_distribution_view(big_effect):
    result, _ = me.run(big_effect, "mann_whitney", [SCORE, ARM],
                       Design.CORRELATIONAL)
    spec = charts.chart_for(result, big_effect)
    assert spec["default_view"] == "distribution"


def test_jitter_is_seeded_so_the_picture_never_moves(big_effect):
    result, _ = me.run(big_effect, "welch_t", [SCORE, ARM],
                       Design.EXPERIMENTAL)
    first = charts.chart_for(result, big_effect)
    second = charts.chart_for(result, big_effect)
    assert first["groups"][0]["jitter"] == second["groups"][0]["jitter"]
    assert first["groups"][0]["values"] == second["groups"][0]["values"]


def test_large_datasets_are_thinned_for_drawing_only():
    rng = np.random.default_rng(4)
    n = 4000
    frame = pd.DataFrame({
        "score": np.r_[rng.normal(50, 10, n), rng.normal(52, 10, n)],
        "arm": ["a"] * n + ["b"] * n})
    result, _ = me.run(frame, "welch_t", [SCORE, ARM], Design.CORRELATIONAL)
    spec = charts.chart_for(result, frame)
    assert spec["thinned"] is True
    assert len(spec["groups"][0]["values"]) <= charts.MAX_PLOTTED
    assert spec["groups"][0]["n"] == n          # the summary uses every case


def test_scatter_band_widens_away_from_the_centre(linear_data):
    result, _ = me.run(linear_data, "pearson_correlation",
                       [Variable("hours", VarType.CONTINUOUS),
                        Variable("grade", VarType.CONTINUOUS)],
                       Design.CORRELATIONAL)
    spec = charts.chart_for(result, linear_data)
    assert spec["kind"] == "scatter"
    assert len(spec["band"]) == 40
    assert all(b["low"] < b["y"] < b["high"] for b in spec["band"])
    widths = [b["high"] - b["low"] for b in spec["band"]]
    assert widths[0] > min(widths) and widths[-1] > min(widths)


def test_three_groups_produce_three_columns(three_groups):
    result, _ = me.run(three_groups, "one_way_anova", [SCORE, SITE],
                       Design.QUASI_EXPERIMENTAL)
    spec = charts.chart_for(result, three_groups)
    assert len(spec["groups"]) == 3


def test_mosaic_carries_counts_expected_and_residuals(table_data):
    result, _ = me.run(table_data, "chi_square",
                       [Variable("attended", VarType.NOMINAL),
                        Variable("outcome", VarType.NOMINAL)],
                       Design.CORRELATIONAL)
    spec = charts.chart_for(result, table_data)
    assert spec["kind"] == "mosaic"
    assert spec["views"] == ["share", "residual"]
    assert all({"count", "expected", "residual", "share"} <= set(cell)
               for cell in spec["cells"])
    assert sum(spec["row_totals"]) == result.n
