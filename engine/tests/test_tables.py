"""Results tables are built in the engine, so the interface cannot mangle them."""

import pytest

import mora_engine as me
from mora_engine.models import Design, VarType, Variable
from mora_engine.tables import render_text

SCORE = Variable("score", VarType.CONTINUOUS, unit="points")
ARM = Variable("arm", VarType.NOMINAL)


def keys(interpretation):
    return [t.key for t in interpretation.tables]


def test_a_comparison_produces_four_tables(big_effect):
    _, interpretation = me.run(big_effect, "welch_t", [SCORE, ARM],
                               Design.EXPERIMENTAL)
    assert keys(interpretation) == ["test", "descriptives", "precision"]
    test = interpretation.tables[0]
    assert len(test.rows) == 1
    labels = [c.label for c in test.columns]
    assert labels[:1] == ["Comparison"]
    assert "p" in labels and "95% CI" in labels


def test_figures_are_formatted_by_the_engine(big_effect):
    _, interpretation = me.run(big_effect, "welch_t", [SCORE, ARM],
                               Design.EXPERIMENTAL)
    row = interpretation.tables[0].rows[0]
    assert row["p"] == "< .001"
    assert row["effect"].startswith("-.") or row["effect"].startswith("-1")
    assert row["interval"].startswith("[")


def test_chi_square_adds_a_contingency_table(table_data):
    _, interpretation = me.run(
        table_data, "chi_square",
        [Variable("attended", VarType.NOMINAL),
         Variable("outcome", VarType.NOMINAL)], Design.CORRELATIONAL)
    assert "contingency" in keys(interpretation)
    contingency = next(t for t in interpretation.tables
                       if t.key == "contingency")
    assert contingency.rows
    first = contingency.rows[0]
    assert "(" in [v for k, v in first.items() if k.startswith("c_")
                   and not k.endswith("__flag")][0]


def test_every_analysis_yields_a_test_table(big_effect, three_groups,
                                            linear_data, table_data):
    cases = [
        (big_effect, "welch_t", [SCORE, ARM]),
        (big_effect, "mann_whitney", [SCORE, ARM]),
        (three_groups, "one_way_anova",
         [SCORE, Variable("site", VarType.NOMINAL)]),
        (three_groups, "kruskal_wallis",
         [SCORE, Variable("site", VarType.NOMINAL)]),
        (linear_data, "pearson_correlation",
         [Variable("hours", VarType.CONTINUOUS),
          Variable("grade", VarType.CONTINUOUS)]),
        (table_data, "chi_square",
         [Variable("attended", VarType.NOMINAL),
          Variable("outcome", VarType.NOMINAL)]),
    ]
    for frame, test, variables in cases:
        _, interpretation = me.run(frame, test, variables,
                                   Design.CORRELATIONAL)
        assert interpretation.tables[0].key == "test", test
        assert interpretation.tables[0].rows[0]["term"] != "—", test


def test_precision_table_names_the_exaggeration_when_it_applies(null_data):
    _, interpretation = me.run(null_data, "welch_t", [SCORE, ARM],
                               Design.EXPERIMENTAL)
    precision = next(t for t in interpretation.tables if t.key == "precision")
    quantities = [r["quantity"] for r in precision.rows]
    assert any("Smallest effect" in q for q in quantities)


def test_text_rendering_aligns_columns(big_effect):
    _, interpretation = me.run(big_effect, "welch_t", [SCORE, ARM],
                               Design.EXPERIMENTAL)
    text = render_text(interpretation.tables[1]).split("\n")
    assert text[0] == "Descriptives"
    assert len(set(len(line) for line in text[1:5])) == 1
