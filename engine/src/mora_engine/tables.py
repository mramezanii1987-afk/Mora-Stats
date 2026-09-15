"""Results as tables, built by the engine rather than by the interface.

Formatting a p value or an effect size is a statistical decision, not a
display decision, so the engine emits finished strings and the frontend only
lays them out. Adding a test later needs no change in the interface at all.

Every table is a journal table: a stub column on the left, figures aligned on
the right, one note underneath carrying what the reader needs to read it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .fmt import ci, df_str, lead, num, pct, pval_bare
from .models import AnalysisResult

BOUNDED = {"Hedges' g", "Pearson's r", "Spearman's rho", "rank-biserial r",
           "Cramer's V", "omega squared", "epsilon squared"}

SYMBOL = {
    "Hedges' g": "g",
    "Pearson's r": "r",
    "Spearman's rho": "r\u209b",
    "rank-biserial r": "r\u1d63\u1d66",
    "Cramer's V": "V",
    "omega squared": "\u03c9\u00b2",
    "epsilon squared": "\u03b5\u00b2",
}


@dataclass
class Column:
    key: str
    label: str
    align: str = "right"


@dataclass
class Table:
    key: str
    title: str
    columns: List[Column]
    rows: List[Dict[str, Any]] = field(default_factory=list)
    note: str = ""
    emphasis: str = ""   # column key drawn as the figure that carries the result


def _effect_text(result: AnalysisResult) -> str:
    e = result.effect
    if e is None:
        return "—"
    drop = e.name in BOUNDED
    return lead(e.value, 2) if drop else num(e.value, 2)


def _effect_ci(result: AnalysisResult) -> str:
    e = result.effect
    if e is None or e.ci_low is None or e.ci_high is None:
        return "—"
    return ci(e.ci_low, e.ci_high, 2, drop_zero=e.name in BOUNDED)


def _statistic_label(result: AnalysisResult) -> str:
    return {"chi2": "\u03c7\u00b2"}.get(result.statistic_name,
                                        result.statistic_name)


def test_table(result: AnalysisResult) -> Table:
    """One row, the way a results table reads in a paper."""
    e = result.effect
    symbol = SYMBOL.get(e.name, e.name) if e else "—"
    columns = [Column("term", "Comparison", "left")]
    row: Dict[str, Any] = {"term": _comparison_label(result)}

    if result.statistic is not None:
        columns.append(Column("statistic", _statistic_label(result)))
        row["statistic"] = num(result.statistic, 2)
    if result.df is not None:
        columns.append(Column("df", "df"))
        row["df"] = (f"{df_str(result.df)}, {df_str(result.df2)}"
                     if result.df2 is not None else df_str(result.df))
    columns.append(Column("n", "n"))
    row["n"] = str(result.n)
    if result.p is not None:
        columns.append(Column("p", "p"))
        row["p"] = pval_bare(result.p)
    if e is not None:
        columns.append(Column("effect", symbol))
        row["effect"] = _effect_text(result)
        columns.append(Column("interval", "95% CI"))
        row["interval"] = _effect_ci(result)

    notes = []
    if e is not None:
        notes.append(f"{symbol} is {e.name}, interval by {e.method}")
    if result.n_missing:
        notes.append(f"{result.n_missing} cases dropped for missing values")
    return Table(
        key="test",
        title=result.label,
        columns=columns,
        rows=[row],
        note=". ".join(notes) + "." if notes else "",
        emphasis="effect",
    )


def _comparison_label(result: AnalysisResult) -> str:
    if result.analysis.endswith("correlation") or result.analysis == "chi_square":
        return (f"{result.predictor.display} with {result.outcome.display}"
                if result.predictor and result.outcome else "—")
    labels = [g.label for g in result.groups]
    if len(labels) == 2:
        return f"{labels[0]} against {labels[1]}"
    if result.predictor:
        return f"{result.outcome.display} across {result.predictor.display}"
    return "—"


def descriptives_table(result: AnalysisResult) -> Optional[Table]:
    groups = [g for g in result.groups if g.n]
    if not groups:
        return None
    numeric = any(g.mean is not None for g in groups)
    columns = [Column("group", "Group", "left"), Column("n", "n")]
    if numeric:
        columns += [Column("mean", "M"), Column("sd", "SD"),
                    Column("median", "Mdn"), Column("iqr", "IQR")]
    rows = []
    for g in groups:
        row = {"group": g.label, "n": str(g.n)}
        if numeric:
            row.update({
                "mean": num(g.mean) if g.mean is not None else "—",
                "sd": num(g.sd) if g.sd is not None else "—",
                "median": num(g.median) if g.median is not None else "—",
                "iqr": num(g.iqr) if g.iqr is not None else "—",
            })
        rows.append(row)
    unit = result.outcome.units if result.outcome else ""
    return Table(key="descriptives", title="Descriptives",
                 columns=columns, rows=rows,
                 note=f"Figures are in {unit}." if unit and numeric else "")


def precision_table(result: AnalysisResult) -> Optional[Table]:
    p = result.power
    rows = []
    if p.mde is not None:
        label = p.mde_label
        value = lead(p.mde, 2) if label in BOUNDED or label in ("r", "rho") \
            else num(p.mde, 2)
        detail = f"{label}"
        if p.mde_raw is not None and result.outcome is not None:
            detail += f", about {num(p.mde_raw)} {result.outcome.units}"
        rows.append({
            "quantity": f"Smallest effect this design catches "
                        f"{int(p.target_power * 100)} times in 100",
            "value": value,
            "detail": detail,
        })
    if p.observed_power is not None:
        rows.append({
            "quantity": "Chance of catching an effect the size seen here",
            "value": pct(p.observed_power, 0),
            "detail": "at this n and this threshold",
        })
    if p.exaggeration is not None:
        rows.append({
            "quantity": "Overstatement among results that reach significance",
            "value": f"{num(p.exaggeration, 1)}\u00d7",
            "detail": "if the true effect were the size estimated here",
        })
    if result.raw_effect is not None:
        rows.append({
            "quantity": result.raw_effect.name[:1].upper() + result.raw_effect.name[1:],
            "value": num(result.raw_effect.value, 2),
            "detail": result.raw_effect.method,
        })
    if not rows:
        return None
    note = p.note or ""
    return Table(
        key="precision",
        title="Precision of this design",
        columns=[Column("quantity", "Quantity", "left"),
                 Column("value", "Value"),
                 Column("detail", "Read as", "left")],
        rows=rows,
        note=note,
    )


def contingency_table(result: AnalysisResult) -> Optional[Table]:
    if result.analysis != "chi_square":
        return None
    table = result.extra.get("table") or {}
    expected = result.extra.get("expected") or []
    residuals = result.extra.get("residuals") or []
    columns_keys = list(table.keys())
    if not columns_keys:
        return None
    row_keys = list(table[columns_keys[0]].keys())

    columns = [Column("row", result.predictor.display if result.predictor else "", "left")]
    for c in columns_keys:
        columns.append(Column(f"c_{c}", str(c)))
    columns.append(Column("total", "Total"))

    rows = []
    for i, rk in enumerate(row_keys):
        row: Dict[str, Any] = {"row": str(rk)}
        total = 0.0
        for j, ck in enumerate(columns_keys):
            count = float(table[ck][rk])
            total += count
            exp = expected[i][j] if i < len(expected) and j < len(expected[i]) else None
            res = residuals[i][j] if i < len(residuals) and j < len(residuals[i]) else 0.0
            text = f"{int(count)}"
            if exp is not None:
                text += f" ({num(exp, 1)})"
            row[f"c_{ck}"] = text
            row[f"c_{ck}__flag"] = abs(res) > 1.96
        row["total"] = f"{int(total)}"
        rows.append(row)

    return Table(
        key="contingency",
        title="Counts, with what independence would give in brackets",
        columns=columns,
        rows=rows,
        note="Cells further than two standard residuals from independence are "
             "marked.",
    )


def tables_for(result: AnalysisResult) -> List[Table]:
    if not result.inferential:
        descriptives = descriptives_table(result)
        return [descriptives] if descriptives else []
    out: List[Optional[Table]] = [test_table(result),
                                  descriptives_table(result),
                                  contingency_table(result),
                                  precision_table(result)]
    return [t for t in out if t is not None]


def render_text(table: Table, width: int = 78) -> str:
    """Plain text rendering, used by the command line and the demo."""
    widths = {}
    for column in table.columns:
        cells = [str(row.get(column.key, "")) for row in table.rows]
        widths[column.key] = max(len(column.label), *(len(c) for c in cells)) \
            if cells else len(column.label)

    def line(cells, align_left_first=True):
        parts = []
        for column in table.columns:
            text = str(cells.get(column.key, ""))
            pad = widths[column.key]
            parts.append(text.ljust(pad) if column.align == "left"
                         else text.rjust(pad))
        return "  ".join(parts)

    header = line({c.key: c.label for c in table.columns})
    rule = "-" * len(header)
    body = [line(row) for row in table.rows]
    out = [table.title, rule, header, rule, *body, rule]
    if table.note:
        out.append(f"Note. {table.note}")
    return "\n".join(out)
