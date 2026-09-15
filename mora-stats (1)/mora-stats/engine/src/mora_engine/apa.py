"""Copy-ready APA 7th edition results sentences.

One function per analysis family. The output is meant to be pasted into a
manuscript without editing, so it carries the statistic, the degrees of
freedom, the exact p value and the effect size with its interval.
"""

from __future__ import annotations

from .fmt import ci, df_str, lead, num, pval
from .models import AnalysisResult


def _effect_clause(result: AnalysisResult) -> str:
    e = result.effect
    if e is None:
        return ""
    bounded = e.name in ("Hedges' g", "Pearson's r", "Spearman's rho",
                         "rank-biserial r", "Cramer's V", "omega squared",
                         "epsilon squared")
    symbol = {
        "Hedges' g": "g",
        "Pearson's r": "r",
        "Spearman's rho": "rs",
        "rank-biserial r": "rrb",
        "Cramer's V": "V",
        "omega squared": "omega2",
        "epsilon squared": "epsilon2",
    }.get(e.name, e.name)
    value = lead(e.value, 2) if bounded else num(e.value, 2)
    if e.ci_low is None or e.ci_high is None:
        return f"{symbol} = {value}"
    interval = ci(e.ci_low, e.ci_high, 2, drop_zero=bounded)
    return f"{symbol} = {value}, 95% CI {interval}"


def _group_clause(result: AnalysisResult) -> str:
    parts = []
    for g in result.groups:
        if g.mean is None:
            parts.append(f"{g.label}: n = {g.n}")
        else:
            parts.append(f"{g.label}: M = {num(g.mean)}, SD = {num(g.sd)}, "
                         f"n = {g.n}")
    return "; ".join(parts)


def apa_sentence(result: AnalysisResult) -> str:
    a = result.analysis
    effect = _effect_clause(result)
    groups = _group_clause(result)

    if a in ("student_t", "welch_t"):
        name = "Welch's t-test" if a == "welch_t" else "an independent samples t-test"
        return (f"{groups}. A comparison using {name} gave "
                f"t({df_str(result.df)}) = {num(result.statistic)}, "
                f"{pval(result.p)}, {effect}.")

    if a == "mann_whitney":
        return (f"{groups}. A Mann-Whitney U test gave U = "
                f"{num(result.statistic, 1)}, {pval(result.p)}, {effect}.")

    if a == "one_way_anova":
        return (f"{groups}. A one-way analysis of variance gave "
                f"F({df_str(result.df)}, {df_str(result.df2)}) = "
                f"{num(result.statistic)}, {pval(result.p)}, {effect}.")

    if a == "kruskal_wallis":
        return (f"{groups}. A Kruskal-Wallis test gave H({df_str(result.df)})"
                f" = {num(result.statistic)}, {pval(result.p)}, {effect}.")

    if a.endswith("correlation"):
        label = ("Spearman's rank correlation" if "spearman" in a
                 else "A Pearson correlation")
        return (f"{label} between {result.predictor.display} and "
                f"{result.outcome.display} was {effect}, "
                f"{pval(result.p)}, n = {result.n}.")

    if a == "chi_square":
        return (f"A chi-square test of independence between "
                f"{result.predictor.display} and {result.outcome.display} "
                f"gave chi2({df_str(result.df)}, N = {result.n}) = "
                f"{num(result.statistic)}, {pval(result.p)}, {effect}.")

    if a == "descriptives":
        return groups + "."

    return groups
