"""The interpretation engine.

This module is the product. It is a pure function: an AnalysisResult goes in,
an Interpretation comes out. No model, no network, no randomness, no state
carried between calls. The same numbers produce the same words every time,
and every sentence is traceable to a value the engine computed, which is
recorded in ``Interpretation.provenance``.

The rules, in the order they are applied:

1. Lead with magnitude in the units of the measure, then in standardised
   terms with its interval.
2. Match the language to the declared design. A correlational design never
   gets causal verbs, and the reason is stated in the same sentence.
3. A non-significant result gets the minimum detectable effect and an
   explicit statement that this is not evidence of no effect.
4. A significant result from an underpowered design gets an exaggeration
   factor rather than a post hoc power figure.
5. Threats are listed separately, each with the action that answers it.
6. From the third inferential test onward, the multiple comparisons count is
   attached and cannot be dismissed.

The engine interprets the statistic and nothing else. It never says a
finding is important, novel, surprising or supportive of a theory.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .apa import apa_sentence
from .effects import interpret_magnitude
from .fmt import (article, ci, df_str, join_list, lead, num, pct, pval,
                  pval_bare)
from .models import AnalysisResult, Design, Interpretation, Threat
from .session import SessionLedger
from .tables import tables_for

BOUNDARY_NOTE = (
    "This engine reads the statistic only. Whether the finding matters, "
    "whether it is new, and whether it supports your theory are judgements "
    "for you and your reviewers, and no rule in here can make them."
)

CAUSAL_VERBS = ["causes", "caused", "leads to", "led to", "affects",
                "affected", "predicts", "predicted", "produces", "produced",
                "results in", "resulted in", "drives", "driven by",
                "influences", "influenced", "makes", "improves", "improved",
                "reduces", "reduced", "increases", "increased",
                "due to", "because of", "effect of"]

VALUE_WORDS = ["important", "unimportant", "novel", "exciting", "striking",
               "remarkable", "impressive", "meaningful finding", "supports the theory",
               "confirms the hypothesis", "proves", "proven", "disproves",
               "trivial finding", "significant contribution"]

BOUNDED = {"Hedges' g", "Pearson's r", "Spearman's rho", "rank-biserial r",
           "Cramer's V", "omega squared", "epsilon squared", "r", "rho",
           "g", "V"}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _fmt_effect(name: str, value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return lead(value, 2) if name in BOUNDED else num(value, 2)


def _effect_with_ci(result: AnalysisResult) -> str:
    e = result.effect
    if e is None:
        return ""
    drop = e.name in BOUNDED
    text = f"{e.name} = {_fmt_effect(e.name, e.value)}"
    if e.ci_low is not None and e.ci_high is not None:
        text += f", 95% CI {ci(e.ci_low, e.ci_high, 2, drop_zero=drop)}"
    return text


def _magnitude_word(result: AnalysisResult) -> str:
    e = result.effect
    if e is None:
        return ""
    kind = {
        "Hedges' g": "g", "Pearson's r": "r", "Spearman's rho": "rho",
        "rank-biserial r": "rank_biserial", "omega squared": "omega_squared",
        "epsilon squared": "epsilon_squared", "Cramer's V": "cramers_v",
    }.get(e.name, "")
    return interpret_magnitude(e.value, kind) if kind else ""


def _design_sentence(result: AnalysisResult) -> str:
    predictor = result.predictor.display if result.predictor else "the grouping"
    outcome = result.outcome.display if result.outcome else "the outcome"
    if result.design is Design.CORRELATIONAL:
        return (f"This project is declared correlational, so the finding is "
                f"that {predictor} is associated with {outcome}, and nothing "
                f"here separates that association from the other ways the "
                f"cases differ.")
    if result.design is Design.QUASI_EXPERIMENTAL:
        return (f"This project is declared quasi-experimental, so the groups "
                f"were compared as they already existed, and the gap carries "
                f"whatever else distinguishes them alongside any effect of "
                f"{predictor} itself.")
    return (f"This project is declared experimental with random assignment, "
            f"so within this sample the gap can be read as an effect of "
            f"{predictor}, because assignment is what made the groups "
            f"comparable to begin with.")


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _interval_phrase(low, high, unit: str, direction: str) -> str:
    """State the interval in the direction the sentence already used."""
    if low is None or high is None:
        return "the interval around it is not available"
    lo, hi = sorted((abs(low), abs(high)))
    if (low >= 0) == (high >= 0):
        return (f"the gap is somewhere between {num(lo)} and {num(hi)} "
                f"{unit} at 95% confidence")
    smaller, larger = (abs(low), abs(high)) if direction == "higher" else \
        (abs(high), abs(low))
    return (f"the interval still runs from {num(smaller)} {unit} the other "
            f"way to {num(larger)} {unit} {direction}, so the direction "
            f"itself is unsettled")


def _mde_core(result: AnalysisResult) -> str:
    p = result.power
    if p.mde is None:
        return ""
    label = p.mde_label or "effect"
    return f"{label} of {_fmt_effect(label, p.mde)}"


def _mde_raw_clause(result: AnalysisResult) -> str:
    p = result.power
    if p.mde_raw is None or result.outcome is None:
        return ""
    return f" (about {num(p.mde_raw)} {result.outcome.units})"


def _significance_sentence(result: AnalysisResult) -> str:
    p = result.power
    hits = int(p.target_power * 100)
    if not result.significant:
        base = (f"At {pval(result.p)} this does not clear the "
                f"{result.alpha:g} line, and that is not evidence that there "
                f"is no effect")
        if p.mde is not None:
            core = _mde_core(result)
            base += (f", because at n = {result.n} the smallest effect this "
                     f"design could catch {hits} times out of 100 was "
                     f"{article(core)} {core}{_mde_raw_clause(result)}, so "
                     f"anything smaller than that would most likely have "
                     f"slipped past")
            if p.approximate:
                base += " (an approximation, since rank tests have no exact "\
                        "power formula)"
        return base + "."
    base = f"At {pval(result.p)} the result clears the {result.alpha:g} line"
    if p.mde is not None:
        core = _mde_core(result)
        base += (f", and at n = {result.n} the design could catch "
                 f"{article(core)} {core} or larger {hits} times out of 100"
                 f"{_mde_raw_clause(result)}")
    return base + "."


def _exaggeration_sentence(result: AnalysisResult) -> Optional[str]:
    p = result.power
    e = result.effect
    if not result.significant or e is None or p.exaggeration is None:
        return None
    underpowered = p.mde is not None and abs(e.value) < abs(p.mde)
    if not (underpowered or p.exaggeration > 1.15):
        return None
    return (f"Take care with the size of it. If the true effect were the size "
            f"estimated here, then among studies this small the ones that "
            f"reach significance overstate it by about "
            f"{num(p.exaggeration, 1)} times on average, so read "
            f"{_fmt_effect(e.name, e.value)} as the top of the plausible "
            f"range rather than the best guess, and lean on the interval "
            f"instead.")


# --------------------------------------------------------------------------
# magnitude sentences, one per analysis family
# --------------------------------------------------------------------------

def _magnitude_sentences(result: AnalysisResult) -> List[str]:
    a = result.analysis
    raw = result.raw_effect
    e = result.effect
    unit = result.outcome.units if result.outcome else "points"
    word = _magnitude_word(result)
    out: List[str] = []

    if a in ("student_t", "welch_t"):
        g1, g2 = result.groups[0], result.groups[1]
        direction = "higher" if raw.value >= 0 else "lower"
        out.append(
            f"{_cap(g1.label)} scored {num(abs(raw.value))} {unit} "
            f"{direction} than {g2.label} on average ({num(g1.mean)} against "
            f"{num(g2.mean)}), and "
            f"{_interval_phrase(raw.ci_low, raw.ci_high, unit, direction)}.")
        out.append(
            f"Measured against how much people vary, that comes to "
            f"{_effect_with_ci(result)}, a {word} difference by the usual "
            f"benchmarks.")

    elif a == "mann_whitney":
        g1, g2 = result.groups[0], result.groups[1]
        ps = result.extra.get("prob_superiority", 0.5)
        out.append(
            f"A case picked at random from {g1.label} scores higher than one "
            f"picked from {g2.label} about {pct(ps, 0)} of the time, and the "
            f"medians sit {num(abs(raw.value))} {unit} apart "
            f"({num(g1.median)} against {num(g2.median)}).")
        out.append(
            f"As a standardised figure that is {_effect_with_ci(result)}, a "
            f"{word} separation between the two distributions.")

    elif a == "one_way_anova":
        means = result.extra["means"]
        high, low = result.extra["highest"], result.extra["lowest"]
        listed = join_list([f"{k} at {num(v)}" for k, v in means.items()])
        out.append(
            f"The group means run {listed} {unit}, so {high} sits "
            f"{num(abs(raw.value))} {unit} above {low}.")
        share = result.effect.value if result.effect else 0.0
        out.append(
            f"Group membership accounts for {pct(share, 0)} of the variation "
            f"in {result.outcome.display} ({_effect_with_ci(result)}), a "
            f"{word} share.")

    elif a == "kruskal_wallis":
        medians = result.extra["medians"]
        high, low = result.extra["highest"], result.extra["lowest"]
        listed = join_list([f"{k} at {num(v)}" for k, v in medians.items()])
        out.append(
            f"The group medians run {listed} {unit}, with {high} sitting "
            f"{num(abs(raw.value))} {unit} above {low}.")
        share = result.effect.value if result.effect else 0.0
        out.append(
            f"Across the whole comparison the grouping accounts for "
            f"{pct(share, 0)} of the variation in the ranks "
            f"({_effect_with_ci(result)}), a {word} share.")

    elif a.endswith("correlation"):
        x = result.predictor.display
        y = result.outcome.display
        r2 = result.extra.get("r_squared", 0.0)
        direction = "rises" if raw.value >= 0 else "falls"
        out.append(
            f"Each one-point step up in {x} goes with {y} that {direction} by "
            f"{num(abs(raw.value))} {unit} on average, across "
            f"{result.n} cases.")
        out.append(
            f"The standardised figure is {_effect_with_ci(result)}, a {word} "
            f"association, and the two variables move together enough to "
            f"share {pct(r2, 0)} of their variation.")

    elif a == "chi_square":
        cell = result.extra.get("largest_cell", "the largest cell")
        obs = result.extra.get("observed_share", 0.0)
        exp = result.extra.get("expected_share", 0.0)
        out.append(
            f"The widest gap in the table is {cell}, which turns up in "
            f"{pct(obs, 0)} of those cases against the {pct(exp, 0)} that "
            f"independence would give, a gap of {num(abs(raw.value), 1)} "
            f"percentage points.")
        out.append(
            f"Taking the table as a whole, {_effect_with_ci(result)}, a "
            f"{word} association between "
            f"{result.predictor.display} and {result.outcome.display}.")

    return out


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------

def interpret(result: AnalysisResult,
              ledger: Optional[SessionLedger] = None,
              filter_key: str = "") -> Interpretation:
    """Pure function. Results object in, interpretation object out."""

    if not result.inferential:
        return _interpret_descriptives(result)

    sentences: List[str] = []
    sentences += _magnitude_sentences(result)
    sentences.append(_design_sentence(result))
    sentences.append(_significance_sentence(result))
    extra = _exaggeration_sentence(result)
    if extra:
        sentences.append(extra)
    sentences = [s for s in sentences if s][:5]

    multiplicity = None
    if ledger is not None:
        key = ledger.record(result, filter_key)
        multiplicity = ledger.statement(key)

    provenance: Dict[str, Any] = {
        "analysis": result.analysis,
        "statistic": result.statistic,
        "df": result.df,
        "df2": result.df2,
        "p": result.p,
        "n": result.n,
        "n_missing": result.n_missing,
        "effect": None if result.effect is None else result.effect.to_dict(),
        "raw_effect": None if result.raw_effect is None else result.raw_effect.to_dict(),
        "mde": result.power.mde,
        "mde_raw": result.power.mde_raw,
        "observed_power": result.power.observed_power,
        "exaggeration": result.power.exaggeration,
        "design": result.design.value,
        "threat_codes": [t.code for t in result.threats],
    }

    return Interpretation(
        result_rows=_result_rows(result),
        tables=tables_for(result),
        apa=apa_sentence(result),
        sentences=sentences,
        threats=list(result.threats),
        multiplicity=multiplicity,
        provenance=provenance,
    )


def _interpret_descriptives(result: AnalysisResult) -> Interpretation:
    rows = _result_rows(result)
    n_vars = len(result.variables)
    missing = result.n_missing
    sentences = [
        f"This is a description of {n_vars} "
        f"{'variable' if n_vars == 1 else 'variables'} across "
        f"{result.n} rows, with no test performed and nothing inferred.",
        (f"Across those variables {missing} values are missing, which the "
         f"analyses below will drop case by case."
         if missing else
         "No values are missing in these variables."),
    ]
    return Interpretation(result_rows=rows, tables=tables_for(result),
                          apa=apa_sentence(result), sentences=sentences,
                          threats=list(result.threats),
                          provenance={"analysis": "descriptives"})


def _result_rows(result: AnalysisResult) -> List[str]:
    rows: List[str] = []
    if result.statistic is not None:
        dfs = ""
        if result.df is not None and result.df2 is not None:
            dfs = f"({df_str(result.df)}, {df_str(result.df2)})"
        elif result.df is not None:
            dfs = f"({df_str(result.df)})"
        rows.append(f"{result.statistic_name}{dfs} = {num(result.statistic)}")
    if result.p is not None:
        rows.append(pval(result.p))
    if result.effect is not None:
        rows.append(_effect_with_ci(result))
    if result.raw_effect is not None:
        rows.append(f"{result.raw_effect.name} = {num(result.raw_effect.value)}")
    rows.append(f"n = {result.n}" + (f", {result.n_missing} dropped as missing"
                                     if result.n_missing else ""))
    return rows


# --------------------------------------------------------------------------
# language guard, used by the test suite
# --------------------------------------------------------------------------

def language_violations(interpretation: Interpretation,
                        design: Design) -> List[str]:
    """Words the engine is not allowed to use, given the design.

    The test suite runs this over every generated interpretation in the
    fixture set, so a careless template edit fails the build rather than
    reaching a user.
    """
    text = " ".join(interpretation.sentences).lower()
    problems = [w for w in VALUE_WORDS if w in text]
    if design is Design.CORRELATIONAL:
        for verb in CAUSAL_VERBS:
            if verb in text:
                problems.append(verb)
    return problems
