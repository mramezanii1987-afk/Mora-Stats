"""Rendering. The three blocks always appear in the same order.

A glossary can be switched on for readers who are meeting these terms for
the first time. It explains the words the interpretation uses, and it never
changes the interpretation itself.
"""

from __future__ import annotations

from typing import Optional

from .fmt import num
from .models import AnalysisResult, Interpretation
from .tables import render_text

GLOSSARY = {
    "p value": ("How often numbers at least this extreme would turn up if "
                "there were nothing to find. A small p makes an accident a "
                "poor explanation for the pattern. It is not the chance that "
                "you are wrong."),
    "confidence interval": ("The range of values that fit the data "
                           "reasonably well. Read the whole range rather "
                           "than the single number in front of it, because "
                           "the range is what tells you how much is still "
                           "unsettled."),
    "effect size": ("How big the difference or the association is, put on a "
                    "scale that does not depend on how many people you "
                    "measured. A p value tells you whether something is "
                    "there, and an effect size tells you how much of it "
                    "there is."),
    "power": ("The chance that a study of this size would notice an effect "
              "of a given size if that effect were real. Low power is why a "
              "small study can miss something that is genuinely there."),
}


def render(result: AnalysisResult, interpretation: Interpretation,
           glossary: bool = False, width: int = 78) -> str:
    lines = []
    rule = "-" * width

    lines.append(rule)
    lines.append(f"{result.label.upper()}  ·  {' × '.join(result.variables)}"
                 f"  ·  design: {result.design.value.replace('_', '-')}")
    lines.append(rule)

    lines.append("")
    lines.append("RESULT")
    for table in interpretation.tables:
        lines.append("")
        for row in render_text(table, width).split("\n"):
            lines.append(f"  {row}")

    lines.append("")
    lines.append("APA")
    lines.append(_wrap(interpretation.apa, width, indent="  "))

    lines.append("")
    lines.append("INTERPRETATION")
    lines.append(_wrap(interpretation.plain_text, width, indent="  "))

    if interpretation.threats:
        lines.append("")
        lines.append("CHECKS")
        for t in interpretation.threats:
            mark = {"note": "·", "warn": "!", "serious": "!!"}.get(t.severity, "·")
            lines.append(_wrap(f"{mark} {t.message}", width, indent="  "))
            lines.append(_wrap(f"What to do: {t.action}", width, indent="    "))

    if interpretation.multiplicity:
        lines.append("")
        lines.append("MULTIPLE COMPARISONS")
        lines.append(_wrap(interpretation.multiplicity, width, indent="  "))

    if glossary:
        lines.append("")
        lines.append("WORDS USED HERE")
        for term, text in GLOSSARY.items():
            lines.append(_wrap(f"{term}: {text}", width, indent="  "))

    lines.append("")
    return "\n".join(lines)


def _wrap(text: str, width: int, indent: str = "") -> str:
    import textwrap
    return "\n".join(textwrap.wrap(text, width=width, initial_indent=indent,
                                   subsequent_indent=indent))
