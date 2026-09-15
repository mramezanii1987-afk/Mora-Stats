"""Number and text formatting, APA 7th edition conventions."""

from __future__ import annotations

from typing import Optional


def num(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "NA"
    try:
        if x != x:  # NaN
            return "NA"
    except TypeError:
        return "NA"
    return f"{x:.{digits}f}"


def lead(x: Optional[float], digits: int = 2) -> str:
    """Statistics bounded by one lose the leading zero under APA."""
    if x is None:
        return "NA"
    s = f"{abs(x):.{digits}f}"
    if s.startswith("0."):
        s = s[1:]
    return ("-" if x < 0 else "") + s


def pval(p: Optional[float]) -> str:
    if p is None:
        return "NA"
    if p < 0.001:
        return "p < .001"
    return f"p = {lead(p, 3)}"


def pval_bare(p: Optional[float]) -> str:
    if p is None:
        return "NA"
    if p < 0.001:
        return "< .001"
    return lead(p, 3)


def df_str(df: Optional[float]) -> str:
    if df is None:
        return "NA"
    if abs(df - round(df)) < 1e-9:
        return str(int(round(df)))
    return f"{df:.2f}"


def ci(low: Optional[float], high: Optional[float], digits: int = 2,
       drop_zero: bool = False) -> str:
    f = lead if drop_zero else num
    return f"[{f(low, digits)}, {f(high, digits)}]"


def pct(x: Optional[float], digits: int = 1) -> str:
    if x is None:
        return "NA"
    return f"{100 * x:.{digits}f}%"


def units(value: float, unit: str, digits: int = 2) -> str:
    return f"{num(value, digits)} {unit}"


def plural(n: int, singular: str, plural_form: Optional[str] = None) -> str:
    if n == 1:
        return f"{n} {singular}"
    return f"{n} {plural_form or singular + 's'}"


def join_list(items) -> str:
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def article(word: str) -> str:
    """a or an, decided by how the word is read aloud rather than spelled.

    Statistical labels are read as letters, so "an r" and "an F" are right
    while "a Cramer's V" is not, and a spelling rule alone gets these wrong.
    """
    if not word:
        return "a"
    spoken_vowel = {"f", "h", "l", "m", "n", "r", "s", "x"}
    first = word[0]
    if len(word) == 1 or (word[:2].isupper() and len(word) <= 3):
        return "an" if first.lower() in spoken_vowel or first.lower() in "aeiou" else "a"
    return "an" if first.lower() in "aeiou" else "a"
