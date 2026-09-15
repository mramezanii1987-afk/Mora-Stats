"""The multiple comparisons ledger.

Counting rules, which matter as much as the arithmetic:

* Only inferential tests count. Descriptives never do.
* An analysis is identified by the question it asks, which is the family of
  test plus the variables plus the row filter. Switching from Student to
  Welch on the same variables is the same question asked two ways and counts
  once. Changing a display option counts for nothing.
* The count belongs to the project rather than to the window, so quitting
  the app does not reset it. The ledger reports both figures and computes
  the family-wise rate from the project count.

From the third distinct test onward this is stated on every result, and it
cannot be turned off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .fmt import pct, pval_bare
from .models import AnalysisResult


@dataclass
class LedgerEntry:
    key: str
    label: str
    variables: List[str]
    p: Optional[float]
    session: int


@dataclass
class SessionLedger:
    alpha: float = 0.05
    session_id: int = 1
    entries: Dict[str, LedgerEntry] = field(default_factory=dict)

    # ----------------------------------------------------------------
    def record(self, result: AnalysisResult, filter_key: str = "") -> str:
        if not result.inferential:
            return ""
        key = result.key(filter_key)
        self.entries[key] = LedgerEntry(
            key=key,
            label=result.label,
            variables=list(result.variables),
            p=result.p,
            session=self.session_id,
        )
        return key

    @property
    def project_count(self) -> int:
        return len(self.entries)

    @property
    def session_count(self) -> int:
        return sum(1 for e in self.entries.values()
                   if e.session == self.session_id)

    def family_wise_rate(self, count: Optional[int] = None) -> float:
        k = self.project_count if count is None else count
        if k <= 0:
            return 0.0
        return float(1.0 - (1.0 - self.alpha) ** k)

    # ----------------------------------------------------------------
    def p_values(self) -> List[Tuple[str, float]]:
        return [(k, e.p) for k, e in self.entries.items() if e.p is not None]

    def holm(self) -> Dict[str, float]:
        items = self.p_values()
        if not items:
            return {}
        order = sorted(items, key=lambda kv: kv[1])
        m = len(order)
        adjusted: Dict[str, float] = {}
        running = 0.0
        for i, (key, p) in enumerate(order):
            value = min(1.0, (m - i) * p)
            running = max(running, value)
            adjusted[key] = running
        return adjusted

    def benjamini_hochberg(self) -> Dict[str, float]:
        items = self.p_values()
        if not items:
            return {}
        order = sorted(items, key=lambda kv: kv[1])
        m = len(order)
        adjusted: Dict[str, float] = {}
        running = 1.0
        for i in range(m - 1, -1, -1):
            key, p = order[i]
            value = min(1.0, p * m / (i + 1))
            running = min(running, value)
            adjusted[key] = running
        return adjusted

    # ----------------------------------------------------------------
    def statement(self, current_key: str = "") -> Optional[str]:
        """The sentence attached to a result once three tests have been run."""
        k = self.project_count
        if k < 3:
            return None
        rate = self.family_wise_rate()
        holm = self.holm()
        bh = self.benjamini_hochberg()
        text = (f"This is inferential test {k} in this project "
                f"({self.session_count} in this session). Running {k} tests "
                f"at alpha = {self.alpha:g} gives about a {pct(rate, 0)} "
                "chance that at least one of them is a false positive even "
                "if nothing is going on.")
        if current_key and current_key in holm:
            text += (f" Adjusted for the whole set, this result has a Holm "
                     f"p of {pval_bare(holm[current_key])} and a "
                     f"Benjamini-Hochberg p of "
                     f"{pval_bare(bh[current_key])}.")
        text += (" Decide which of these tests belong to one family and apply "
                 "Holm for strict control or Benjamini-Hochberg for a "
                 "tolerable false discovery rate.")
        return text
