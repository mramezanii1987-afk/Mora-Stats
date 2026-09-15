"""Core data structures for MoRa Stats.

Everything the engine produces is a plain dataclass so that results can be
serialised, diffed, stored in a project file and unit tested without any
reference to a UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


class Design(str, Enum):
    """Study design declared when the project is created.

    The design controls the causal language the interpretation engine is
    allowed to use. It is deliberately not inferable from the data.
    """

    EXPERIMENTAL = "experimental"
    QUASI_EXPERIMENTAL = "quasi_experimental"
    CORRELATIONAL = "correlational"


class VarType(str, Enum):
    CONTINUOUS = "continuous"
    ORDINAL = "ordinal"
    NOMINAL = "nominal"


@dataclass(frozen=True)
class Variable:
    """Metadata for one column.

    ``lower_bound`` and ``upper_bound`` are the declared limits of the
    measurement scale, not the observed minimum and maximum. They are what
    makes floor and ceiling detection trustworthy, and they are optional.
    """

    name: str
    type: VarType
    label: Optional[str] = None
    unit: Optional[str] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

    @property
    def display(self) -> str:
        return self.label or self.name.replace("_", " ")

    @property
    def units(self) -> str:
        return self.unit or "points"


@dataclass
class Estimate:
    """A point estimate with an interval and the method used to build it."""

    name: str
    value: float
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    ci_level: float = 0.95
    method: str = ""
    approximate: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GroupDescriptives:
    label: str
    n: int
    mean: Optional[float] = None
    sd: Optional[float] = None
    median: Optional[float] = None
    iqr: Optional[float] = None
    missing: int = 0


@dataclass
class Threat:
    """A concrete, computed threat to this particular result.

    ``action`` is mandatory. The app never states a problem without stating
    what the user can do about it.
    """

    code: str
    severity: str  # "note" | "warn" | "serious"
    message: str
    action: str
    values: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PowerSummary:
    alpha: float = 0.05
    target_power: float = 0.80
    mde: Optional[float] = None            # minimum detectable effect, standardised
    mde_label: str = ""                    # e.g. "Hedges' g"
    mde_raw: Optional[float] = None        # same thing in the units of the measure
    observed_power: Optional[float] = None
    exaggeration: Optional[float] = None   # Type M ratio at the observed effect
    approximate: bool = False
    note: str = ""


@dataclass
class AnalysisResult:
    """Everything the statistical layer computed. No prose lives here."""

    analysis: str                  # machine key, e.g. "welch_t"
    label: str                     # human name, e.g. "Welch's t-test"
    design: Design
    statistic_name: str = ""
    statistic: Optional[float] = None
    df: Optional[float] = None
    df2: Optional[float] = None
    p: Optional[float] = None
    alpha: float = 0.05
    n: int = 0
    n_missing: int = 0
    effect: Optional[Estimate] = None
    raw_effect: Optional[Estimate] = None
    groups: List[GroupDescriptives] = field(default_factory=list)
    variables: List[str] = field(default_factory=list)
    outcome: Optional[Variable] = None
    predictor: Optional[Variable] = None
    power: PowerSummary = field(default_factory=PowerSummary)
    threats: List[Threat] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)
    inferential: bool = True

    @property
    def significant(self) -> bool:
        return self.p is not None and self.p < self.alpha

    def key(self, filter_key: str = "") -> str:
        """Identity of the question being asked, for the test counter.

        Changing an option that does not change the question (Student to
        Welch on the same variables, a display toggle) must not increment
        the multiple comparisons count, so the key ignores the test family
        variant and keys on the variables and the row filter.
        """
        family = self.extra.get("family", self.analysis)
        return f"{family}|{'+'.join(sorted(self.variables))}|{filter_key}"


@dataclass
class Interpretation:
    """The three blocks, as structured data. Rendering happens elsewhere."""

    result_rows: List[str] = field(default_factory=list)
    tables: List[Any] = field(default_factory=list)
    apa: str = ""
    sentences: List[str] = field(default_factory=list)
    threats: List[Threat] = field(default_factory=list)
    multiplicity: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def plain_text(self) -> str:
        return " ".join(self.sentences)


@dataclass
class Suggestion:
    test: str
    label: str
    eligible: bool
    reason: str = ""
    recommended: bool = False
