"""Run the engine over a set of known datasets and print the three blocks.

This is the thing to read before any UI exists. Each case is chosen to make
one rule visible: the published t-test, a null result at small n, a
significant result from an underpowered design, a ceiling effect, a
contingency table with thin cells, and the running test count.

    python demo/run_demo.py            print everything
    python demo/run_demo.py --json out.json   also write the structured form
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mora_engine as me  # noqa: E402
from mora_engine.models import Design, VarType, Variable  # noqa: E402
from mora_engine.session import SessionLedger  # noqa: E402


def sleep_trial() -> tuple:
    """Student's sleep data. Published: t = -1.8608, df = 17.776, p = .0794."""
    frame = pd.DataFrame({
        "extra_sleep": [0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0,
                        1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4],
        "drug": ["drug 1"] * 10 + ["drug 2"] * 10,
    })
    return ("Student's sleep data, the dataset the t-test was published on",
            frame, "welch_t",
            [Variable("extra_sleep", VarType.CONTINUOUS, unit="hours"),
             Variable("drug", VarType.NOMINAL)],
            Design.EXPERIMENTAL)


def null_at_small_n() -> tuple:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({
        "wellbeing": np.r_[rng.normal(50, 10, 18), rng.normal(50, 10, 18)],
        "condition": ["waitlist"] * 18 + ["programme"] * 18,
    })
    return ("A null result at n = 36, where SPSS would stop at 'not "
            "significant'",
            frame, "welch_t",
            [Variable("wellbeing", VarType.CONTINUOUS, unit="points"),
             Variable("condition", VarType.NOMINAL)],
            Design.EXPERIMENTAL)


def underpowered_win() -> tuple:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({
        "recall": np.r_[rng.normal(50, 10, 11), rng.normal(56, 10, 11)],
        "method": ["massed"] * 11 + ["spaced"] * 11,
    })
    return ("A significant result from a small design, where the estimate is "
            "inflated",
            frame, "welch_t",
            [Variable("recall", VarType.CONTINUOUS, unit="items"),
             Variable("method", VarType.NOMINAL)],
            Design.EXPERIMENTAL)


def ceiling_survey() -> tuple:
    rng = np.random.default_rng(19)
    a = np.clip(np.round(rng.normal(6.3, 1.0, 60)), 1, 7)
    b = np.clip(np.round(rng.normal(5.6, 1.2, 60)), 1, 7)
    frame = pd.DataFrame({
        "satisfaction": np.r_[a, b],
        "plan": ["premium"] * 60 + ["standard"] * 60,
    })
    return ("A seven point satisfaction scale with a ceiling, in a "
            "correlational project",
            frame, "mann_whitney",
            [Variable("satisfaction", VarType.ORDINAL, unit="scale points",
                      lower_bound=1, upper_bound=7),
             Variable("plan", VarType.NOMINAL)],
            Design.CORRELATIONAL)


def three_sites() -> tuple:
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({
        "turnaround": np.r_[rng.normal(50, 9, 40), rng.normal(55, 9, 40),
                            rng.normal(58, 9, 40)],
        "site": ["north"] * 40 + ["south"] * 40 + ["east"] * 40,
    })
    return ("Three sites compared on one outcome",
            frame, "one_way_anova",
            [Variable("turnaround", VarType.CONTINUOUS, unit="minutes"),
             Variable("site", VarType.NOMINAL)],
            Design.QUASI_EXPERIMENTAL)


def study_hours() -> tuple:
    rng = np.random.default_rng(17)
    hours = rng.normal(10, 2, 90)
    frame = pd.DataFrame({
        "study_hours": hours,
        "grade": 3.0 + 0.8 * hours + rng.normal(0, 1.5, 90),
    })
    return ("Two continuous variables in a correlational project",
            frame, "pearson_correlation",
            [Variable("study_hours", VarType.CONTINUOUS, unit="hours"),
             Variable("grade", VarType.CONTINUOUS, unit="marks")],
            Design.CORRELATIONAL)


def thin_table() -> tuple:
    frame = pd.DataFrame({
        "attended": ["yes"] * 18 + ["no"] * 14,
        "outcome": (["passed"] * 15 + ["failed"] * 3 +
                    ["passed"] * 8 + ["failed"] * 6),
    })
    return ("A contingency table with cells too thin for chi-square",
            frame, "chi_square",
            [Variable("attended", VarType.NOMINAL),
             Variable("outcome", VarType.NOMINAL)],
            Design.CORRELATIONAL)


CASES = [sleep_trial, null_at_small_n, underpowered_win, ceiling_survey,
         three_sites, study_hours, thin_table]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--glossary", action="store_true")
    args = parser.parse_args()

    ledger = SessionLedger()
    payload = []

    print("=" * 78)
    print("MoRa Stats engine, version " + me.__version__)
    print("Every number below is computed. Every sentence is a template "
          "filled from")
    print("those numbers. There is no model anywhere in this path.")
    print("=" * 78)

    for case in CASES:
        title, frame, test, variables, design = case()
        print()
        print(f"### {title}")
        result, interpretation = me.run(frame, test, variables, design,
                                        ledger=ledger)
        print(me.report.render(result, interpretation,
                               glossary=args.glossary))
        payload.append({
            "title": title,
            "result": asdict(result),
            "interpretation": asdict(interpretation),
        })

    print("=" * 78)
    print(f"Tests recorded this session: {ledger.session_count}. "
          f"Family-wise false positive rate at alpha = 0.05: "
          f"{ledger.family_wise_rate() * 100:.0f}%.")
    print(me.BOUNDARY_NOTE)
    print("=" * 78)

    if args.json:
        args.json.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nStructured output written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
