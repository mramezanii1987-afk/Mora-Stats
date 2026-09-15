"""Command line access to the engine, so it can be checked without a UI.

    python -m mora_engine profile data.csv
    python -m mora_engine run data.csv --test welch_t --vars score condition \\
        --design experimental --glossary
    python -m mora_engine suggest data.csv --vars score condition
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import List

from . import ANALYSES, dataio, report, run
from .models import Design, VarType, Variable
from .session import SessionLedger
from .suggest import suggest_tests


def _variables(frame, names: List[str], bounds: bool) -> List[Variable]:
    profile = {v.name: v for v in dataio.profile(frame, declare_bounds=bounds)}
    missing = [n for n in names if n not in profile]
    if missing:
        raise SystemExit(f"Column not found: {', '.join(missing)}. "
                         f"Available: {', '.join(profile)}")
    return [profile[n] for n in names]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="mora_engine")
    sub = parser.add_subparsers(dest="command", required=True)

    p_profile = sub.add_parser("profile", help="show the variable list")
    p_profile.add_argument("path")
    p_profile.add_argument("--header-rows", type=int, default=1)

    p_run = sub.add_parser("run", help="run one analysis")
    p_run.add_argument("path")
    p_run.add_argument("--test", required=True, choices=sorted(ANALYSES))
    p_run.add_argument("--vars", nargs="+", required=True)
    p_run.add_argument("--design", default="correlational",
                       choices=[d.value for d in Design])
    p_run.add_argument("--alpha", type=float, default=0.05)
    p_run.add_argument("--header-rows", type=int, default=1)
    p_run.add_argument("--bounds", action="store_true",
                       help="treat the observed range of ordinal variables "
                            "as their scale limits")
    p_run.add_argument("--glossary", action="store_true")
    p_run.add_argument("--json", action="store_true")

    p_suggest = sub.add_parser("suggest", help="list the tests that fit")
    p_suggest.add_argument("path")
    p_suggest.add_argument("--vars", nargs="+", required=True)
    p_suggest.add_argument("--header-rows", type=int, default=1)

    args = parser.parse_args(argv)
    frame = dataio.load(args.path, header_rows=args.header_rows)

    if args.command == "profile":
        variables = dataio.profile(frame)
        for row in dataio.variable_summary(frame, variables):
            spark = row.get("sparkline", "")
            print(f"{row['name']:<24} {row['type']:<11} "
                  f"missing {row['missing']:<5} {spark}")
        return 0

    if args.command == "suggest":
        variables = _variables(frame, args.vars, bounds=False)
        for s in suggest_tests(frame, variables):
            mark = "*" if s.recommended else (" " if s.eligible else "x")
            print(f"{mark} {s.label:<32} {s.reason}")
        return 0

    variables = _variables(frame, args.vars, bounds=args.bounds)
    ledger = SessionLedger(alpha=args.alpha)
    result, interpretation = run(frame, args.test, variables,
                                 Design(args.design), alpha=args.alpha,
                                 ledger=ledger)
    if args.json:
        print(json.dumps({"result": asdict(result),
                          "interpretation": asdict(interpretation)},
                         indent=2, default=str))
    else:
        print(report.render(result, interpretation, glossary=args.glossary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
