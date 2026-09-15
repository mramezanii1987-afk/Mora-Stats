"""The sidecar process. One JSON object per line, in and out.

The Tauri shell spawns this, writes a request line to stdin and reads a
response line from stdout. Choosing stdio over a local HTTP port avoids
firewall prompts, port collisions and the question of who else on the
machine can reach the statistics engine, which for a tool whose main promise
is that data stays put is the whole point.

Protocol:
    ->  {"id": 1, "command": "run", "params": {...}}
    <-  {"id": 1, "ok": true, "data": {...}}
    <-  {"id": 1, "ok": false, "error": {"message": "...", "action": "..."}}

Every error carries an action, because a dead end in the engine becomes a
dead end in the interface.
"""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import pandas as pd

from . import ANALYSES, __version__, charts, dataio, run
from .interpret import BOUNDARY_NOTE
from .models import Design, VarType, Variable
from .session import SessionLedger
from .suggest import assumption_report, suggest_tests

MAX_PREVIEW_ROWS = 200


class Sidecar:
    """Holds the open project. The imported frame is never written to."""

    def __init__(self) -> None:
        self.frame: Optional[pd.DataFrame] = None
        self.variables: Dict[str, Variable] = {}
        self.design: Design = Design.CORRELATIONAL
        self.ledger = SessionLedger()
        self.source: Optional[str] = None

    # ------------------------------------------------------------ commands
    def ping(self, **_: Any) -> Dict[str, Any]:
        return {"version": __version__, "analyses": ANALYSES,
                "boundary": BOUNDARY_NOTE}

    def new_project(self, design: str = "correlational",
                    alpha: float = 0.05, **_: Any) -> Dict[str, Any]:
        self.design = Design(design)
        self.ledger = SessionLedger(alpha=alpha)
        return {"design": self.design.value, "alpha": alpha}

    def open(self, path: str, header_rows: int = 1,
             declare_bounds: bool = True, **_: Any) -> Dict[str, Any]:
        frame = dataio.load(path, header_rows=header_rows)
        if frame.empty:
            raise EngineError("That file opened with no rows in it.",
                              "Check the header row count on import, then "
                              "open it again.")
        self.frame = frame
        self.source = path
        self.variables = {v.name: v for v in
                          dataio.profile(frame, declare_bounds=declare_bounds)}
        return self.dataset()

    def dataset(self, **_: Any) -> Dict[str, Any]:
        self._require_data()
        frame = self.frame
        variables = [self.variables[c] for c in frame.columns]
        preview = frame.head(MAX_PREVIEW_ROWS).where(pd.notna(frame), None)
        return {
            "source": self.source,
            "n_rows": int(len(frame)),
            "columns": [c for c in frame.columns],
            "variables": [asdict(v) for v in variables],
            "summary": dataio.variable_summary(frame, variables),
            "preview": preview.to_dict(orient="records"),
            "design": self.design.value,
            "truncated": len(frame) > MAX_PREVIEW_ROWS,
        }

    def set_variable(self, name: str, **changes: Any) -> Dict[str, Any]:
        self._require_data()
        if name not in self.variables:
            raise EngineError(f"There is no column called {name}.",
                              "Pick a column from the variable list.")
        current = self.variables[name]
        updated = Variable(
            name=current.name,
            type=VarType(changes.get("type", current.type.value)
                         if isinstance(changes.get("type", current.type), str)
                         else current.type),
            label=changes.get("label", current.label),
            unit=changes.get("unit", current.unit),
            lower_bound=changes.get("lower_bound", current.lower_bound),
            upper_bound=changes.get("upper_bound", current.upper_bound),
        )
        self.variables[name] = updated
        return {"variable": asdict(updated)}

    def suggest(self, variables: List[str], **_: Any) -> Dict[str, Any]:
        self._require_data()
        picked = [self._variable(n) for n in variables]
        suggestions = [asdict(s) for s in suggest_tests(self.frame, picked)]
        assumptions = {}
        if len(picked) == 2:
            outcome, group = picked[0], picked[1]
            if group.type is VarType.NOMINAL and outcome.type is not VarType.NOMINAL:
                assumptions = assumption_report(self.frame, outcome, group)
        return {"suggestions": suggestions, "assumptions": assumptions}

    def run(self, test: str, variables: List[str], alpha: float = 0.05,
            filter_key: str = "", **_: Any) -> Dict[str, Any]:
        self._require_data()
        picked = [self._variable(n) for n in variables]
        result, interpretation = run(self.frame, test, picked, self.design,
                                     alpha=alpha, ledger=self.ledger,
                                     filter_key=filter_key)
        return {
            "result": asdict(result),
            "interpretation": asdict(interpretation),
            "chart": charts.chart_for(result, self.frame),
            "ledger": self.ledger_state(),
        }

    def ledger_state(self, **_: Any) -> Dict[str, Any]:
        return {
            "project_count": self.ledger.project_count,
            "session_count": self.ledger.session_count,
            "family_wise_rate": self.ledger.family_wise_rate(),
            "holm": self.ledger.holm(),
            "benjamini_hochberg": self.ledger.benjamini_hochberg(),
        }

    # ------------------------------------------------------------ helpers
    def _require_data(self) -> None:
        if self.frame is None:
            raise EngineError("No dataset is open yet.",
                              "Open a .csv or .xlsx file to get started.")

    def _variable(self, name: str) -> Variable:
        self._require_data()
        if name not in self.variables:
            raise EngineError(f"There is no column called {name}.",
                              "Pick a column from the variable list.")
        return self.variables[name]

    COMMANDS = ("ping", "new_project", "open", "dataset", "set_variable",
                "suggest", "run", "ledger_state")

    def handle(self, message: Dict[str, Any]) -> Dict[str, Any]:
        request_id = message.get("id")
        command = message.get("command", "")
        params = message.get("params") or {}
        if command not in self.COMMANDS:
            return _error(request_id, f"Unknown command '{command}'.",
                          f"Use one of: {', '.join(self.COMMANDS)}.")
        try:
            data = getattr(self, command)(**params)
            return {"id": request_id, "ok": True, "data": data}
        except EngineError as exc:
            return _error(request_id, exc.message, exc.action)
        except (ValueError, KeyError) as exc:
            return _error(request_id, str(exc),
                          "Change the variables or the test and try again.")
        except Exception as exc:  # pragma: no cover - last resort
            return _error(request_id, f"The engine stopped: {exc}",
                          "This is a bug. The details below belong in an "
                          "issue report.",
                          trace=traceback.format_exc())


class EngineError(Exception):
    def __init__(self, message: str, action: str) -> None:
        super().__init__(message)
        self.message = message
        self.action = action


def _error(request_id: Any, message: str, action: str,
           trace: str = "") -> Dict[str, Any]:
    payload = {"message": message, "action": action}
    if trace:
        payload["trace"] = trace
    return {"id": request_id, "ok": False, "error": payload}


def serve(stdin=None, stdout=None) -> None:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    sidecar = Sidecar()
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            response = _error(None, "That request was not valid JSON.",
                              "Send one JSON object per line.")
        else:
            response = sidecar.handle(message)
        stdout.write(json.dumps(response, default=str) + "\n")
        stdout.flush()


if __name__ == "__main__":
    serve()
