"""The protocol the desktop shell depends on."""

import io
import json

import numpy as np
import pandas as pd
import pytest

from mora_engine.sidecar import Sidecar, serve


@pytest.fixture
def csv_path(tmp_path):
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({
        "score": np.r_[rng.normal(50, 9, 45), rng.normal(56, 9, 45)],
        "arm": ["control"] * 45 + ["treatment"] * 45,
        "mood": rng.integers(1, 8, 90),
    })
    path = tmp_path / "study.csv"
    frame.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def sidecar(csv_path):
    s = Sidecar()
    s.handle({"id": 0, "command": "new_project",
              "params": {"design": "experimental"}})
    s.handle({"id": 1, "command": "open", "params": {"path": csv_path}})
    return s


def call(sidecar, command, **params):
    return sidecar.handle({"id": 9, "command": command, "params": params})


def test_open_detects_types(sidecar):
    data = call(sidecar, "dataset")["data"]
    types = {v["name"]: v["type"] for v in data["variables"]}
    assert types == {"score": "continuous", "arm": "nominal",
                     "mood": "ordinal"}
    assert data["n_rows"] == 90


def test_ordinal_bounds_are_offered_on_import(sidecar):
    data = call(sidecar, "dataset")["data"]
    mood = next(v for v in data["variables"] if v["name"] == "mood")
    assert mood["lower_bound"] is not None
    assert mood["upper_bound"] is not None


def test_a_type_can_be_corrected_inline(sidecar):
    response = call(sidecar, "set_variable", name="mood", type="continuous")
    assert response["data"]["variable"]["type"] == "continuous"
    data = call(sidecar, "dataset")["data"]
    types = {v["name"]: v["type"] for v in data["variables"]}
    assert types["mood"] == "continuous"


def test_run_returns_all_three_blocks_and_a_chart(sidecar):
    payload = call(sidecar, "run", test="welch_t",
                   variables=["score", "arm"])["data"]
    assert payload["interpretation"]["apa"]
    assert len(payload["interpretation"]["sentences"]) >= 3
    assert payload["result"]["effect"]["ci_low"] is not None
    assert payload["chart"]["kind"] == "groups"
    assert len(payload["chart"]["groups"]) == 2
    assert payload["chart"]["default_view"] in payload["chart"]["views"]


def test_every_analysis_produces_a_chart(sidecar):
    for test, variables in (("welch_t", ["score", "arm"]),
                            ("mann_whitney", ["score", "arm"]),
                            ("pearson_correlation", ["score", "mood"]),
                            ("chi_square", ["arm", "mood"])):
        payload = call(sidecar, "run", test=test, variables=variables)
        assert payload["ok"], (test, payload.get("error"))
        assert payload["data"]["chart"]["kind"] != "none", test


def test_errors_carry_an_action(sidecar):
    bad = call(sidecar, "run", test="welch_t", variables=["score", "mood"])
    assert bad["ok"] is False
    assert bad["error"]["action"]
    unknown = sidecar.handle({"id": 1, "command": "teleport"})
    assert unknown["ok"] is False
    assert "Use one of" in unknown["error"]["action"]


def test_no_dataset_is_a_direction_not_a_crash():
    fresh = Sidecar()
    response = call(fresh, "run", test="welch_t", variables=["a", "b"])
    assert response["ok"] is False
    assert "Open a .csv" in response["error"]["action"]


def test_ledger_travels_with_every_response(sidecar):
    call(sidecar, "run", test="welch_t", variables=["score", "arm"])
    payload = call(sidecar, "run", test="pearson_correlation",
                   variables=["score", "mood"])["data"]
    assert payload["ledger"]["project_count"] == 2
    assert payload["ledger"]["family_wise_rate"] == pytest.approx(
        1 - 0.95 ** 2)


def test_the_frame_is_never_mutated(sidecar):
    before = sidecar.frame.copy(deep=True)
    call(sidecar, "run", test="welch_t", variables=["score", "arm"])
    call(sidecar, "set_variable", name="mood", type="continuous")
    pd.testing.assert_frame_equal(before, sidecar.frame)


def test_stdio_round_trip(csv_path):
    requests = [
        {"id": 1, "command": "ping"},
        {"id": 2, "command": "open", "params": {"path": csv_path}},
        {"id": 3, "command": "run",
         "params": {"test": "welch_t", "variables": ["score", "arm"]}},
    ]
    stdin = io.StringIO("\n".join(json.dumps(r) for r in requests) + "\n")
    stdout = io.StringIO()
    serve(stdin, stdout)
    lines = [json.loads(line) for line in
             stdout.getvalue().strip().split("\n")]
    assert [line["id"] for line in lines] == [1, 2, 3]
    assert all(line["ok"] for line in lines)
    assert lines[0]["data"]["version"]


def test_malformed_json_does_not_stop_the_loop(csv_path):
    stdin = io.StringIO('not json\n{"id": 2, "command": "ping"}\n')
    stdout = io.StringIO()
    serve(stdin, stdout)
    lines = [json.loads(line) for line in
             stdout.getvalue().strip().split("\n")]
    assert lines[0]["ok"] is False
    assert lines[1]["ok"] is True
