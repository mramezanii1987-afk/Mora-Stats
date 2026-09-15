import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mora_engine.models import Design, VarType, Variable  # noqa: E402


@pytest.fixture
def sleep_data() -> pd.DataFrame:
    """Student's sleep data, the dataset the t-test was first published on.

    Published values for the independent samples comparison: t = -1.8608,
    Welch df = 17.776, p = .0794.
    """
    group1 = [0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0]
    group2 = [1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4]
    return pd.DataFrame({
        "extra": group1 + group2,
        "drug": ["drug 1"] * 10 + ["drug 2"] * 10,
    })


@pytest.fixture
def sleep_vars():
    return (Variable("extra", VarType.CONTINUOUS, unit="hours"),
            Variable("drug", VarType.NOMINAL))


@pytest.fixture
def big_effect() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return pd.DataFrame({
        "score": np.r_[rng.normal(50, 8, 80), rng.normal(62, 8, 80)],
        "arm": ["control"] * 80 + ["treatment"] * 80,
    })


@pytest.fixture
def null_data() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "score": np.r_[rng.normal(50, 10, 18), rng.normal(50, 10, 18)],
        "arm": ["a"] * 18 + ["b"] * 18,
    })


@pytest.fixture
def three_groups() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    return pd.DataFrame({
        "score": np.r_[rng.normal(50, 9, 40), rng.normal(55, 9, 40),
                       rng.normal(58, 9, 40)],
        "site": ["north"] * 40 + ["south"] * 40 + ["east"] * 40,
    })


@pytest.fixture
def ceiling_data() -> pd.DataFrame:
    values = [7] * 30 + [6] * 10 + [5] * 5 + [4] * 5
    other = [4, 5, 6, 7] * 12 + [5, 6]
    return pd.DataFrame({
        "satisfaction": values + other,
        "group": ["a"] * 50 + ["b"] * 50,
    })


@pytest.fixture
def scale_var():
    return Variable("satisfaction", VarType.ORDINAL, unit="scale points",
                    lower_bound=1, upper_bound=7)


@pytest.fixture
def table_data() -> pd.DataFrame:
    rows = (["yes"] * 30 + ["no"] * 70)
    cols = (["passed"] * 24 + ["failed"] * 6 +
            ["passed"] * 35 + ["failed"] * 35)
    return pd.DataFrame({"attended": rows, "outcome": cols})


@pytest.fixture
def linear_data() -> pd.DataFrame:
    rng = np.random.default_rng(17)
    x = rng.normal(10, 2, 90)
    y = 3.0 + 0.8 * x + rng.normal(0, 1.5, 90)
    return pd.DataFrame({"hours": x, "grade": y})
