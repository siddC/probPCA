import json
import pathlib

import numpy as np
import pytest
from hypothesis import HealthCheck, settings

settings.register_profile(
    "default", deadline=None, max_examples=25, suppress_health_check=[HealthCheck.too_slow]
)
settings.load_profile("default")

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "reproduction" / "fixtures"


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def load_fixture(name):
    """Published values to reproduce, stored in ``reproduction/fixtures/<name>.json``."""
    return json.loads((FIXTURES / f"{name}.json").read_text())
