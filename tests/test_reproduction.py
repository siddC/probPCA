"""Layer 3: rerun each source paper's experiments and match its published numbers.

Every file in ``reproduction/fixtures`` describes one experiment::

    {
      "model": "PPCA",
      "paper": "Tipping & Bishop 1999a, Table 1",
      "script": "tipping1999a_table1",
      "published": {"metric_name": 0.123},
      "rtol": 0.005
    }

``reproduction/scripts/<script>.py`` defines ``run()``, which rebuilds the paper's inputs and
settings and returns the same metrics. Published values are compared to the precision the
paper reports; ``rtol`` (or ``atol``) states it.
"""

import importlib.util
import json

import numpy as np
import pytest

from .conftest import FIXTURES, ROOT

FILES = sorted(FIXTURES.glob("*.json"))


def _run(script):
    path = ROOT / "reproduction" / "scripts" / f"{script}.py"
    spec = importlib.util.spec_from_file_location(f"reproduction_{script}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run()


@pytest.mark.slow
@pytest.mark.parametrize("path", FILES, ids=[p.stem for p in FILES])
def test_reproduction(path):
    fixture = json.loads(path.read_text())
    result = _run(fixture["script"])
    for name, published in fixture["published"].items():
        np.testing.assert_allclose(
            result[name], published,
            rtol=fixture.get("rtol", 0.0), atol=fixture.get("atol", 0.0),
            err_msg=f"{fixture['paper']}: {name}",
        )
