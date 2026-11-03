"""Layer 1: objectives never increase between iterations (relative tolerance 1e-10)."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from probpca.utils.estimator_checks import check_monotone_objective

from ._registry import SPECS


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
@pytest.mark.parametrize("missing", [False, True], ids=["complete", "masked"])
def test_monotone(spec, missing):
    for seed in range(3):
        X, mask, _ = spec.simulate(seed)
        check_monotone_objective(spec.make(seed), X, mask if missing else None)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
@given(data=st.data())
def test_monotone_random_inputs(spec, data):
    shape = data.draw(st.tuples(st.integers(3, 12), st.integers(2, 8)), label="shape")
    X = data.draw(
        arrays(np.float64, shape, elements=st.floats(-50.0, 50.0, allow_subnormal=False)),
        label="X",
    )
    seed = data.draw(st.integers(0, 2**16), label="seed")
    check_monotone_objective(spec.make(seed), X)
