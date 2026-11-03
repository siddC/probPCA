"""Layer 2: each model reduces to its documented special case.

Each ``ModelSpec`` lists its limiting cases (for example NB-NMF approaching Poisson NMF as
the dispersion grows, or a one-component mixture of PPCA equal to PPCA).
"""

import pytest

from ._registry import SPECS

CASES = [(spec, limit) for spec in SPECS for limit in spec.limits]


@pytest.mark.parametrize(
    "spec, limit", CASES, ids=[f"{s.name}-{f.__name__}" for s, f in CASES]
)
def test_limit(spec, limit):
    limit(seed=0)
