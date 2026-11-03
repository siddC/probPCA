"""Layer 5: first-order (KKT) conditions hold at convergence."""

import pytest

from probpca.utils.estimator_checks import check_stationarity

from ._registry import SPECS

CASES = [s for s in SPECS if s.stationarity is not None]


@pytest.mark.parametrize("spec", CASES, ids=lambda s: s.name)
@pytest.mark.parametrize("missing", [False, True], ids=["complete", "masked"])
def test_stationarity(spec, missing):
    X, mask, _ = spec.simulate(0)
    mask = mask if missing else None
    est = spec.make(0).fit(X, mask=mask)
    assert est.converged_, "fit must converge before stationarity is checked"
    for params, grad, nonnegative, scale in spec.stationarity(est, X, mask):
        check_stationarity(params, grad, nonnegative=nonnegative, scale=scale)
