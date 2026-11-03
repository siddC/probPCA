"""Layer 6: scikit-learn conformance, pickling, dtypes and the missing-data contract."""

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from probpca.utils.estimator_checks import run_generic_checks

from ._registry import SPECS


@parametrize_with_checks([spec.make(0) for spec in SPECS])
def test_sklearn_compatible(estimator, check):
    check(estimator)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_generic_checks(spec):
    X, mask, _ = spec.simulate(0)
    run_generic_checks(spec.make(0), X, mask)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_float32(spec):
    X, _, _ = spec.simulate(0)
    est = spec.make(0)
    Z32 = est.fit_transform(X.astype(np.float32))
    assert Z32.dtype == np.float32
    Z64 = spec.make(0).fit_transform(X)
    np.testing.assert_allclose(Z32, Z64, rtol=1e-3, atol=1e-3 * np.abs(Z64).max())
