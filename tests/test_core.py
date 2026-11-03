import warnings

import numpy as np
import pytest

from probpca._core import EPS, ConvergenceMonitor, n_observed, safe_divide, safe_log
from probpca._core._validation import validate_X_mask
from probpca.exceptions import MonotonicityWarning

from ._toy import ToyRankOne


def _validate(X, mask=None, **kw):
    return validate_X_mask(ToyRankOne(), X, mask, reset=True, **kw)


class TestValidation:
    def test_complete_data_has_no_mask(self, rng):
        X = rng.random((5, 4))
        Xv, mask = _validate(X)
        assert mask is None
        np.testing.assert_array_equal(Xv, X)
        assert Xv is not X

    def test_nan_becomes_mask_and_zero(self, rng):
        X = rng.random((5, 4))
        X[1, 2] = np.nan
        Xv, mask = _validate(X)
        assert mask.dtype == bool and not mask[1, 2] and mask.sum() == 19
        assert Xv[1, 2] == 0.0
        assert np.isnan(X[1, 2]), "input must not be modified"

    def test_mask_and_nan_combine(self, rng):
        X = rng.random((4, 3))
        X[0, 0] = np.nan
        m = np.ones((4, 3), dtype=bool)
        m[2, 1] = False
        _, mask = _validate(X, m)
        assert not mask[0, 0] and not mask[2, 1] and mask.sum() == 10

    def test_integer_mask_accepted(self, rng):
        X = rng.random((3, 3))
        m = np.ones((3, 3), dtype=int)
        m[0, 0] = 0
        _, mask = _validate(X, m)
        assert mask.dtype == bool and not mask[0, 0]

    @pytest.mark.parametrize(
        "mask, match",
        [(np.ones((2, 2), dtype=bool), "shape"), (np.full((3, 3), 2), "boolean")],
    )
    def test_bad_mask(self, rng, mask, match):
        with pytest.raises(ValueError, match=match):
            _validate(rng.random((3, 3)), mask)

    def test_no_observed_entries(self):
        with pytest.raises(ValueError, match="no observed entries"):
            _validate(np.full((2, 2), np.nan))

    def test_inf_rejected(self):
        with pytest.raises(ValueError):
            _validate(np.array([[1.0, np.inf], [0.0, 1.0]]))

    def test_float32_preserved(self, rng):
        Xv, _ = _validate(rng.random((3, 3)).astype(np.float32))
        assert Xv.dtype == np.float32
        Xv, _ = _validate(np.arange(9).reshape(3, 3))
        assert Xv.dtype == np.float64

    def test_positive(self):
        X = np.array([[1.0, -1.0], [np.nan, 2.0]])
        with pytest.raises(ValueError, match="Negative values"):
            _validate(X, positive=True)
        X[0, 1] = np.nan  # a negative value at a missing entry is never seen
        _validate(X, positive=True)

    def test_n_observed(self, rng):
        X = rng.random((3, 4))
        assert n_observed(X, None) == 12
        assert n_observed(X, X > 2) == 0


class TestConvergenceMonitor:
    def test_patience(self):
        mon = ConvergenceMonitor(tol=1e-3, max_iter=100, patience=3)
        stops = [mon.update(v) for v in [10.0, 5.0, 5.0, 5.0, 5.0]]
        assert stops == [False, False, False, False, True]
        assert mon.converged and mon.n_iter == 5

    def test_small_change_streak_resets(self):
        mon = ConvergenceMonitor(tol=1e-3, patience=2)
        stops = [mon.update(v) for v in [10.0, 10.0, 5.0, 5.0, 5.0]]
        assert stops == [False, False, False, False, True]

    def test_max_iter(self):
        mon = ConvergenceMonitor(tol=0.0, max_iter=3)
        assert [mon.update(v) for v in [3.0, 2.0, 1.0]] == [False, False, True]
        assert not mon.converged

    def test_increase_warns(self):
        mon = ConvergenceMonitor()
        mon.update(1.0)
        with pytest.warns(MonotonicityWarning, match="increased"):
            mon.update(2.0)
        assert mon.violations == [2]

    def test_rounding_tolerated(self):
        mon = ConvergenceMonitor(monotone_rtol=1e-10)
        mon.update(1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mon.update(1.0 + 1e-12)

    def test_not_monotone_by_design(self):
        mon = ConvergenceMonitor(monotone=False)
        mon.update(1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mon.update(2.0)

    def test_non_finite(self):
        with pytest.raises(FloatingPointError):
            ConvergenceMonitor().update(np.nan)

    @pytest.mark.parametrize("kw", [{"tol": -1}, {"max_iter": 0}, {"patience": 0}])
    def test_bad_arguments(self, kw):
        with pytest.raises(ValueError):
            ConvergenceMonitor(**kw)


def test_numerics():
    assert np.isfinite(safe_log(0.0)) and safe_log(0.0) == np.log(EPS)
    assert safe_divide(1.0, 0.0) == 1.0 / EPS
    np.testing.assert_allclose(safe_divide(np.array([2.0]), np.array([4.0])), [0.5])
