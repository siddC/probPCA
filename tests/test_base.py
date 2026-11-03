import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from probpca import BaseFactorization
from probpca.utils.estimator_checks import (
    check_api_contract,
    check_mask_equivalence,
    check_pickle,
)

from ._toy import ToyRankOne, simulate


@pytest.fixture
def data():
    X, mask, _ = simulate(0)
    return X, mask


def test_api_contract(data):
    X, mask = data
    check_api_contract(ToyRankOne(random_state=0), X)
    check_api_contract(ToyRankOne(random_state=0), X, mask=mask)


def test_mask_equivalence(data):
    X, mask = data
    check_mask_equivalence(ToyRankOne(random_state=0), X, mask)


def test_pickle(data):
    check_pickle(ToyRankOne(random_state=0), data[0])


def test_score_is_per_observed_entry(data):
    X, mask = data
    est = ToyRankOne(random_state=0).fit(X, mask=mask)
    np.testing.assert_allclose(
        est.score(X, mask=mask), est.score_samples(X, mask=mask).sum() / mask.sum()
    )


def test_unfitted():
    with pytest.raises(NotFittedError):
        ToyRankOne().transform(np.ones((2, 2)))


def test_inverse_transform_shape(data):
    est = ToyRankOne(random_state=0).fit(data[0])
    with pytest.raises(ValueError, match="components"):
        est.inverse_transform(np.ones((3, 2)))


@pytest.mark.parametrize("n", [0, -1, 1.5])
def test_sample_bad_n(data, n):
    est = ToyRankOne(random_state=0).fit(data[0])
    with pytest.raises(ValueError, match="n_samples"):
        est.sample(n)


def test_missing_fitted_attribute(data):
    class Incomplete(ToyRankOne):
        def _fit(self, X, mask):
            Z = super()._fit(X, mask)
            del self.converged_
            return Z

    with pytest.raises(RuntimeError, match="converged_"):
        Incomplete(random_state=0).fit(data[0])


def test_hooks_must_be_implemented():
    with pytest.raises(NotImplementedError):
        BaseFactorization().fit(np.ones((3, 2)))


def test_feature_names_out(data):
    est = ToyRankOne(random_state=0).fit(data[0])
    assert est.get_feature_names_out().tolist() == ["toyrankone0"]
