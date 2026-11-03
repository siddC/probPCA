"""Base estimator implementing the API contract shared by every probpca model."""

from __future__ import annotations

import numbers

import numpy as np
from sklearn.base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted

from ._core import n_observed, validate_X_mask

#: Attributes every fitted estimator sets (see the API contract in the docs).
FITTED_ATTRIBUTES = (
    "components_",
    "n_components_",
    "objective_",
    "n_iter_",
    "converged_",
    "loglik_",
)


class BaseFactorization(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Base class for probpca estimators.

    It implements the public API (validation, masks, scoring, sampling) once, so every model
    behaves the same way. ``X`` has shape (n_samples, n_features). ``transform`` returns
    latent values of shape (n_samples, n_components) and ``components_`` has shape
    (n_components, n_features). Missing entries are passed as NaN in ``X`` or as a boolean
    ``mask`` (True = observed); likelihoods, updates and scores sum over observed entries only.

    A subclass implements five methods, each called on validated input:

    ``_fit(X, mask)``
        Fit the model and set the fitted attributes listed in ``FITTED_ATTRIBUTES``
        (``loglik_`` may be left to the base class). ``objective_`` is the minimized
        objective after each iteration. Return the latent values of the training samples,
        or None to have them computed by ``_transform``.
    ``_transform(X, mask)``
        Latent values of new samples, with the fitted components held fixed.
    ``_inverse_transform(Z)``
        Mean of the observation model on the data scale.
    ``_score_samples(X, mask)``
        Log-likelihood of each sample over its observed entries, shape (n_samples,). For a
        Bayesian model this is the posterior-predictive log-likelihood.
    ``_sample(n_samples, random_state)``
        Synthetic data from the fitted model, shape (n_samples, n_features).

    ``mask`` is either None (every entry observed) or a boolean array the shape of ``X``.
    """

    #: Reject negative observed entries.
    _positive_only: bool = False
    #: Accept SciPy sparse input.
    _accept_sparse: bool = False

    # fitted attributes (set by ``_fit``)
    components_: np.ndarray
    n_components_: int
    objective_: np.ndarray
    n_iter_: int
    converged_: bool
    loglik_: float

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.positive_only = self._positive_only
        tags.input_tags.sparse = self._accept_sparse
        tags.transformer_tags.preserves_dtype = ["float64", "float32"]
        return tags

    @property
    def _n_features_out(self):
        return self.components_.shape[0]

    # -- hooks implemented by each model ---------------------------------------------------

    def _fit(self, X, mask):
        raise NotImplementedError

    def _transform(self, X, mask):
        raise NotImplementedError

    def _inverse_transform(self, Z):
        raise NotImplementedError

    def _score_samples(self, X, mask):
        raise NotImplementedError

    def _sample(self, n_samples, random_state):
        raise NotImplementedError

    # -- public API ------------------------------------------------------------------------

    def _validate(self, X, mask, *, reset):
        return validate_X_mask(
            self,
            X,
            mask,
            reset=reset,
            positive=self._positive_only,
            accept_sparse=self._accept_sparse,
        )

    def _fit_validated(self, X, mask):
        X, mask = self._validate(X, mask, reset=True)
        Z = self._fit(X, mask)
        missing = [a for a in FITTED_ATTRIBUTES if a != "loglik_" and not hasattr(self, a)]
        if missing:
            raise RuntimeError(f"{type(self).__name__}._fit did not set {missing}.")
        self.objective_ = np.asarray(self.objective_, dtype=np.float64)
        if not hasattr(self, "loglik_"):
            self.loglik_ = float(np.sum(self._score_samples(X, mask)))
        if Z is None:
            Z = self._transform(X, mask)
        return self._as_input_dtype(Z, X)

    @staticmethod
    def _as_input_dtype(Z, X):
        # float32 in, float32 out: models may compute in float64 internally
        return np.asarray(Z).astype(X.dtype, copy=False)

    def fit(self, X, y=None, mask=None):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. NaN marks a missing entry.
        y : None
            Ignored; present for scikit-learn compatibility.
        mask : array-like of bool, shape (n_samples, n_features), default=None
            True where an entry is observed. None means every non-NaN entry is observed.

        Returns
        -------
        self : object
            The fitted estimator.
        """
        self._fit_validated(X, mask)
        return self

    def fit_transform(self, X, y=None, mask=None):
        """Fit the model, then return the latent values of the training samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. NaN marks a missing entry.
        y : None
            Ignored; present for scikit-learn compatibility.
        mask : array-like of bool, shape (n_samples, n_features), default=None
            True where an entry is observed.

        Returns
        -------
        Z : ndarray of shape (n_samples, n_components)
            Latent values of the training samples.
        """
        return self._fit_validated(X, mask)

    def transform(self, X, mask=None):
        """Latent values for new samples, with the fitted components held fixed.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data. NaN marks a missing entry.
        mask : array-like of bool, shape (n_samples, n_features), default=None
            True where an entry is observed.

        Returns
        -------
        Z : ndarray of shape (n_samples, n_components)
            Latent values.
        """
        check_is_fitted(self)
        X, mask = self._validate(X, mask, reset=False)
        return self._as_input_dtype(self._transform(X, mask), X)

    def inverse_transform(self, Z):
        """Mean of the observation model on the data scale (the reconstruction).

        Parameters
        ----------
        Z : array-like of shape (n_samples, n_components)
            Latent values, as returned by ``transform``.

        Returns
        -------
        X_mean : ndarray of shape (n_samples, n_features)
            Mean parameter of each entry.
        """
        check_is_fitted(self)
        Z = check_array(Z, dtype=(np.float64, np.float32))
        if Z.shape[1] != self.n_components_:
            raise ValueError(
                f"Z has {Z.shape[1]} columns but the model has {self.n_components_} components."
            )
        return self._inverse_transform(Z)

    def score_samples(self, X, mask=None):
        """Log-likelihood of each sample, summed over its observed entries.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data. NaN marks a missing entry.
        mask : array-like of bool, shape (n_samples, n_features), default=None
            True where an entry is observed.

        Returns
        -------
        loglik : ndarray of shape (n_samples,)
            Log-likelihood per sample. Bayesian models return the posterior-predictive
            log-likelihood.
        """
        check_is_fitted(self)
        X, mask = self._validate(X, mask, reset=False)
        return np.asarray(self._score_samples(X, mask), dtype=np.float64)

    def score(self, X, y=None, mask=None):
        """Mean log-likelihood per observed entry.

        Dividing by the number of observed entries, not samples, makes scores comparable
        across masks with different amounts of missing data, which is how held-out
        likelihood is used to choose a rank.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data. NaN marks a missing entry.
        y : None
            Ignored; present for scikit-learn compatibility.
        mask : array-like of bool, shape (n_samples, n_features), default=None
            True where an entry is observed (for held-out scoring, the held-out entries).

        Returns
        -------
        score : float
            Mean log-likelihood per observed entry.
        """
        check_is_fitted(self)
        X, mask = self._validate(X, mask, reset=False)
        total = float(np.sum(self._score_samples(X, mask)))
        return total / n_observed(X, mask)

    def sample(self, n_samples=1, random_state=None):
        """Draw synthetic data from the fitted model.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to draw.
        random_state : int, RandomState instance or None, default=None
            Controls the draw.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Synthetic data. Bayesian models draw from the posterior predictive.
        """
        check_is_fitted(self)
        if not isinstance(n_samples, numbers.Integral) or n_samples < 1:
            raise ValueError(f"n_samples must be a positive integer, got {n_samples!r}.")
        return self._sample(int(n_samples), check_random_state(random_state))
