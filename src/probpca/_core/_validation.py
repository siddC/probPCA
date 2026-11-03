"""Input validation shared by every estimator: data, missing-data masks and dtypes."""

from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_non_negative, validate_data

FLOAT_DTYPES = (np.float64, np.float32)


def validate_X_mask(
    estimator,
    X,
    mask=None,
    *,
    reset: bool,
    positive: bool = False,
    accept_sparse: bool = False,
):
    """Validate ``X`` and its observation mask.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator calling this function. With ``reset=True`` its ``n_features_in_`` and
        ``feature_names_in_`` are set; with ``reset=False`` they are checked.
    X : array-like of shape (n_samples, n_features)
        Data. NaN marks a missing entry; infinite values are rejected.
    mask : array-like of bool, shape (n_samples, n_features), default=None
        True where an entry is observed. An entry is treated as observed only if the mask is
        True there and ``X`` is not NaN.
    reset : bool
        Whether this call starts a fit.
    positive : bool, default=False
        Reject negative observed entries.
    accept_sparse : bool, default=False
        Accept a SciPy sparse matrix (CSR or CSC). Sparse input does not take a mask yet.

    Returns
    -------
    X : ndarray or sparse matrix of float64 or float32
        A copy of the data with missing entries set to 0, so arithmetic on it is safe. float32
        input stays float32; every other dtype becomes float64.
    mask : ndarray of bool or None
        True where an entry is observed. None means every entry is observed.
    """
    X = validate_data(
        estimator,
        X,
        reset=reset,
        accept_sparse=["csr", "csc"] if accept_sparse else False,
        dtype=FLOAT_DTYPES,
        ensure_all_finite="allow-nan",
        copy=True,
    )
    whom = type(estimator).__name__

    if hasattr(X, "toarray"):  # sparse
        if mask is not None:
            raise ValueError(f"{whom} does not accept a mask with sparse X yet.")
        if np.isnan(X.data).any():
            raise ValueError(f"{whom} does not accept NaN in sparse X yet.")
        if positive:
            check_non_negative(X, whom)
        return X, None

    observed = ~np.isnan(X)
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != X.shape:
            raise ValueError(f"mask has shape {mask.shape} but X has shape {X.shape}.")
        if mask.dtype != bool:
            if not np.isin(mask, (0, 1)).all():
                raise ValueError("mask must be boolean (True = observed).")
            mask = mask.astype(bool)
        observed &= mask

    if observed.all():
        mask = None
    else:
        if not observed.any():
            raise ValueError(f"{whom} received no observed entries.")
        X[~observed] = 0.0
        mask = observed

    if positive:
        check_non_negative(X, whom)
    return X, mask


def n_observed(X, mask) -> int:
    """Number of observed entries."""
    return int(np.prod(X.shape)) if mask is None else int(mask.sum())
