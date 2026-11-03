"""Metrics for comparing fitted factors with each other or with a known truth.

Factorizations are identifiable only up to a permutation of their components (and, depending
on the model, a scaling or sign), so recovery is measured after matching components.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ["match_components", "principal_angles"]


def _similarity(A, B, metric):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError(
            "A and B must be 2-D with the same number of columns, got "
            f"{np.shape(A)} and {np.shape(B)}."
        )
    if metric == "correlation":
        A = A - A.mean(axis=1, keepdims=True)
        B = B - B.mean(axis=1, keepdims=True)
    elif metric != "cosine":
        raise ValueError(f"metric must be 'correlation' or 'cosine', got {metric!r}.")
    na = np.linalg.norm(A, axis=1, keepdims=True)
    nb = np.linalg.norm(B, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        S = (A @ B.T) / (na * nb.T)
    return np.nan_to_num(S, nan=0.0)


def match_components(A, B, *, metric="correlation"):
    """Match the rows of ``A`` to the rows of ``B`` by the Hungarian algorithm.

    Each row is a component (for example a row of ``components_``). The assignment maximizes
    the total absolute similarity, so a component whose sign flipped still matches.

    Parameters
    ----------
    A : array-like of shape (k_a, n_features)
        First set of components.
    B : array-like of shape (k_b, n_features)
        Second set of components.
    metric : {"correlation", "cosine"}, default="correlation"
        Similarity between two components.

    Returns
    -------
    rows : ndarray of shape (min(k_a, k_b),)
        Indices into ``A``.
    cols : ndarray of shape (min(k_a, k_b),)
        Indices into ``B``; ``A[rows[i]]`` is matched to ``B[cols[i]]``.
    similarity : ndarray of shape (min(k_a, k_b),)
        Signed similarity of each matched pair.

    Examples
    --------
    >>> import numpy as np
    >>> from probpca.metrics import match_components
    >>> A = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 1.0]])
    >>> rows, cols, sim = match_components(A, A[::-1])
    >>> cols.tolist(), np.round(sim, 6).tolist()
    ([1, 0], [1.0, 1.0])
    """
    S = _similarity(A, B, metric)
    rows, cols = linear_sum_assignment(-np.abs(S))
    return rows, cols, S[rows, cols]


def principal_angles(A, B):
    """Principal angles between the row spaces of ``A`` and ``B``.

    Fitted PCA loadings are identifiable only up to a rotation within their subspace, so
    recovery is measured by the angles between subspaces rather than by matching rows.

    Parameters
    ----------
    A : array-like of shape (k_a, n_features)
        Rows spanning the first subspace (for example ``components_``).
    B : array-like of shape (k_b, n_features)
        Rows spanning the second subspace.

    Returns
    -------
    angles : ndarray of shape (min(k_a, k_b),)
        Angles in radians, in ascending order. All zero means one subspace contains the
        other.

    Examples
    --------
    >>> import numpy as np
    >>> from probpca.metrics import principal_angles
    >>> A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    >>> B = np.array([[1.0, 1.0, 0.0]])
    >>> np.round(principal_angles(A, B), 6).tolist()
    [0.0]
    """
    from scipy.linalg import subspace_angles

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError(
            "A and B must be 2-D with the same number of columns, got "
            f"{np.shape(A)} and {np.shape(B)}."
        )
    return np.sort(subspace_angles(A.T, B.T))
