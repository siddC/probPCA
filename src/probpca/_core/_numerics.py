"""Floors for logs and divisions, so no update divides by zero or takes log(0)."""

from __future__ import annotations

import numpy as np

#: Default floor for logs, digamma arguments and denominators.
EPS = 1e-12


def safe_log(x, eps: float = EPS):
    """``log(max(x, eps))``, elementwise."""
    return np.log(np.maximum(x, eps))


def safe_divide(a, b, eps: float = EPS):
    """``a / max(b, eps)``, elementwise."""
    return a / np.maximum(b, eps)
