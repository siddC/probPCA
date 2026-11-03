"""The models under test.

Adding a ``ModelSpec`` to ``MODELS`` runs that model through every layer of the correctness
standard: monotone objective, limiting cases, recovery, stationarity and conformance (the
reproduction layer reads its own fixtures). ``SPECS`` also holds the test double, so the
harness itself stays tested.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from . import _toy


@dataclass(frozen=True)
class ModelSpec:
    """How to build, simulate and check one model.

    Attributes
    ----------
    name : str
        Test id.
    make : callable
        ``make(seed)`` returns an unfitted estimator.
    simulate : callable
        ``simulate(seed)`` returns ``(X, mask, true_components)`` from the model itself.
    limits : list of callable
        Limiting-case tests (layer 2); each takes a seed and raises on failure.
    stationarity : callable or None
        ``stationarity(est, X, mask)`` returns ``[(params, grad, nonnegative, scale), ...]``
        at the fitted point, for the KKT check (layer 5); ``scale`` is the size of the terms
        the gradient is made of.
    recovery_similarity : float
        Smallest matched absolute similarity accepted in the recovery test (layer 4).
    match_metric : str
        Similarity used to match components.
    """

    name: str
    make: Callable
    simulate: Callable
    limits: list[Callable] = field(default_factory=list)
    stationarity: Callable | None = None
    recovery_similarity: float = 0.95
    match_metric: str = "correlation"


TOY = ModelSpec(
    name="toy",
    make=lambda seed: _toy.ToyRankOne(random_state=seed),
    simulate=_toy.simulate,
    stationarity=_toy.stationarity,
    recovery_similarity=0.99,
)

#: Models of this package. Each new estimator adds its spec here.
MODELS: list[ModelSpec] = []

SPECS = [TOY, *MODELS]
