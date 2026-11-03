"""Layer 4: recover factors and rank from data simulated by the model itself, over 10 seeds."""

import pytest

from probpca.utils.estimator_checks import check_recovery

from ._registry import SPECS


@pytest.mark.slow
@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_recovery(spec):
    check_recovery(
        spec.make, spec.simulate, seeds=range(10),
        min_similarity=spec.recovery_similarity, metric=spec.match_metric,
    )


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_recovery_smoke(spec):
    check_recovery(
        spec.make, spec.simulate, seeds=[0],
        min_similarity=spec.recovery_similarity, metric=spec.match_metric,
    )
