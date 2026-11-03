## What this changes

<!-- One or two sentences. Link the issue if there is one. -->

## Checklist for a new or changed model

- [ ] Implemented from the source paper only; no author code was read or translated
- [ ] Updates derived in this package's orientation (X is samples × features, X ≈ Z · components_), derivation in the model's docs page
- [ ] `ModelSpec` added to `tests/_registry.py`, with a simulator in `probpca.datasets`
- [ ] Layer 1: objective monotone (complete and masked data)
- [ ] Layer 2: limiting cases listed in the spec and passing
- [ ] Layer 3: reproduction fixture and script for the paper's experiments
- [ ] Layer 4: recovery over 10 seeds
- [ ] Layer 5: stationarity at convergence
- [ ] Layer 6: scikit-learn checks, pickling, float32
- [ ] Model page in `docs/`, entry in `CHANGELOG.md`
