Correctness standard
====================

No estimator ships without passing all six layers. The generic checks are public in
:mod:`probpca.utils.estimator_checks`.

1. **Monotone objective.** MM and EM objectives never increase, and variational ELBOs never
   decrease, at any iteration (relative tolerance 1e-10). Most derivation and implementation
   bugs show up here first.
2. **Limiting cases.** Each model reduces to its documented special case.
3. **Reproduction of published results.** Each model reruns its source paper's experiments
   with the same data, settings and metrics and must match the reported numbers to the
   precision the paper reports.
4. **Recovery.** On data simulated from the model itself, factors (after Hungarian
   matching), dispersion and effective rank are recovered within stated tolerances across
   10 seeds.
5. **Stationarity.** At convergence, KKT conditions are checked numerically: gradients vanish
   on free coordinates and are nonnegative on coordinates held at zero.
6. **Conformance.** scikit-learn's estimator checks, pickling, float32 and float64, and dense
   and sparse inputs where supported.

Papers usually report summary metrics rather than fitted factors, and rarely state their
random initialization. Layer 3 therefore confirms the algorithm reaches the paper's results,
while layers 1, 2, 4 and 5 carry the exact correctness checks.

Algorithms are implemented from the papers only. Where a maintained public package already
implements the same model, it may be run as a black-box cross-check on shared inputs, never
as a source.
