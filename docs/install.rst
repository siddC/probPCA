Installation
============

probPCA supports Python 3.12 to 3.14. Until the first PyPI release, install from GitHub:

.. code-block:: bash

   pip install "git+https://github.com/siddC/probPCA"

Required dependencies are NumPy, SciPy, scikit-learn and numba. Optional extras:

- ``[jax]``: the JAX backend, for the models that offer it.
- ``[mcmc]``: NumPyro, for the models that sample their posterior.
- ``[dev]``: everything needed to run the tests and build these docs.
