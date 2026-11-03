"""Smoke test of an installed probpca (CI runs it on the built wheel, outside the checkout).

Checks that probpca is imported from site-packages and that every module in the package
imports, so a module left out of the wheel fails here.
"""

import importlib
import pathlib
import pkgutil

import probpca

path = pathlib.Path(probpca.__file__)
assert "site-packages" in path.parts, f"probpca imported from {path}, not from the install"
assert (path.parent / "py.typed").exists(), "py.typed is missing from the wheel"
for mod in pkgutil.walk_packages(probpca.__path__, "probpca."):
    importlib.import_module(mod.name)
print(f"probpca {probpca.__version__} from {path.parent}: every module imports")
