# Releasing probPCA

Releases are tag-triggered. Pushing a tag `vX.Y.Z` runs `.github/workflows/release.yml`:
the full CI suite, a build checked by `twine`, a check that the tag matches `__version__`,
trusted publishing to PyPI, then a GitHub release (which Zenodo archives with a DOI).

## One-time setup

1. **PyPI trusted publisher.** On PyPI, add a pending publisher for project `probpca`:
   owner `siddC`, repository `probPCA`, workflow `release.yml`, environment `pypi`.
2. **GitHub environment.** In the repository settings, create the environment `pypi`
   (optionally with required reviewers, so each publish waits for approval).
3. **Zenodo.** Sign in to Zenodo with GitHub and switch `siddC/probPCA` on under
   GitHub integration. Each published GitHub release is then archived with a DOI; add the
   concept DOI to `CITATION.cff` and the README after the first one.
4. **Read the Docs.** Import the repository; `.readthedocs.yaml` configures the build.
5. **Codecov.** Add the repository and store its token as the `CODECOV_TOKEN` secret.
6. **Branch ruleset.** Protect `main`: require a pull request and the `ci-passed` status check.

## Each release

1. On a branch, set `__version__` in `src/probpca/__init__.py` (for example `0.1.0`, or
   `0.1.0rc1` for a pre-release), move the `CHANGELOG.md` entries under the version with
   the date, and add `version` and `date-released` to `CITATION.cff`.
2. Merge the pull request once `ci-passed` is green.
3. Tag the merge commit and push the tag:

   ```bash
   git switch main && git pull
   git tag -a v0.1.0 -m "probPCA 0.1.0"
   git push origin v0.1.0
   ```

4. After the release, set `__version__` to the next development version (`0.2.0.dev0`) and
   open a new `Unreleased` section in `CHANGELOG.md`.
5. **conda-forge:** for the first release, submit a recipe to `conda-forge/staged-recipes`;
   afterwards the feedstock's bot opens a pull request for each new PyPI release.
