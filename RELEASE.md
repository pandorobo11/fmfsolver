# Release Guide

This file defines the release operation for `fmfsolver`.

## Policy

- Versioning uses SemVer: `MAJOR.MINOR.PATCH`.
- `pyproject.toml` is the single source of truth for the package version.
- Version is bumped only for release commits.
- Git tag must match version using `vX.Y.Z`.

## Release Steps

1. Ensure working tree is clean.
2. Run tests:
   ```bash
   .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
   ```
3. Bump version in `pyproject.toml`:
   ```bash
   python scripts/bump_version.py patch
   ```
   or:
   ```bash
   python scripts/bump_version.py minor
   python scripts/bump_version.py major
   python scripts/bump_version.py 1.2.3
   ```
4. Refresh `uv.lock` (do not skip):
   ```bash
   uv lock
   ```
5. Commit:
   ```bash
   git add pyproject.toml uv.lock
   git commit -m "Release vX.Y.Z"
   ```
6. Create tag and push:
   ```bash
   git tag vX.Y.Z
   git push
   git push origin vX.Y.Z
   ```
7. GitHub Actions runs `uv build`, then automatically creates a GitHub Release for the pushed `vX.Y.Z` tag, attaching `dist/*.whl` and `dist/*.tar.gz`.
8. Review the generated release notes on GitHub and edit if needed.

## Notes

- CI checks that tag `vX.Y.Z` matches `pyproject.toml` version before release creation.
- Release creation and build artifact upload are handled by `.github/workflows/ci.yml` (tag push trigger).
- If you need a prerelease, use a separate branch strategy; this repository currently tracks stable SemVer releases in `pyproject.toml`.
