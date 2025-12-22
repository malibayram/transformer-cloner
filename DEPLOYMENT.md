# Deployment Guide

This guide covers how to deploy the `transformer-cloner` package using the automated GitHub Actions workflow.

## Prerequisites

1.  **PyPI Token**: Ensure a PyPI API token is added to the GitHub repository secrets as `PYPI_API_TOKEN`.

## Automated Deployment

The deployment process is automated via GitHub Actions. It triggers whenever a new tag starting with `v` is pushed (e.g., `v0.2.1`).

### Steps to Release

1.  **Bump Version**:
    Use the helper script to bump the version in `pyproject.toml` and `src/transformer_cloner/__init__.py`.

    ```bash
    # Bump patch version (0.2.0 -> 0.2.1)
    python scripts/bump_version.py patch

    # Or bump minor/major
    python scripts/bump_version.py minor
    ```

2.  **Commit and Tag**:

    ```bash
    git add pyproject.toml src/transformer_cloner/__init__.py
    git commit -m "Bump version to $(python -c "import tomli; print(tomli.load(open('pyproject.toml', 'rb'))['project']['version'])" 2>/dev/null || grep 'version =' pyproject.toml | cut -d '"' -f 2)"
    # Or just check the version and tag manually:

    # Tag the release
    git tag v0.2.1  # Replace with actual new version
    ```

3.  **Push to GitHub**:

    ```bash
    git push origin main --tags
    ```

4.  **Watch the Magic**:
    - Go to the "Actions" tab in the GitHub repository.
    - You will see a "Publish to PyPI" workflow running.
    - Once completed, the package will be available on PyPI.

## Verification

After the workflow finishes:

```bash
# Check PyPI page
open https://pypi.org/project/transformer-cloner/

# Install and test
pip install --upgrade transformer-cloner
python -c "from transformer_cloner import __version__; print(__version__)"
```

## Manual Deployment (Fallback)

If the automation fails, you can still deploy manually:

```bash
rm -rf dist/
python -m build
python -m twine upload dist/*
```
