# Deployment Guide

This guide covers how to test, build, and publish the `transformer-cloner` package to PyPI.

## Prerequisites

```bash
pip install build twine pytest
```

## Pre-Deployment Checklist

1. **Update version** in both files:

   - `pyproject.toml`: `version = "X.Y.Z"`
   - `src/transformer_cloner/__init__.py`: `__version__ = "X.Y.Z"`

2. **Run tests**:

   ```bash
   python -m pytest tests/ -v
   ```

3. **Verify all tests pass** before proceeding.

---

## Build Package

```bash
# Clean previous builds
rm -rf dist/

# Build wheel and source distribution
python -m build
```

This creates:

- `dist/transformer_cloner-X.Y.Z-py3-none-any.whl`
- `dist/transformer_cloner-X.Y.Z.tar.gz`

---

## PyPI API Token

**Token Name:** cloner  
**Scope:** Entire account (all projects)

```
pypi-your-pypi-token
```

### Configure ~/.pypirc (Optional)

```ini
[pypi]
username = __token__
password = pypi-your-pypi-token
```

---

## Upload to PyPI

### Option 1: With ~/.pypirc configured

```bash
python -m twine upload dist/*
```

### Option 2: Manual token entry

```bash
python -m twine upload dist/*
# Enter username: __token__
# Enter password: <paste token>
```

---

## One-Liner Deploy

```bash
rm -rf dist/ && python -m build && python -m twine upload dist/*
```

---

## Verify Deployment

```bash
# Check PyPI page
open https://pypi.org/project/transformer-cloner/

# Install and test
pip install --upgrade transformer-cloner
python -c "from transformer_cloner import __version__; print(__version__)"
```

---

## Version History

| Version | Changes                                                            |
| ------- | ------------------------------------------------------------------ |
| 0.1.6   | Added `VocabPrunedTokenizer` wrapper for automatic token remapping |
| 0.1.5   | Comprehensive README docs, 49 tests                                |
| 0.1.4   | Fixed Gemma-style model validation (head_dim check)                |
| 0.1.3   | Fixed `clone_with_vocab_pruning` to return 3 values                |
| 0.1.2   | Updated README                                                     |
| 0.1.1   | Initial PyPI release                                               |
