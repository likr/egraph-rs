---
name: Release Checklist
about: Step-by-step release checklist for publishing v6 prereleases or GA tags to PyPI and npm via OIDC.
title: "release: v6.X.Y"
labels:
  - release
---

## Pre-Release Preparation
- [ ] Update version across all 11 locations (`Cargo.toml`, `crates/python/Cargo.toml`, `crates/wasm/Cargo.toml`, `crates/wasm/package.json`, git-tracked `crates/wasm/package-lock.json` root & `packages[""]`, `js/examples/package.json`, `CHANGELOG.md`, `crates/python/README.md`, `crates/python/docs/source/getting_started/installation.rst`, `crates/python/docs/source/getting_started/migration_v6.rst`, `docs/MIGRATION_V6.md`).
- [ ] For `beta` / `rc` / GA releases, update classifiers in `crates/python/pyproject.toml` (remove `"Development Status :: 3 - Alpha"`).
- [ ] Confirm `crates/wasm/package.json` `"repository"` block (`url`, `directory: "crates/wasm"`) for npm `--provenance`.
- [ ] Simulate `.github/workflows/release.yml` `verify-tag-and-version` parser locally with `GITHUB_REF_NAME=v6.X.Y...`.
- [ ] Run `EGRAPH_PARALLEL_WORKERS=1 ./scripts/verify.sh --all --build-wasm --docs` (`hard_fails=0`, Sphinx `doctest`, TypeDoc, Vite build) and `node crates/wasm/tests/playwright_smoke.mjs`.
- [ ] Confirm `jp_cjk=0` across all modified files and clean git status.

## Tag & Publish
- [ ] Commit (`chore(release): bump version to ...`), push `origin/master`, and wait for `.github/workflows/ci.yml` (`6/6` jobs) green.
- [ ] Confirm `pypi-release` / `npm-release` GitHub Environments and PyPI/npm Trusted Publishers (`release.yml`) are configured.
- [ ] Create and push annotated tag: `git tag -a v6.X.Y... -m "Release v6.X.Y..." && git push origin v6.X.Y...` (never force-move failed tags; bump to next patch/prerelease instead).
- [ ] Monitor `.github/workflows/release.yml` (`gh run watch <id> --exit-status`) through all 11 jobs.

## Live Registry Clean-Env Smoke Tests
- [ ] Verify PyPI install in fresh venv (`pip install --pre "egraph==..."` or stable pin), `importlib.metadata.version("egraph")`, and Track A/B + `evaluate_metrics` execution.
- [ ] Verify `npm view egraph dist-tags --json` (`next` for prereleases, `latest` for GA), clean `npm install`, and Node ESM smoke execution.
- [ ] Verify GitHub Release assets (`.whl`, `.tar.gz`, `.tgz`) and clean local working tree.
