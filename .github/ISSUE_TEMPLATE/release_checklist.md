---
name: Release Checklist
about: Step-by-step release checklist for publishing v6 prereleases or GA tags to crates.io, PyPI, and npm via OIDC.
title: "release: v6.X.Y"
labels:
  - release
---

## Pre-Release Preparation
- [ ] Update version across all active manifests, lockfiles, docs, notebooks, and templates (`Cargo.toml` version + 24 `[workspace.dependencies]` pins, `crates/python/Cargo.toml`, `crates/wasm/Cargo.toml`, `crates/wasm/package.json`, `crates/wasm/package-lock.json`, root `package-lock.json`, `examples/web/package.json`, `CHANGELOG.md`, `README.md`, `crates/egraph/README.md`, `crates/python/README.md`, `crates/python/docs/source/getting_started/{installation,migration_v6}.rst`, `docs/MIGRATION_V6.md`, `notebooks/README.md`, `notebooks/01..09_*.ipynb`, `.github/ISSUE_TEMPLATE/bug_report.yml`, `.agents/AGENTS.md`).
- [ ] For `beta` / `rc` / GA releases, update classifiers in `crates/python/pyproject.toml` (remove `"Development Status :: 3 - Alpha"`).
- [ ] Confirm `crates/wasm/package.json` `"repository"` block (`url`, `directory: "crates/wasm"`) for npm `--provenance`.
- [ ] Verify 24-crate crates.io packaging locally:
  - `python3 scripts/verify_helpers.py check-packaging . --strict-metadata`
  - `rm -rf target/package && cargo generate-lockfile && cargo publish --workspace --dry-run --locked`
- [ ] Simulate `.github/workflows/release.yml` `verify-tag-and-version` parser locally with `GITHUB_REF_NAME=v6.X.Y...`.
- [ ] Run `EGRAPH_PARALLEL_WORKERS=1 ./scripts/verify.sh --all --build-wasm --docs --strict` (`hard_fails=0`, Sphinx `doctest`, TypeDoc, Vite build) and `node crates/wasm/tests/playwright_smoke.mjs`.
- [ ] Confirm `jp_cjk=0` across all modified files and clean git status.

## Initial crates.io Bootstrap & OIDC Prerequisites (One-Time Setup for New Crates)
- [ ] Bootstrap all 24 publishable crates (`egraph-core` -> ... -> `egraph-rs`) via `CARGO_REGISTRY_TOKEN=... python3 scripts/bootstrap_crates_io.py --locked` (respecting the 5-crate new-crate burst + 10-minute cooldown).
- [ ] Configure crates.io Trusted Publishers (`github.com/likr/egraph-rs`, workflow `release.yml`, environment `crates-release`) on all 24 crates.
- [ ] Confirm `crates-release`, `pypi-release`, and `npm-release` GitHub Environments (`v6.*` tag deployment rule) and PyPI/npm Trusted Publishers (`release.yml`) are configured.

## Tag & Publish
- [ ] Commit (`chore(release): bump version to ...`), push `origin/master`, and wait for `.github/workflows/ci.yml` (`6/6` jobs including `cargo publish --workspace --dry-run --locked`) green.
- [ ] Confirm `>= 30` minutes have elapsed since any previous 24-crate release tag (so crates.io's 30-crate existing-version burst bucket has refilled inside the 30-minute OIDC token window).
- [ ] Create and push annotated tag: `git tag -a v6.X.Y... -m "Release v6.X.Y..." && git push origin v6.X.Y...` (never force-move failed tags; bump to next patch/prerelease instead).
- [ ] Monitor `.github/workflows/release.yml` (`gh run watch <id> --exit-status`) through all 12 jobs (`verify-tag-and-version`, `build-wheels`, `build-sdist`, `build-wasm`, `publish-crates`, `publish-pypi`, `publish-npm`, `github-release`). On partial `publish-crates` failure inside the 30-min window, retry via `gh run rerun <id> --failed` (resumes idempotently).
- [ ] If yanking a superseded prerelease version, run `cargo yank --version <ver> <crate>` (no `--crate` flag) in reverse topological order (`egraph-rs` -> ... -> `egraph-core`) with `>= 1s` spacing.

## Live Registry Clean-Env Smoke Tests
- [ ] Verify crates.io install in a fresh binary crate (`cargo new` + `egraph-rs = "=..."`, `use egraph::prelude::*;`, `cargo run --release`).
- [ ] Verify PyPI install in fresh venv (`pip install --pre "egraph==..."` or stable pin), `importlib.metadata.version("egraph")`, and Track A/B + `evaluate_metrics` execution.
- [ ] Verify `npm view egraph dist-tags --json` (`next` for prereleases, `latest` for GA), clean `npm install`, and Node ESM smoke execution.
- [ ] Verify GitHub Release assets (`.whl`, `.tar.gz`, `.tgz`) and clean local working tree.
