---
name: egraph-rs-release
description: Release runbook for bumping egraph-rs versions, verifying tag and OIDC alignment, bootstrapping and publishing crates.io/PyPI/npm releases without polling, recovering from failed tags, and smoke-testing live registries.
---

# Release and Live-Registry Verification Runbook for `egraph-rs`

Use this skill whenever preparing, bumping, tagging, publishing, or verifying a prerelease or GA release (`v6.*`) of `egraph-rs` to crates.io (`egraph-rs`), PyPI (`egraph`), and npm (`egraph`).

## 1. Complete Version-Bump File Checklist

> Note: Feature, bugfix, and experiment commits/pushes should land per verified stage (`egraph-rs-github`). This runbook governs only the final version-bump commit, annotated `v6.*` tag, CI/OIDC publishing, and live-registry verification.

Update every file below atomically when bumping `X.Y.Z-alpha.N` / `X.Y.Z-beta.N` / `X.Y.Z-rc.N` / `X.Y.Z` (PEP 440 PyPI equivalent: `X.Y.ZaN` / `X.Y.ZbN` / `X.Y.ZrcN` / `X.Y.Z`):

1. `Cargo.toml` (`[workspace.package].version` AND all 24 internal publishable `egraph*` version pins under `[workspace.dependencies]`, including `egraph-rs`, enforced by `scripts/verify_helpers.py` / `scripts/packaging_checks.py`)
2. `crates/python/Cargo.toml` (`package.version`)
3. `crates/wasm/Cargo.toml` (`package.version`)
4. `crates/wasm/package.json` (`"version"`)
5. **`crates/wasm/package-lock.json` (`"version"` at root AND `packages[""].version`)** — tracked in git; omitting this causes CI `docs-check` (`npm ci --prefix crates/wasm`) or lockfile drift failures.
6. `examples/web/package.json` (`"version"`)
7. **Root `package-lock.json` (`packages["crates/wasm"].version` AND `packages["examples/web"].version`)** — npm workspace lockfile at repository root; keep aligned with `crates/wasm/package.json` and `examples/web/package.json`.
8. `CHANGELOG.md` (Keep a Changelog section matching the release version and date)
9. `README.md` (header, Python Quick Start version/pip pin, Rust Quick Start `cargo add egraph-rs@...`)
10. `crates/egraph/README.md` (`egraph-rs = "<version>"` Quick Start snippet)
11. `crates/python/README.md` (`Version:` header and `pip install` examples)
12. `crates/python/docs/source/getting_started/installation.rst` (`pip install` version pins)
13. `crates/python/docs/source/getting_started/migration_v6.rst` (version mention)
14. `docs/MIGRATION_V6.md` (title and summary table header)
15. `notebooks/README.md` and `notebooks/01..09_*.ipynb` (Colab `!pip install` version pins)
16. `.github/ISSUE_TEMPLATE/bug_report.yml` (placeholder/description version example)
17. `.agents/AGENTS.md` (active release version summary)

For `beta`, `rc`, or GA tags (`pre_kind != "alpha"`), also replace `"Development Status :: 3 - Alpha"` in `crates/python/pyproject.toml` (`classifiers`)—for example with `"Development Status :: 4 - Beta"` on beta tags (or `"Development Status :: 5 - Production/Stable"` for GA); `.github/workflows/release.yml` explicitly fails if an alpha classifier remains on non-alpha tags. Keep `crates/wasm/package.json` `"repository": {"type": "git", "url": "git+https://github.com/likr/egraph-rs.git", "directory": "crates/wasm"}` intact for `npm publish --provenance`.

## 2. Pre-Tag Verification & Hygiene

Run single-threaded before committing or tagging:

1. Refresh local `Cargo.lock`: `cargo generate-lockfile`
2. Verify all 24 publishable crates metadata & tarballs:
   - `python3 scripts/verify_helpers.py check-packaging . --strict-metadata`
   - `rm -rf target/package && cargo generate-lockfile && cargo publish --workspace --dry-run --locked`
3. Simulate `.github/workflows/release.yml`'s `verify-tag-and-version` parser with `GITHUB_REF_NAME=v<tag_version>` against `Cargo.toml`, all workspace member `Cargo.toml` files, `crates/wasm/package.json`, and `crates/python/pyproject.toml`.
4. Run:
   ```bash
   EGRAPH_PARALLEL_WORKERS=1 ./scripts/verify.sh --invariants --docs --strict
   ```
   Confirm `Cross-target packaging metadata is aligned` (all 28 crates: `24` publishable library crates including `egraph-rs` + `4` `publish = false` crates `egraph-ffi`, `egraph-python`, `egraph-wasm`, `egraph-cli`), `hard_fails=0`, Sphinx HTML + `doctest`, TypeDoc, and `npm --prefix examples/web run build` all pass, and `jp_cjk=0` across all modified files.
   - **Version-Bump-Only Scope Guard**: If the release bump commit changes only the version/lockfile/changelog/doc files (verified via `git diff --stat <verified_parent>..HEAD` touching zero `.rs`/`.py`/`.mjs` logic files), `--invariants --docs` + `npx tsc --noEmit` + local clean-env `maturin build`/`sdist`/`npm pack` + `cargo publish --workspace --dry-run --locked` smoke is sufficient prior to pushing `origin/master`; do not duplicate the parent commit's full Rust/Python/WASM test suite locally because `ci.yml` runs the full matrix remotely before tagging.
   - **Distribution Build Profile**: `.github/workflows/release.yml` builds published PyPI wheels with `maturin build --profile dist-release` (`lto = "thin"`, `codegen-units = 1`), distinct from the fast parallel `[profile.release]` (`lto = false`) used by `ci.yml` and local `verify.sh --rust`/`--python` checks.
5. Commit (`chore(release): bump version to <version>`), push `origin/master`, and wait for `.github/workflows/ci.yml` (`6/6` jobs, including `cargo publish --workspace --dry-run --locked`) to complete green before creating a release tag.

## 3. Initial crates.io Bootstrap & OIDC Prerequisites

- **Why Bootstrap Precedes OIDC**: crates.io Trusted Publishing (`rust-lang/crates-io-auth-action@v1`) cannot register brand-new crates. Every one of the 24 publishable crates (`egraph-core` -> ... -> `egraph-rs`) must first be published once with a personal `CARGO_REGISTRY_TOKEN`, and then configured on crates.io under **Settings -> Trusted Publishing** with owner `likr`, repository `egraph-rs`, workflow `release.yml`, and environment `crates-release`.
- **Rate-Limit-Aware 24-Crate Bootstrapper (`scripts/bootstrap_crates_io.py`)**:
  - Brand-new crates are rate-limited by crates.io to a burst of **5** creations followed by **1 every 10 minutes** (~190 minutes total for 24 new crates). Existing crates allow a burst of **30** versions followed by **1/minute**.
  - Preview the 7-tier topological plan with `python3 scripts/bootstrap_crates_io.py --dry-run`.
  - Run the resumable, sparse-index-polling bootstrapper once per new crate family:
    ```bash
    CARGO_REGISTRY_TOKEN="<crates-io-token>" python3 scripts/bootstrap_crates_io.py --locked
    ```
  - It skips already-published `<name>/<version>` pairs automatically and retries `429` responses safely.
- **Trusted Publishing Environments (No Long-Lived API Tokens in CI)**: `.github/workflows/release.yml` publishes via OIDC using GitHub Environments `crates-release`, `pypi-release`, and `npm-release` (deployment tag rule `v6.*`) and matching Trusted Publishers configured on crates.io, PyPI, and npmjs.com with workflow filename `release.yml`.
- **Permission Boundary**: Standard developer PATs typically return HTTP 403 on repository Environments APIs. Never attempt to bypass permission checks; if `crates-release` / `pypi-release` / `npm-release` or Trusted Publishers are missing, stop and ask the repository owner to provision them in GitHub / crates.io / PyPI / npm web settings.
- **Failed Tag Policy (`docs/archive/V6_RELEASE_PLAN.md` §8.2)**: Never force-move (`git tag -f` / `git push --force`) an already-pushed release tag after a workflow failure. Fix `master`, verify green remote CI, bump to the next version (e.g., `v6.0.0-beta.3`), and push a new annotated tag (`git tag -a v<version> -m "Release v<version>" && git push origin v<version>`).

## 4. Non-Polling GitHub Actions Monitoring

Never poll `gh run view` in a `sleep` loop. Locate the triggered run ID via `gh run list --workflow <ci|release>.yml --limit 3 --json databaseId,headSha,headBranch,status,url`, then launch:

```bash
gh run watch <run_id> --exit-status && gh run view <run_id> --json status,conclusion,jobs --jq '{status, conclusion, jobs: [.jobs[] | {name, status, conclusion}]}'
```

in the background (`WaitMsBeforeAsync: 500`) and wait for task completion notification. If `publish-crates` / `publish-pypi` / `publish-npm` pauses in `waiting` state for manual Environment approval, inform the user immediately with the run URL.

## 5. Clean-Environment Live Registry Smoke Checks

After `release.yml` completes `success` across all 12 jobs:

### crates.io (`cargo add egraph-rs`)
- Create a fresh standalone crate (`cargo new /tmp/egraph_rs_smoke && cd /tmp/egraph_rs_smoke`), add `egraph-rs = "=<tag_version>"` (`[lib] name = "egraph"`), write a `main.rs` using `use egraph::prelude::*;` (`CsrGraph::<f64>::from_unweighted_edges`, `PivotMds`, `TsNetSolver::<Layout2d<f64>>::from_graph_pure_2hop`, `evaluate_metrics`), and run `cargo run --release`.
- Confirm `cargo info egraph-rs` (or crates.io API) shows `<tag_version>` and docs.rs builds cleanly.

### PyPI (`pip install --pre`)
- Install into a fresh `python3 -m venv` with `PIP_NO_CACHE_DIR=1 pip install --pre "egraph==<pep440>"` (add `--only-binary=:all:` when testing wheel-only installs without Rust).
- **Important v6 Python API facts**:
  - Check installed version with `import importlib.metadata as md, egraph as eg; assert eg.__version__ == md.version("egraph") == "<pep440>"` (`egraph.__version__` is exported in PEP 440 form, e.g. `6.0.0b2`, matching `CARGO_PKG_VERSION`, and intentionally omitted from `egraph.__all__`).
  - `eg.SgdSolver.sparse_barrier_dog(...)` returns `SgdSolver` directly (not a tuple).
  - `report = eg.evaluate_metrics(g, lay, ["sns", "np2", "crossing_number"])` accepts short aliases as input and returns a frozen `MetricReport` supporting typed properties (`report.scale_normalized_stress`, `report.neighborhood_preservation_2hop`, `report.crossing_number`), mapping lookup (`report["scale_normalized_stress"]`), and `report.to_dict()`.
  - Smoke-test `Layout3d` (`target="layout3d"` on `ForceAtlas2Solver` / `TsNetSolver`), `EdgePolylines.straight_line_from_graph(g, lay)` + `FdebSolver(polylines)`, and fail-fast `UnsupportedOperationError` on illegal `(target, backend)` pairs (e.g. `TsNetSolver(g, target="layout3d", backend="fit_fft")`).
- **Downstream `uv`-Managed `.venv` Upgrade (`graph-drawing-experiments`)**: Downstream worktrees provisioned via `uv` do not bundle `pip`. Upgrade them to the published release with `uv pip install --python .venv/bin/python --pre --no-cache "egraph==<pep440>"`.

### npm (`npm install egraph@next` / `--tag next` for prereleases)
- Verify `npm view egraph dist-tags --json` shows updated `next` while keeping existing `latest` untouched for prereleases.
- Install into a clean `"type": "module"` directory (`npm install egraph@next`), confirm `node_modules/egraph/package.json` `"version"`, and smoke-run via Node.js ESM (`import init, { ... } from "egraph"`):
  - `evaluateMetrics(g, lay, ["sns", "np2", "crossingNumber"]).toRecord()` returns camelCase fields (`scaleNormalizedStress`, `neighborhoodPreservation2Hop`, `crossingNumber`).
  - Smoke-test `Layout3d.init(n, { seed: 42n })` (`BigInt` seed + `coordsView()`), `EdgePolylines.straightLineFromGraph(g, lay)` + `new FdebSolver(polylines)`, and structured `.code` (`"UnsupportedOperation"`) on invalid `(target, backend)` combinations.

Remove all temporary smoke directories afterwards and verify clean `git status -sb`.
