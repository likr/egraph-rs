---
name: egraph-rs-release
description: Release runbook for bumping egraph-rs versions, verifying tag and OIDC alignment, pushing PyPI/npm release tags without polling, recovering from failed tags, and smoke-testing live registries.
---

# Release and Live-Registry Verification Runbook for `egraph-rs`

Use this skill whenever preparing, bumping, tagging, publishing, or verifying a prerelease or GA release (`v6.*`) of `egraph-rs` to PyPI and npm.

## 1. Complete Version-Bump File Checklist

Update every file below atomically when bumping `X.Y.Z-alpha.N` / `X.Y.Z-beta.N` / `X.Y.Z-rc.N` / `X.Y.Z` (PEP 440 PyPI equivalent: `X.Y.ZaN` / `X.Y.ZbN` / `X.Y.ZrcN` / `X.Y.Z`):

1. `Cargo.toml` (`[workspace.package].version`)
2. `crates/python/Cargo.toml` (`package.version`)
3. `crates/wasm/Cargo.toml` (`package.version`)
4. `crates/wasm/package.json` (`"version"`)
5. **`crates/wasm/package-lock.json` (`"version"` at root AND `packages[""].version`)** — tracked in git; omitting this causes CI `docs-check` (`npm ci --prefix crates/wasm`) or lockfile drift failures.
6. `js/examples/package.json` (`"version"`)
7. `CHANGELOG.md` (Keep a Changelog section matching the release version and date)
8. `crates/python/README.md` (`Version:` header and `pip install` examples)
9. `crates/python/docs/source/getting_started/installation.rst` (`pip install` version pins)
10. `crates/python/docs/source/getting_started/migration_v6.rst` (version mention)
11. `docs/MIGRATION_V6.md` (title and summary table header)

For `beta`, `rc`, or GA tags (`pre_kind != "alpha"`), also remove `"Development Status :: 3 - Alpha"` from `crates/python/pyproject.toml` (`classifiers`) and update it appropriately; `.github/workflows/release.yml` explicitly fails if an alpha classifier remains on non-alpha tags. Keep `crates/wasm/package.json` `"repository": {"type": "git", "url": "git+https://github.com/likr/egraph-rs.git", "directory": "crates/wasm"}` intact for `npm publish --provenance`.

## 2. Pre-Tag Verification & Hygiene

Run single-threaded before committing or tagging:

1. Refresh local `Cargo.lock`: `cargo metadata --format-version 1 >/dev/null`
2. Simulate `.github/workflows/release.yml`'s `verify-tag-and-version` parser with `GITHUB_REF_NAME=v<tag_version>` against `Cargo.toml`, all workspace member `Cargo.toml` files, `crates/wasm/package.json`, and `crates/python/pyproject.toml`.
3. Run:
   ```bash
   EGRAPH_PARALLEL_WORKERS=1 ./scripts/verify.sh --invariants --docs
   ```
   Confirm `Cross-target packaging metadata is aligned`, `hard_fails=0`, `.wasm` gzip `<= 600.0 KiB`, Sphinx HTML + `doctest`, TypeDoc, and `npm --prefix js/examples run build` all pass, and `jp_cjk=0` across all modified files.
   - **Version-Bump-Only Scope Guard**: If the release bump commit changes only the 11 version/changelog/doc files (verified via `git diff --stat <verified_parent>..HEAD` touching zero `.rs`/`.py`/`.mjs` logic files), `--invariants --docs` + `npx tsc --noEmit` + local clean-env `maturin build`/`sdist`/`npm pack` smoke is sufficient prior to pushing `origin/master`; do not duplicate the parent commit's full Rust/Python/WASM test suite locally because `ci.yml` (`6/6`) runs the full matrix remotely before tagging.
4. Commit (`chore(release): bump version to <version>`), push `origin/master`, and wait for `.github/workflows/ci.yml` (`6/6` jobs) to complete green before creating a release tag.

## 3. OIDC Prerequisites & Tag Recovery Policy

- **Trusted Publishing (No Long-Lived API Tokens)**: `.github/workflows/release.yml` publishes via OIDC using GitHub Environments `pypi-release` and `npm-release` (deployment tag rule `v6.*`) and matching Trusted Publishers configured on PyPI and npmjs.com with workflow filename `release.yml`.
- **Permission Boundary**: Standard developer PATs typically return HTTP 403 on repository Environments APIs. Never attempt to bypass permission checks; if `pypi-release` / `npm-release` or Trusted Publishers are missing, stop and ask the repository owner to provision them in GitHub / PyPI / npm web settings.
- **Failed Tag Policy (`docs/V6_RELEASE_PLAN.md` §8.2)**: Never force-move (`git tag -f` / `git push --force`) an already-pushed release tag after a workflow failure (e.g., `v6.0.0-alpha.4` stayed unpublished due to a pre-`fbfd70c` `check_invariants.sh` stderr bug). Fix `master`, verify green remote CI, bump to the next version (`v6.0.0-alpha.5`), and push a new annotated tag (`git tag -a v<version> -m "Release v<version>" && git push origin v<version>`).

## 4. Non-Polling GitHub Actions Monitoring

Never poll `gh run view` in a `sleep` loop. Locate the triggered run ID via `gh run list --workflow <ci|release>.yml --limit 3 --json databaseId,headSha,headBranch,status,url`, then launch:

```bash
gh run watch <run_id> --exit-status && gh run view <run_id> --json status,conclusion,jobs --jq '{status, conclusion, jobs: [.jobs[] | {name, status, conclusion}]}'
```

in the background (`WaitMsBeforeAsync: 500`) and wait for task completion notification. If `publish-pypi` / `publish-npm` pauses in `waiting` state for manual Environment approval, inform the user immediately with the run URL.

## 5. Clean-Environment Live Registry Smoke Checks

After `release.yml` completes `success` across all 11 jobs:

### PyPI (`pip install --pre`)
- Install into a fresh `python3 -m venv` with `PIP_NO_CACHE_DIR=1 pip install --pre "egraph==<pep440>"` (add `--only-binary=:all:` when testing wheel-only installs without Rust).
- **Important v6 Python API facts**:
  - Check installed version with `import importlib.metadata as md; assert md.version("egraph") == "<pep440>"` (`egraph.__version__` is **not** defined).
  - `eg.SgdSolver.sparse_barrier_dog(...)` returns `SgdSolver` directly (not a tuple).
  - `eg.evaluate_metrics(g, lay, ["sns", "np2", "crossing_number"])` accepts short aliases as input, but returned `dict[str, float]` keys are canonical snake_case names: `"scale_normalized_stress"`, `"neighborhood_preservation_2hop"`, `"crossing_number"`.
- **Downstream `uv`-Managed `.venv` Upgrade (`diffusion-kernel-experiments`)**: Downstream worktrees provisioned via `uv` do not bundle `pip`. Upgrade them to the published release with `uv pip install --python .venv/bin/python --pre --no-cache "egraph==<pep440>"`.

### npm (`npm install egraph@next` / `--tag next` for prereleases)
- Verify `npm view egraph dist-tags --json` shows updated `next` while keeping existing `latest` untouched for prereleases.
- Install into a clean `"type": "module"` directory (`npm install egraph@next`), confirm `node_modules/egraph/package.json` `"version"`, verify `node_modules/egraph/pkg/egraph_wasm_bg.wasm` level-9 gzip `<= 614400` bytes (`600 KiB`), and smoke-run via Node.js ESM (`import init, { ... } from "egraph"`):
  - `evaluateMetrics(g, lay, ["sns", "np2", "crossingNumber"]).toRecord()` returns camelCase fields (`scaleNormalizedStress`, `neighborhoodPreservation2Hop`, `crossingNumber`).

Remove all temporary smoke directories under `/tmp` afterwards and verify clean `git status -sb`.
