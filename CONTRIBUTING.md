# Contributing to `egraph-rs`

Thank you for contributing to `egraph-rs`. This project targets **Native Rust**, **Python (`egraph` via PyO3 `abi3-py310`)**, and **WebAssembly (`egraph` via `wasm-bindgen` pure ESM)** equally as first-class targets.

## Non-Negotiable Engineering Invariants

1. **Strictly Single-Threaded Pure CPU Rust**:
   - Do not add `rayon`, `std::thread`, Web Workers, `SharedArrayBuffer`, CUDA/WebGPU (`wgpu`), or C/Fortran FFI dependencies (`ndarray` always sets `default-features = false`).
   - Run verification and benchmarks under single-thread environment caps (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`).
2. **Deterministic Execution & Cooperative Stepping**:
   - Use explicit `u64` seeds (`DeterministicRng::from_seed_u64`), never ambient entropy (`thread_rng()`, `from_entropy()`).
   - Keep solvers free of non-test `.unwrap()`, `.expect()`, `panic!`, `unreachable!`, and `unsafe`.
   - Expose cooperative `step()` / `run_steps(k)` / `run()` for iterative algorithms so browser `requestAnimationFrame` remains non-blocking.
3. **3-Target Parity & Minimal WASM Hygiene**:
   - Align public features across Rust, Python (`_core.pyi` + `mypy --strict` + `stubtest`), and WASM (pure ESM, no `serde`/`getrandom` in default/minimal WASM).
4. **English Repository Hygiene**:
   - All source code, code comments, commit messages, documentation, GitHub Issues, and Pull Requests must be written in English (`jp_cjk=0`).

## Git Branch, Commit, and Push Rules

- **Branches**: Use descriptive topic branches (`feat/<slug>`, `fix/<slug>`, `perf/<slug>`, `docs/<slug>`, `chore/<slug>`). Parallel worktree development creates `wt/<slug>` branches under `.worktrees/<slug>` via `./scripts/worktree.sh create <slug>`.
- **Commit Messages**: Follow **Conventional Commits** (`<type>(<scope>): <description>`, e.g. `feat(egraph-layout): ...`, `fix(egraph-python): ...`, `chore: ...`). Keep each commit focused on a single logical change.
- **No Unwanted Files or Large Binaries**: Never commit secrets, virtual environments (`.venv/`), build artifacts (`target/`, `crates/wasm/pkg/`, `examples/web/dist/`), `Cargo.lock` (gitignored), or files exceeding 100 MB.
- **Immutable History & Release Tags**: Never force-push `master` (`git push --force`) and never force-move or delete pushed release tags (`v6.*`). If a release workflow fails before publishing, fix `master` and bump to the next prerelease/patch version.
- **Verify Before Pushing**: Run [scripts/verify.sh](scripts/verify.sh) matching your scope before pushing or opening a PR.

## Local Verification Commands

Run `./scripts/setup_venv.sh` once to provision the isolated Python `.venv`, then verify via [scripts/verify.sh](scripts/verify.sh):

```bash
# Fast crate-level feedback (debug fmt, clippy -D warnings, nextest)
./scripts/verify.sh --fast --crate egraph-layout

# Full Rust release checks (all feature matrices)
./scripts/verify.sh --rust

# Python wheel build, unittest, mypy --strict, and stubtest
./scripts/verify.sh --python

# WASM check, Node tests (default / --no-default-features), and release bundle build
./scripts/verify.sh --wasm --build-wasm

# Invariants, cross-target parity, and documentation (Sphinx + doctest + TypeDoc + examples/web)
./scripts/verify.sh --all --build-wasm --docs
```

For architectural background and v6 API changes, see [docs/V6_API_FREEZE_SPEC.md](docs/V6_API_FREEZE_SPEC.md), [docs/archive/V6_RELEASE_PLAN.md](docs/archive/V6_RELEASE_PLAN.md), and [docs/MIGRATION_V6.md](docs/MIGRATION_V6.md).

## Issues & Pull Requests

- Use [Bug Report](.github/ISSUE_TEMPLATE/bug_report.yml) or [Feature Request](.github/ISSUE_TEMPLATE/feature_request.yml) templates when opening issues, and include explicit `seed` values and single-thread settings for reproductions.
- Fill out [the Pull Request template](.github/pull_request_template.md), reference related issues, list exact verification commands run, and wait for `.github/workflows/ci.yml` (`6/6` jobs) to pass before merging.
