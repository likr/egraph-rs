---
name: egraph-rs-parallel-dev
description: Playbook for parallel multi-agent and git worktree development in egraph-rs, covering worktree isolation, 12-core/24-GiB resource budgets, Rust-core-first two-stage parallel waves, domain-scoped disjoint crate ownership, and author-verifier handoffs.
---

# Parallel Multi-Agent & Git Worktree Development Playbook

Use this skill whenever coordinating or executing concurrent work across multiple crates, bindings, or milestones in `egraph-rs`.

## 1. Worktree Isolation Modes

1. **Swarm / Subagent Tool Isolation (Preferred for Agent Tasks)**:
   - Use `swarm{action:"add", workspace:"share", ...}` or `invoke_subagent{Workspace:"share", ...}` for authors that edit or build simultaneously in a single-repo git workspace.
   - In multi-root sessions that include non-git directories (such as `/Users/likr/papers`), `workspace:"share"` can fail across non-git roots; spawn authors with `workspace:"inherit"` and assign each author an isolated path created sequentially via `./scripts/worktree.sh create <slug> master`.
   - Leave read-only investigators and independent verifiers on `workspace:"inherit"` (or run verifier checks inside the author's dedicated worktree path).
   - Never run two write/build agents in the same directory.
2. **Manual / CLI Worktree Isolation (`scripts/worktree.sh`)**:
   - **Pre-Flight Dirty-Root Gate**: Always run `git status --short` in the main checkout *before* starting Milestone 1 edits. If any unrelated dirty or untracked files exist (even for a single-milestone author), create and enter an isolated worktree immediately (`./scripts/worktree.sh create <slug> master`) rather than editing the dirty root and migrating mid-milestone.
   - After adding or removing workspace crates on the integration branch, refresh the gitignored root `Cargo.lock` (`cargo metadata --format-version 1 >/dev/null`) before creating downstream worktrees so every worktree inherits an already-resolved `Cargo.lock`.
   - Create an isolated worktree + branch `wt/<slug>` + local `.venv`:
     ```bash
     ./scripts/worktree.sh create <slug> [base_ref]
     ```
   - Worktrees live under `.worktrees/<slug>` (gitignored).
   - Verify inside that worktree without touching main checkout:
     ```bash
     EGRAPH_PARALLEL_WORKERS=3 ./scripts/worktree.sh verify <slug> --fast --json-summary <path>
     ```
   - After merging/cherry-picking the committed branch into the integration branch, confirm clean worktree status and clean up (`--delete-branch` for fast-forward merges, `--delete-branch --force` for cherry-picked branches after confirming all commits landed):
     ```bash
     ./scripts/worktree.sh clean <slug> --delete-branch
     # or for cherry-picked branches after verifying commits landed and worktree is clean:
     ./scripts/worktree.sh clean <slug> --delete-branch --force
     ```
   - **Dirty-Root Direct Worktree Push & Cleanup**: When the main checkout must remain untouched due to pre-existing external dirty work, push verified commits directly from the clean worktree without touching root `HEAD` or running `merge --ff-only` in root: confirm `git rev-parse origin/master` is an ancestor of `wt/<slug>`, run `git push origin wt/<slug>:master` inside `.worktrees/<slug>`, verify `git ls-remote origin refs/heads/master`, and immediately clean `.worktrees/<slug>` (`./scripts/worktree.sh clean <slug> --delete-branch --force` once confirmed clean and pushed; note that the local root branch ref remains behind until the external worktree clears).

## 2. Host Resource & Build Cache Discipline (12 Cores / 24 GiB RAM)

- **Shared `sccache`**: `scripts/verify.sh` automatically sets `RUSTC_WRAPPER=sccache` when installed. Never override `RUSTC_WRAPPER` to empty during parallel runs; `sccache` shares compiled crate artifacts across all worktrees while each worktree keeps its own `target/` lock-free.
- **Worktree-Local `.venv`**: Every worktree uses `$REPO_ROOT/.venv` (created via `scripts/setup_venv.sh` using `uv` hardlinks). **NEVER** install development wheels into `/Users/likr/diffusion-kernel-experiments/.venv` while working inside parallel `egraph-rs` worktrees; only install into `diffusion-kernel-experiments/.venv` from the integrated checkout when running paired benchmark switches.
- **Immutable Wheel Handoff & SHA256 Manifest**: When exporting a built release wheel from a worktree to downstream `diffusion-kernel-experiments` benchmarks, archive the `.whl` in a dedicated immutable path, record both the `.whl` SHA256 and installed `_core.abi3.so` SHA256 in the milestone handoff, and verify downstream `.venv` has not been overwritten by concurrent sessions before benchmarking.
- **CPU & RAM Throttling (`EGRAPH_PARALLEL_WORKERS`)**: Set `EGRAPH_PARALLEL_WORKERS` to the number of concurrent building agents so `scripts/verify.sh` caps `CARGO_BUILD_JOBS` cleanly:

| `EGRAPH_PARALLEL_WORKERS` | `CARGO_BUILD_JOBS` per worker | Total peak Rust jobs | Recommended Use Case |
|---:|---:|---:|---|
| `1` (default) | `8` | `8` | Single agent / final `--all` verification |
| `2` | `5` | `10` | Two concurrent crate/binding authors |
| `3` | `4` | `12` | Three concurrent crate authors (standard M1–M4 fan-out) |
| `4+` | `2` | `8–10` | Four authors (stay well below 24 GiB RAM during LTO/PyO3 builds) |

- **Single-Thread Runtime Enforcement**: `verify.sh` exports `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, and `RUST_FONTCONFIG_DLOPEN=1`.

## 3. Rust-Core-First Two-Stage Sharding & File Ownership Rules

### Stage Ordering Rule (Rust Core Before Bindings)
- **Never run Rust core implementation lanes and Python/WASM binding lanes concurrently for the same evolving API.**
- **Stage 1 (Rust Core Wave)**: Finalize shared trait/core types first, fan out parallel authors across **disjoint, fine-grained Rust core crates** (keeping files `<= ~500` lines and naming strictly by domain), integrate, and pass `./scripts/verify.sh --rust --invariants`.
- **Stage 2 (Binding Wave)**: Only after Stage 1 passes and Rust public APIs are stable, lock the cross-target signature contract and fan out `crates/python` and `crates/wasm` (plus `crates/cli`/docs) in parallel, followed by integrated `--python --wasm --parity --docs` verification.

### Shared-File Serialization Rule
Two parallel authors **MUST NEVER** edit the same file in the same wave. Treat the following as **coordinator/serialized integration points** (locked by one designated owner per step, merged before downstream fan-out):
- Root `Cargo.toml`, `Cargo.lock`, `Makefile`, `.github/workflows/*`
- `crates/egraph-core/src/**`, `crates/egraph-ffi/src/**`
- Top-level registry files (`crates/python/src/lib.rs`, `crates/wasm/src/lib.rs`, `crates/python/python/egraph/_core.pyi`, `crates/wasm/pkg/index.d.ts`)

### Cross-Lane API Signature Lock
Before launching parallel Python + WASM binding waves (or any paired cross-target implementation), pin the exact Rust, Python (`snake_case`), and WASM/TypeScript (`camelCase`) class names, method vs property/getter distinction, argument order, default parameter/seed/dtype contracts (e.g., `seed=42`, `target_std=None` vs `null`/`undefined`, `SpatialRefinerConfig` `--hinge-fp --w-fn 0.0` defaults, `TsNetConfig::pure_2hop`, `ManifoldLayout2d` manifold constructors/methods, zero-`serde` WASM wrapper classes), `IterativeSolver::step()` / `run_steps(k)` / `run()` return semantics (including bit-exact `step == run` equivalence across `f64`/`f32`), return shapes, and error/validation behavior on the board first. Locking these interfaces upfront prevents cross-target drift and eliminates late rework. In `parity_harness.py`, compare k-NN neighbor sets using row-wise sorted `(col, dist)` pairs when embedding distances can tie up to 1 ULP across FMA/libm targets.

### Cross-Target Integration & CI Hygiene
- **CI Crate-List & Feature-Matrix Parity**: Whenever adding or splitting workspace crates (such as individual domain-scoped core crates, `egraph` facade, and `egraph-cli`), immediately add them to all `.github/workflows/ci.yml` gates (`--all-features` clippy, `--release --all-features` tests, and the strict `getrandom`-free `cargo tree` loop) alongside the workspace-wide `cargo tree --workspace --edges normal -e features` banned-dependency check (`rayon|linfa|linfa-nn|linfa-clustering|wgpu`). When a crate exposes optional features with distinct code paths (such as `egraph-layout` default `delaunay` vs `--no-default-features` `StaticKdTree` fallback, `egraph-spatial` default vs `--features delaunay`, `egraph-dataset` default `petgraph` vs `--no-default-features`, `egraph` default/`--all-features`/`--no-default-features`, and `egraph-wasm` default `full = ["delaunay"]` without `legacy` vs `--features legacy`), verify all feature configurations in `clippy`, release tests, and `check_invariants.sh`.
- **`wasm_test_minimal` & Browser Parity**: Name every new WASM integration test by domain (`crates/wasm/tests/<domain>.rs`, never phase/milestone prefixes like `v6_m*` or `v6_w*`). Whenever adding a feature-ungated WASM integration test, register `[[test]]` in `crates/wasm/Cargo.toml` and add `--test <name>` to both `scripts/verify.sh` (`wasm_test_minimal`) and `.github/workflows/ci.yml` (`wasm-pipeline` `--no-default-features` `wasm-pack test --node`), alongside `--features legacy` coverage, `node crates/wasm/tests/node_esm_smoke.mjs`, `js/examples` Vite build, and `crates/wasm/tests/playwright_smoke.mjs` (keep legacy `v6_*` entries registered until renamed in a dedicated cleanup pass). Always invoke `wasm-pack build` from `crates/wasm` with CWD `.` (`(cd crates/wasm && wasm-pack build --target web --no-opt --profile wasm-release -d pkg .)`) for `0.14+`/`0.15.1` compatibility, and pin `wasm-pack v0.15.1` in CI/release workflows (`rustwasm` `init.sh` installs `v0.13.1`, which lacks `--profile` and forwards it to `cargo build`).
- **Workspace Dependency & Release Pipeline Alignment**:
  - Keep `ndarray = { version = "0.17", default-features = false, features = ["std"] }` aligned with `numpy = "0.28"` / `pyo3 = "0.28"` so `maturin sdist` and `cargo build` resolve a single `ndarray` version without pulling `rayon`.
  - Use `macos-15` (not retired `macos-14`) for macOS CI/release runners, `ubuntu-24.04-arm` for native Linux ARM64 wheels, `"repository": {"type": "git", "url": "git+https://github.com/likr/egraph-rs.git", "directory": "crates/wasm"}` in `crates/wasm/package.json` for `npm publish --provenance`, and enforce the `.github/workflows/release.yml` non-alpha guard rejecting `Development Status :: 3 - Alpha` in `crates/python/pyproject.toml` on `beta`/`rc`/GA tags.
  - Keep release-build timing tests tolerant of shared CI VM jitter (e.g., `ForceAtlas2` `N=5000` `<= 25 ms/step`) while enforcing strict `<= 10 ms/frame` interactive budgets in Playwright (`crates/wasm/tests/playwright_smoke.mjs`).
- **Binding Type, Stubtest, Docs & Parity Lock**: Keep `mypy --strict crates/python/python/egraph crates/python/tests/test_v6_*.py` and `stubtest egraph` (with `scipy`, `networkx`, and `igraph` installed for `egraph.interop`) enforced in both `scripts/verify.sh --python` and `.github/workflows/ci.yml`, keep Sphinx HTML + `doctest` and TypeDoc (`verify.sh --docs` and CI `docs-check`) unconditional, and verify fixtures across both `f64` and `f32` without altering core solver invariants to match brittle synthetic inputs.
- **Cross-Precision `f32` ULP & Convergence Guards**: In geometric/projection solvers (`PrismOverlapRemoval`, `project_rectangle_no_overlap_2d`), scale ULP clearance margins to coordinate magnitude (`coord_ulp = |coord| * S::EPSILON`) so `f32` layouts at large coordinate offsets do not oscillate below `f32` precision, and provide a deterministic stall/max-iteration fallback (e.g., 2D scanline QPSC projection) for extreme aspect ratios.
- **Canonical Coarse CSR Symmetry**: When building coarsened graphs (`YifanHuSolver` `EC+MIVS`), accumulate undirected coarse edge weights once in canonical `c < cv` order and mirror `(c, cv)` / `(cv, c)` entries so both directions receive bit-identical `f64` weights without floating-point non-associativity drift.

### Two-Stage Disjoint Ownership Template

| Stage | Serialized First (Coordinator / Core Owner) | Parallel Fan-Out Lanes (Disjoint Crates/Modules) | Independent Verifier Gate |
|---|---|---|---|
| **Stage 1: Rust Core** | Lock shared traits/types in `egraph-core` / root `Cargo.toml` (refresh `Cargo.lock` after crate splits) | Disjoint domain-scoped Rust core crates (e.g., spatial / linalg / neighborhood / individual layout-family crates / clustering / metrics), keeping files `<= ~500` lines and domain-named | Per-lane `--fast --crate <name>`, then integrated `./scripts/verify.sh --rust --invariants` |
| **Stage 2: Bindings & Docs** | Lock cross-target Rust / Python (`snake_case`) / WASM (`camelCase`) signature contract + shared registry files (`lib.rs`, `_core.pyi`, `index.d.ts`) | Lane 1: `crates/python` (domain-named binding/test modules)<br>Lane 2: `crates/wasm` + `js/examples` (domain-named binding/test modules)<br>Lane 3 (optional): `crates/cli` + Sphinx / TypeDoc | `./scripts/verify.sh --python`, `--wasm --build-wasm`, `--parity`, `--docs` (and Playwright smoke for browser changes) |

## 4. Author <-> Verifier Handoff Contract

1. **Author completion**:
   - Commit inside your own worktree/branch following `egraph-rs-commit`.
   - Run `./scripts/verify.sh --fast --json-summary <scratch>/author_<name>.json` (`--fast` stays debug for quick crate loops; `--rust` and `--python` run in release mode by default, with `--debug-rust` and `--dev-py` available to opt into debug builds).
   - **Author Pre-Gate Adversarial Self-Check**: When modifying geometric, coarsening, layout, or neighbor-selection algorithms, write and run fast unit/stress probes inside the crate covering disconnected components, asymmetric star+chain topologies, large coarse graphs (`N >= 300`), `--no-default-features` (`StaticKdTree` fallback), `f32` precision, and English-only comment hygiene (`jp_cjk=0`) *before* handing off to Challenger/Verifier so avoidable multi-iteration rejections are caught upfront.
   - Post `@verifier [done] branch=<branch> worktree=<path> summary=<json_path>`.
2. **Verifier audit**:
   - Inspect diff (`git --no-pager diff <base>..HEAD`).
   - Run `./scripts/verify.sh --rust --invariants` (release Rust tests by default; add `--debug-rust`/`--dev-py` only when debugging, and `--python`/`--wasm`/`--parity` for binding/integration milestones).
   - Report itemized defects directly to the `@author` or `@coordinator [done] APPROVED` when clean.
3. **Coordinator integration**:
   - Cherry-pick or merge approved branches sequentially into the integration branch.
   - If workspace crates or dependencies changed, refresh root `Cargo.lock` via `cargo metadata --format-version 1 >/dev/null` before opening the next wave's worktrees.
   - Run integrated `./scripts/verify.sh --all --build-wasm` before opening the next milestone wave.
   - Clean merged worktrees with `./scripts/worktree.sh clean <slug> --delete-branch` (or pass `--force` only after confirming clean worktree status and verifying cherry-picked commits landed on the integration branch).
4. **Long-running command heartbeat (Swarm Rule)**:
   - Workers MUST post a brief `@coordinator [interim] <phase>` update on the swarm message board before starting any command expected to take `>60s` (e.g., `--rust`, `--python`, `--wasm`, `--all`, full DKE benchmark runs, or long builds) so `monitor` does not emit false `[stall]` alerts.
5. **Zero-Diff Fast-Forward Gate Rule**:
   - When an integration step is a pure fast-forward of an already-verified commit (`git diff <verified_sha>..HEAD` is empty and working tree is clean), do NOT repeat identical full `--all` verification waves. Re-run only checks not previously executed on that SHA (e.g., Playwright smoke or cross-repo integration).
