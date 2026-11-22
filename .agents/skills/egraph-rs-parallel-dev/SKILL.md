---
name: egraph-rs-parallel-dev
description: Playbook for parallel multi-agent and git worktree development in egraph-rs, covering worktree isolation, 12-core/24-GiB resource budgets, M0-M6 disjoint crate ownership, and author-verifier handoffs.
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

## 2. Host Resource & Build Cache Discipline (12 Cores / 24 GiB RAM)

- **Shared `sccache`**: `scripts/verify.sh` automatically sets `RUSTC_WRAPPER=sccache` when installed. Never override `RUSTC_WRAPPER` to empty during parallel runs; `sccache` shares compiled crate artifacts across all worktrees while each worktree keeps its own `target/` lock-free.
- **Worktree-Local `.venv`**: Every worktree uses `$REPO_ROOT/.venv` (created via `scripts/setup_venv.sh` using `uv` hardlinks). **NEVER** install development wheels into `/Users/likr/diffusion-kernel-experiments/.venv` while working inside parallel `egraph-rs` worktrees; only install into `diffusion-kernel-experiments/.venv` from the integrated checkout when running paired benchmark switches.
- **CPU & RAM Throttling (`EGRAPH_PARALLEL_WORKERS`)**: Set `EGRAPH_PARALLEL_WORKERS` to the number of concurrent building agents so `scripts/verify.sh` caps `CARGO_BUILD_JOBS` cleanly:

| `EGRAPH_PARALLEL_WORKERS` | `CARGO_BUILD_JOBS` per worker | Total peak Rust jobs | Recommended Use Case |
|---:|---:|---:|---|
| `1` (default) | `8` | `8` | Single agent / final `--all` verification |
| `2` | `5` | `10` | Two concurrent crate/binding authors |
| `3` | `4` | `12` | Three concurrent crate authors (standard M1–M4 fan-out) |
| `4+` | `2` | `8–10` | Four authors (stay well below 24 GiB RAM during LTO/PyO3 builds) |

- **Single-Thread Runtime Enforcement**: `verify.sh` exports `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, and `RUST_FONTCONFIG_DLOPEN=1`.

## 3. M0–M6 Parallel Sharding & File Ownership Rules

### Shared-File Serialization Rule
Two parallel authors **MUST NEVER** edit the same file in the same wave. Treat the following as **coordinator/serialized integration points** (locked by one designated owner per step, merged before downstream fan-out):
- Root `Cargo.toml`, `Cargo.lock`, `Makefile`, `.github/workflows/*`
- `crates/egraph-core/src/**`, `crates/egraph-ffi/src/**`
- Top-level registry files (`crates/python/src/lib.rs`, `crates/wasm/src/lib.rs`, `crates/python/python/egraph/_core.pyi`, `crates/wasm/pkg/index.d.ts`)

### Disjoint Ownership Matrix by Milestone (`docs/V6_RELEASE_PLAN.md`)

| Milestone | Serialized First (Owner A) | Parallel Fan-Out Lanes (Disjoint Crates/Files) | Independent Verifier Lane |
|---|---|---|---|
| **Step 0 / M0** | Merge `feat/np2-improvement` -> `master`; update root `Cargo.toml` + `egraph-core` | Lane 1: `egraph-petgraph` + `egraph-dataset`<br>Lane 2: `egraph-ffi` + minimal PyO3/WASM harness<br>Lane 3: `.github/workflows/{ci,release}.yml` + `bfs.rs` hotfix | Full `verify.sh --all` + `check_invariants.sh` |
| **M1** | Lock `egraph-core` v6 traits | Lane 1: `egraph-spatial` (`IbFft2d`, `FitField2d`, `ArenaQuadtree2d`, `StaticKdTree`, `SweepLineCrossings`)<br>Lane 2: `egraph-metrics` (`SNS`, `NP1/NP2`, `Gabriel`, `PRISM`/`dRNG`/`dGG`)<br>Lane 3: Python + WASM M1 metric bindings | `verify.sh --fast` per lane, then integrated `--python --wasm --invariants --parity` + `diffusion-kernel-experiments/utils/metrics.py` check |
| **M2** | Lock `egraph-spatial` & `egraph-linalg` public traits | Lane 1: `egraph-linalg` (`ChebyshevWorkspace`, regular DoG, `SubspaceEigen`, `PivotMds`, `MaxMinSp`)<br>Lane 2: `egraph-neighborhood` (`1+2-hop CSR`, adaptive perplexity, rerank k-NN, `fuzzy_simplicial_set`) | Integrator wires M2 Python/WASM bindings -> Verifier runs M2 numerical checks vs SciPy + `diffusion-kernel-experiments` |
| **M3** | Lock `egraph-neighborhood` APIs | Lane 1: `TsNetSolver` (`egraph-layout/src/ts_net/*`)<br>Lane 2: `SgdSolver` + `SparseBarrier` + `ManifoldLayout2d` (`egraph-layout/src/sgd/*`)<br>Lane 3: `GuMapSolver` + `SpatialIntruderRefiner` + `StressMajorization` + `KamadaKawai`<br>Lane 4: `egraph-clustering` (`KMeans++`, `Spectral`, `Louvain`, `LabelProp`, `coarsen`) | Integrator exposes bindings -> Verifier checks `suite30`, `drgraph` Tier 3, `step()` equivalence & WASM |
| **M4** | Lock M3 `FieldLayout` / `SgdSolver` traits | Lane 1: `FieldLayout` extensions (`t-FDP`, `ibUMAP`)<br>Lane 2: `SgdSolver` extensions (`SnapTFdp`, `PaCMAP-Graph`, `(SGD)^2`)<br>Lane 3: `ContinuousFdp` (`ForceAtlas2`, `Yifan Hu`)<br>Lane 4: `PrismOverlapRemoval`, `SeparationConstraints`, `Layering`, `Fdeb` | Verifier checks 1M/4M memory (`<1.6 GiB`), `PRISM` speed/overlap=0, WASM `<10 ms/step` |
| **M5** | Lock `egraph` facade & `egraph-ffi` signatures | Lane 1: `crates/python` (`interop.py`, `_core.pyi`, `mypy`)<br>Lane 2: `crates/wasm` (`pkg/`, `.d.ts`, `runProgressive`) + `js/examples/`<br>Lane 3: `crates/cli` + Sphinx Docs + TypeDoc + `MIGRATION_V6.md` | Verifier runs full `verify.sh --all --build-wasm --strict` + Playwright + `stubtest` + `doctest` |
| **M6** | Freeze `master` for `v6.0.0-alpha.4` | Lane 1: `suite30` / `drgraph` / `4M` regression benchmarks<br>Lane 2: Wheel / `sdist` / `npm pack` clean-env smoke + release notes | Final Go/No-Go sign-off -> tag `v6.0.0-alpha.4` |

## 4. Author <-> Verifier Handoff Contract

1. **Author completion**:
   - Commit inside your own worktree/branch following `egraph-rs-commit`.
   - Run `./scripts/verify.sh --fast --json-summary <scratch>/author_<name>.json` (`--fast` stays debug for quick crate loops; `--rust` and `--python` run in release mode by default, with `--debug-rust` and `--dev-py` available to opt into debug builds).
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
   - Workers MUST post a brief `[interim]` status update on the swarm message board before starting any command expected to take `>60s` (e.g., `--rust`, `--python`, `--wasm`, `--all`, or long builds) so `monitor` does not emit false `[stall]` alerts.
