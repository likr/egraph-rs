# AGENTS.md - `egraph-rs` Operational Knowledge Base & Development Harness Index

## 1. Project Goal & Active Roadmap

`egraph-rs` is a high-performance graph drawing and layout library targeting **Native Rust, Python (PyO3), and WebAssembly (`wasm-bindgen` / TypeScript)** equally as first-class targets.

- **Active Release Plan**: See [`docs/V6_RELEASE_PLAN.md`](../docs/V6_RELEASE_PLAN.md). Architectural redesign (**Step 0 + M0–M6**) shipped in `v6.0.0-alpha.5`, followed by `v6.0.0-alpha.6`, `v6.0.0-alpha.7`, and **`v6.0.0-alpha.8`** (PyPI `6.0.0a8` / npm `6.0.0-alpha.8 --tag next`, adding G-CAF v2 cluster area faithfulness, Phase W1–W2 published graph layout/overlap/cluster baseline suites including native `Leiden`, Noack `LinLog`, `Backbone`, `GroupInABox`, `GTree`, `Forbid`, `HDE`, `MaxentStress`, `GRIP`, `GEM`, `SEL`, `PFS'`, `SFC`, `cluster_baselines`, and removing the `600 KiB` WASM gzip enforcement gate). Subsequent releases (`beta`, `rc`, `v6.0.0` GA, `crates.io`) build cleanly on this baseline.
- **Paired Research & Benchmark Repo**: `/Users/likr/diffusion-kernel-experiments` provides empirically validated algorithms (Track A Pure 1+2-Hop Adaptive Fit-tsNET, Track B $O(|E|)$ Regularized Chebyshev DoG + 3-Tier Sparse Barrier SGD / GuMap + 2D Spatial Intruder Refinement) and regression benchmarks (`suite30`, `tsnet_ref5`, `drgraph` up to 4M nodes).

---

## 2. Non-Negotiable Engineering Invariants

1. **Strictly Single-Threaded Pure Rust**:
   - **NEVER** add `rayon`, `std::thread`, Web Workers, `SharedArrayBuffer`, CUDA/WebGPU, or C/Fortran FFI dependencies.
   - Always configure `ndarray = { ..., default-features = false }` (never pull `rayon` transitively).
   - Enforced statically via [`clippy.toml`](../clippy.toml) and [`scripts/check_invariants.sh`](../scripts/check_invariants.sh).
2. **Deterministic Explicit RNG**:
   - Never call `rand::thread_rng()` or `StdRng::from_entropy()` in core libraries; accept explicit `u64` seeds or `&mut impl RngCore`.
3. **Flat Zero-Copy CSR / SoA & Cooperative Stepping**:
   - Prefer `u32`-indexed `CsrGraph` and contiguous SoA `Layout2d` buffers over `petgraph` + `HashMap` inside inner loops.
   - Keep host objects (`Py<PyAny>`, `JsValue`) outside the Rust core.
   - All iterative algorithms expose cooperative `step()` / `run_steps(k)` execution for non-blocking single-thread browser `requestAnimationFrame` loops.
4. **Rust-Core-First Sequential Workflow**:
   - Always implement, test, and verify Rust core crates first (`./scripts/verify.sh --fast`, `--rust`, `--invariants`) to stabilize the Rust API before implementing or modifying Python (`crates/python`) and WebAssembly (`crates/wasm`) bindings.
   - Once the Rust core is verified and stable, Python and WASM binding work may proceed in parallel against the settled Rust API (`--python`, `--wasm`, `--parity`, `--docs`).
5. **Compile-Time Crate Granularity & File Size Discipline (`<= ~500` Lines)**:
   - Partition Rust core functionality into cohesive, domain-scoped crates with narrow, acyclic dependency edges so incremental edits recompile minimal compilation units rather than monolithic crates.
   - Decompose hand-written source files (`.rs`, `.py`, `.js`/`.mjs`/`.ts`) into cohesive submodules so each file stays within roughly **500 lines**; move large unit test suites into dedicated `tests.rs` submodules or domain-named integration test files. Existing oversized crates/files are refactored in dedicated passes without expanding them further.
6. **Timeless, Snapshot-Clean Naming (No Phase/Milestone Identifiers)**:
   - Name all crates, files, modules, types, functions, test files, test functions, and code comments strictly by **domain, algorithm, or capability**—never by implementation workflow phases, milestones, waves, or lanes (e.g., forbidden: `v6_m*`, `v6_w*`, `m4_lane*`, `phase_*`, `step0_*`, `stage*`).
   - Keep the codebase readable as a self-contained timeless snapshot; record workflow phases/milestones only in external roadmaps, issues, PRs, or changelogs.
7. **Language Rules**:
   - All files under `.agents/` and all code comments (Rust, Python, JS/TS) **MUST be written in English**.
   - All user-facing replies and reports **MUST be written in Japanese**.

---

## 3. Git & GitHub Operational Rules

- **Atomic Conventional Commits**: Split unrelated changes into atomic commits following [`egraph-rs-commit`](skills/egraph-rs-commit/SKILL.md) and [`CONTRIBUTING.md`](../CONTRIBUTING.md). Confirm `git diff --cached --stat` before committing.
- **Autonomous Verified Commits & Stage Pushes**: In single- or multi-milestone work, commit verified changes atomically (staging explicit file paths only so concurrent edits stay untouched) after each passed verification gate (`./scripts/verify.sh`) and push to `origin/master` at logical stage boundaries autonomously.
- **Pre-Completion Push Gate**: Before delivering the final task completion report to the user, ALL verified task commits MUST be pushed to `origin/master` (`git rev-parse HEAD origin/master` aligned).
- **Release & Destructive Action Gate**: Reserve explicit user confirmation for the 11-file version bump, release tags (`v6.*`), PyPI/npm/crates.io publishing, and destructive Git/GitHub operations.
- **History, Tag & Hygiene Guards**: Never force-push `master` (`--force`), never move or delete pushed release tags (`v6.*`), and never commit secrets, ignored build/venv artifacts (`target/`, `.venv/`, `Cargo.lock`, `crates/wasm/pkg/`), or files `> 100 MB`. Verify `./scripts/verify.sh` before pushing.
- **GitHub Issues & PRs**: Write all Issue bodies, PR descriptions, and release notes in English (`jp_cjk=0`) following [`.github/ISSUE_TEMPLATE/`](../.github/ISSUE_TEMPLATE/config.yml) and [`.github/pull_request_template.md`](../.github/pull_request_template.md); confirm CI green before merging or tagging.

---

## 4. Verification & Parallel Development Harness

| Tool / Script | Purpose |
|---|---|
| [`scripts/setup_venv.sh`](../scripts/setup_venv.sh) | Provision isolated worktree-local Python 3.12 `.venv` (`maturin`, `pytest`, `numpy`, `scipy`, `networkx`, `mypy`, `ruff`) via `uv`. |
| [`scripts/worktree.sh`](../scripts/worktree.sh) | Manage parallel git worktrees under `.worktrees/<slug>` (`wt/<slug>` branch + isolated `.venv`), verify inside worktrees, and safely clean up. |
| [`scripts/verify.sh`](../scripts/verify.sh) | Tiered verification runner (`--fast`, `--rust`, `--python`, `--wasm`, `--invariants`, `--parity`, `--docs`, `--all`, `--json-summary`) auto-configuring `sccache`, single-thread env vars, `CARGO_BUILD_JOBS` from `EGRAPH_PARALLEL_WORKERS`, and `nextest`. |
| [`scripts/check_invariants.sh`](../scripts/check_invariants.sh) | Enforce `cargo tree` (`rayon`/`linfa`/`wgpu`=0), static source scan, `wasm32-unknown-unknown` build, and packaging metadata alignment (`--mode report` or `--mode strict`). |
| [`scripts/parity_harness.py`](../scripts/parity_harness.py) | Deterministic golden fixtures, reproducibility, adversarial non-panic input tests, and optional Node.js WASM parity checks. |
| [`clippy.toml`](../clippy.toml) | Blocks `std::thread::{spawn, scope, Builder, JoinHandle}` workspace-wide via Clippy. |

---

## 5. Skills Index

- [egraph-rs-parallel-dev](skills/egraph-rs-parallel-dev/SKILL.md): Multi-agent + git worktree parallel execution playbook, 12-core/24-GiB resource budgeting, and Rust-core-first two-stage disjoint crate/binding ownership rules.
- [egraph-rs-verification](skills/egraph-rs-verification/SKILL.md): Tiered Rust, Python, WASM, invariant, and parity verification commands.
- [egraph-rs-system-patterns](skills/egraph-rs-system-patterns/SKILL.md): Core algorithmic and architectural patterns across layouts, kernels, clustering, and bindings.
- [egraph-rs-dev-process](skills/egraph-rs-dev-process/SKILL.md): General engineering workflow, comment standards, autonomous verified commit/push completion gate, and skill maintenance rules.
- [egraph-rs-commit](skills/egraph-rs-commit/SKILL.md): Conventional Commits specification and crate scoping rules.
- [egraph-rs-github](skills/egraph-rs-github/SKILL.md): Branch/worktree discipline, push & tag immutability guards, PR/Issue templates, and non-polling `gh` CI workflows.
- [egraph-rs-release](skills/egraph-rs-release/SKILL.md): Version-bump checklist (`Cargo.toml`, `package-lock.json`, docs), OIDC Trusted Publishing prerequisites, non-polling `gh run watch`, and live PyPI/npm clean-env smoke verification.
