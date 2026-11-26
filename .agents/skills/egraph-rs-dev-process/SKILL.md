---
name: egraph-rs-dev-process
description: Guidelines for general development process, coding style, writing comments, parallel worktree workflow, and updating skills after tasks in the egraph-rs project.
---

# Development Process Guidelines for `egraph-rs`

This skill details the general development rules, comment conventions, parallel worktree workflow, autonomous commit/push completion gate, and self-updating skill process for the `egraph-rs` project.

## Code Comment Rules
- **Language**: All comments in code files (Rust, Python, JS, TS) and all documentation inside `.agents/` must be written in English.
- **Content**: Write comments that explain the **WHY**, not the **HOW**. Explain design decisions, numerical stability guards, and algorithmic justifications.
- **Cleanliness**: Avoid unnecessary commented-out code or work-in-progress notes.

## Single-Thread & Cross-Target Engineering Rules
- `egraph-rs` targets **Native Rust, Python (PyO3), and WebAssembly (`wasm-bindgen`)** under strictly **single-threaded plain CPU execution** (no `rayon`, `std::thread`, Web Workers, GPU, or C/Fortran FFI).
- Consult [`.agents/AGENTS.md`](../../AGENTS.md) and [`docs/V6_RELEASE_PLAN.md`](../../../docs/V6_RELEASE_PLAN.md) before modifying crates or bindings.

## Crate Granularity, Module Size & Snapshot Naming Rules
- **Crate Granularity for Fast Compilation**:
  - Split independent Rust core algorithm families and subsystems into cohesive, domain-scoped crates with narrow dependency graphs so incremental builds (`cargo check/test -p <crate>`) avoid recompiling unrelated solvers; re-export unified APIs through `crates/egraph`.
  - Register any newly split workspace crate in root `Cargo.toml`, `scripts/verify.sh`, `scripts/check_invariants.sh`, and `.github/workflows/ci.yml` (preserving strict single-thread and `getrandom`-free invariants).
- **Module Decomposition (`<= ~500` Lines per File)**:
  - Target **roughly 500 lines or fewer** per hand-written source file (`.rs`, `.py`, `.js`/`.mjs`/`.ts`). Split along cohesive domain responsibilities (e.g., `<domain>/mod.rs`, `config.rs`, `solver.rs`, `helpers.rs`, `tests.rs`, and domain-scoped files under `crates/python/src/` and `crates/wasm/src/`) rather than arbitrary line counts.
  - Move bulky unit test blocks into dedicated `#[cfg(test)] mod tests;` (`tests.rs`) files or integration tests (`tests/<domain>.rs`). Do not further bloat pre-existing oversized files; refactor them in dedicated passes.
- **Timeless, Snapshot-Clean Naming**:
  - Name crates, files, modules, types, functions, test files, test functions, and comments strictly after their **domain, algorithm, or capability** (e.g., `overlap_removal.rs`, `cluster_area.rs`, `test_leiden.py`).
  - **NEVER** embed workflow phases, milestones, waves, or parallel lane identifiers (`phase_*`, `v6_m*`, `v6_w*`, `m4_lane*`, `step0_*`, `stage*`) in code or test names. The codebase must stand on its own as a timeless snapshot independent of development history.

## Task Execution Workflow

### 1. Before Starting a Task
- Understand task instructions, crate/domain ownership, and single-thread/WASM constraints.
- Read `egraph-rs-system-patterns` (and `egraph-rs-parallel-dev` when working with parallel worktrees or swarm subagents).
- If modifying code concurrently with other agents, create or enter an isolated git worktree (`./scripts/worktree.sh create <slug>` or `workspace:"share"`) and use its local `.venv`.

### 2. During Implementation (Two-Stage Core-First Flow)
1. **Stage 1 — Rust Core Implementation & Verification First**:
   - Implement and refine all algorithmic logic, public Rust APIs, feature gates, and domain-named unit/oracle tests in pure Rust core crates first, honoring crate granularity, `<= ~500`-line module decomposition, and snapshot-clean naming.
   - Iterate rapidly with Tier 0 per-crate checks (`./scripts/verify.sh --fast --crate <name>`) and verify the full Rust core gate (`./scripts/verify.sh --rust --invariants`) before touching `crates/python` or `crates/wasm`.
   - Never silently introduce `thread_rng()`, `from_entropy()`, `std::thread`, or default-featured `ndarray`/`linfa`.
2. **Stage 2 — Python & WASM Bindings After Core Stabilizes**:
   - Once the Rust core API and tests are green, lock the cross-target Rust/Python/WASM signature contract and expose the bindings in domain-scoped files under `crates/python` and `crates/wasm` (which may proceed in parallel).
   - Verify bindings with `./scripts/verify.sh --python` and `./scripts/verify.sh --wasm --build-wasm`, followed by `--parity` and `--docs` as applicable. (Pure binding/docs/typing fixes that do not touch Rust core may run Stage 2 directly.)

### 3. Before Task Completion (Final Verification Process)
- **Formatting, Invariants, and Tests**:
  - Run `cargo fmt --all`.
  - Run `./scripts/verify.sh` matching scope (`--rust`, `--python`, `--wasm`, `--invariants`, `--parity`, `--docs`, or `--all --build-wasm --docs`).
  - Ensure all modified crates (outside legacy `petgraph-linalg-kernel` baseline warnings) pass `clippy ... -- -D warnings`.
  - Before release tagging, follow `egraph-rs-release` to confirm workspace dependency & `crates/wasm/package-lock.json` alignment (`ndarray` `0.17` `default-features = false` matching `numpy` `0.28`), `crates/wasm/package.json` `repository` provenance metadata, `crates/python/pyproject.toml` classifiers matching tag stability (`alpha`/`beta`/`rc`/GA), and `CHANGELOG.md` accuracy.
- **Prepare Commit**:
  - Split unrelated changes into atomic commits following `egraph-rs-commit` and `egraph-rs-github` rules (`git diff --cached --stat`).
  - In multi-stage work, commit atomically at each verified stage boundary (e.g., after Rust core verification, then after binding verification) and push at major stage boundaries (`egraph-rs-github`) rather than deferring all commits to the final release step.
- **Autonomous Commit, Push & Completion Report**:
  - Stage explicit file paths (`git diff --cached --stat`) and commit atomically once verification passes, keeping unrelated concurrent modifications untouched.
  - Push all verified task commits to `origin/master` (`git rev-parse HEAD origin/master` aligned) before presenting the final summary in **Japanese** to the user.
  - Reserve explicit user confirmation for release version bumps, release tags (`v6.*`), registry publishing, and destructive operations.

## Post-Task Reflection: Updating Skills
Upon completing any task, the agent **MUST** reflect on the work performed:
- Update relevant skill files (`egraph-rs-dev-process`, `egraph-rs-verification`, `egraph-rs-parallel-dev`, `egraph-rs-system-patterns`) or `.agents/AGENTS.md` whenever new architecture invariants, crate layouts, or verification tooling are established.
