---
name: egraph-rs-dev-process
description: Guidelines for general development process, coding style, writing comments, parallel worktree workflow, and updating skills after tasks in the egraph-rs project.
---

# Development Process Guidelines for `egraph-rs`

This skill details the general development rules, comment conventions, parallel worktree workflow, confirmation gate, and self-updating skill process for the `egraph-rs` project.

## Code Comment Rules
- **Language**: All comments in code files (Rust, Python, JS, TS) and all documentation inside `.agents/` must be written in English.
- **Content**: Write comments that explain the **WHY**, not the **HOW**. Explain design decisions, numerical stability guards, and algorithmic justifications.
- **Cleanliness**: Avoid unnecessary commented-out code or work-in-progress notes.

## Single-Thread & Cross-Target Engineering Rules
- `egraph-rs` targets **Native Rust, Python (PyO3), and WebAssembly (`wasm-bindgen`)** under strictly **single-threaded plain CPU execution** (no `rayon`, `std::thread`, Web Workers, GPU, or C/Fortran FFI).
- Consult [`.agents/AGENTS.md`](../../AGENTS.md) and [`docs/V6_RELEASE_PLAN.md`](../../../docs/V6_RELEASE_PLAN.md) before modifying crates or bindings.

## Task Execution Workflow

### 1. Before Starting a Task
- Understand task instructions, milestone ownership, and single-thread/WASM constraints.
- Read `egraph-rs-system-patterns` (and `egraph-rs-parallel-dev` when working with parallel worktrees or swarm subagents).
- If modifying code concurrently with other agents, create or enter an isolated git worktree (`./scripts/worktree.sh create <slug>` or `workspace:"share"`) and use its local `.venv`.

### 2. During Implementation
- Adhere to "Code Comment Rules" and avoid duplicate primitives across crates.
- Iterate quickly with Tier 0 verification (`./scripts/verify.sh --fast [--crate <name>]`) while developing.
- Never silently introduce `thread_rng()`, `from_entropy()`, `std::thread`, or default-featured `ndarray`/`linfa`.

### 3. Before Task Completion (Final Verification Process)
- **Formatting, Invariants, and Tests**:
  - Run `cargo fmt --all`.
  - Run `./scripts/verify.sh` matching scope (`--rust`, `--python`, `--wasm`, `--invariants`, `--parity`, `--docs`, or `--all --build-wasm --docs`).
  - Ensure all modified crates (outside legacy `petgraph-linalg-kernel` baseline warnings) pass `clippy ... -- -D warnings`.
  - Before release tagging, follow `egraph-rs-release` to confirm workspace dependency & `crates/wasm/package-lock.json` alignment (`ndarray` `0.17` `default-features = false` matching `numpy` `0.28`), `crates/wasm/package.json` `repository` provenance metadata, `crates/python/pyproject.toml` classifiers matching tag stability (`alpha`/`beta`/`rc`/GA), and `CHANGELOG.md` accuracy.
- **Prepare Commit**:
  - Split unrelated changes into atomic commits following `egraph-rs-commit` and `egraph-rs-github` rules (`git diff --cached --stat`).
  - In multi-milestone work, commit atomically at each verified milestone/wave boundary and push at major stage boundaries (`egraph-rs-github`) rather than deferring all commits to the final release step.
- **Ask for User Confirmation**:
  - Present a summary of all changes made (in **Japanese** for user communications).
  - Include the proposed commit message and whether pushing to remote is planned (`egraph-rs-github`).
  - Wait for explicit user approval before committing unless the user has explicitly authorized autonomous atomic commits/pushes for the active task (such as an approved `/teamwork` or milestone execution prompt).

## Post-Task Reflection: Updating Skills
Upon completing any task, the agent **MUST** reflect on the work performed:
- Update relevant skill files (`egraph-rs-dev-process`, `egraph-rs-verification`, `egraph-rs-parallel-dev`, `egraph-rs-system-patterns`) or `.agents/AGENTS.md` whenever new architecture invariants, crate layouts, or verification tooling are established.
