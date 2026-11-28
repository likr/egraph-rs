# AGENTS.md - `egraph-rs` Operational Knowledge Base & Development Harness Index

## 1. Project Goal & Active Roadmap

`egraph-rs` is a high-performance graph drawing and layout library targeting **Native Rust, Python (PyO3), and WebAssembly (`wasm-bindgen` / TypeScript)** equally as first-class targets.

- **Active Release Plan**: See [`docs/V6_API_FREEZE_SPEC.md`](../docs/V6_API_FREEZE_SPEC.md) and archived planning docs [`docs/archive/V6_RELEASE_PLAN.md`](../docs/archive/V6_RELEASE_PLAN.md) / [`docs/archive/V6_ALPHA9_REMEDIATION_PLAN.md`](../docs/archive/V6_ALPHA9_REMEDIATION_PLAN.md). Architectural redesign (**Step 0 + M0–M6**) shipped through `v6.0.0-alpha.9`, and the v6 beta API freeze refactored the workspace into a clean **28-crate** legacy-free architecture (`24` publishable + `4` non-publishable `ffi`/`python`/`wasm`/`cli`) for **`v6.0.0-beta.1`** (PyPI `6.0.0b1`, npm `6.0.0-beta.1 --tag next`) with unified `IterativeSolver<Target = ...>`, `LayoutGenerator`, `LayoutTransform`, 3D (`Layout3d`/`ArenaOctree3d`), `EdgePolylines`, compile-time typed backends, and non-Euclidean manifold parallel transport.
- **Paired Research & Benchmark Repo**: `/Users/likr/graph-drawing-experiments` provides empirically validated algorithms (Track A Pure 1+2-Hop Adaptive Fit-tsNET, Track B $O(|E|)$ Regularized Chebyshev DoG + 3-Tier Sparse Barrier SGD / GuMap + 2D Spatial Intruder Refinement) and regression benchmarks (`suite30`, `tsnet_ref5`, `drgraph` up to 4M nodes).

---

## 2. Non-Negotiable Engineering Invariants

1. **Strictly Single-Threaded Pure Rust**:
   - **NEVER** add `rayon`, `std::thread`, Web Workers, `SharedArrayBuffer`, CUDA/WebGPU, or C/Fortran FFI dependencies.
   - Always configure `ndarray = { ..., default-features = false }` (never pull `rayon` transitively).
   - Enforced statically via [`clippy.toml`](../clippy.toml) and [`scripts/check_invariants.sh`](../scripts/check_invariants.sh).
2. **Deterministic Explicit RNG**:
   - Never call `rand::thread_rng()` or `StdRng::from_entropy()` in core libraries; accept explicit `u64` seeds only and encapsulate PRNG internals strictly behind `DeterministicRng` (never expose `rand_core::RngCore` or `rand_chacha::ChaCha8Rng` in public signatures).
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

### First-Principles Design Axioms & Derived Architectural Guardrails

Derive every API, trait, crate-layout, and binding decision from these five minimal axioms rather than ad-hoc convenience:

1. **Simple Over Easy**: Reject convenient multiplexing (mega-structs with optional states, runtime mode locks, implicit wrappers, pass-through umbrella crates) in favor of clean, orthogonal concepts where **1 algorithm = 1 type** and data buffers (`Layout*`, `EdgePolylines<L>`) stay separate from solver state (`LayoutGenerator`, `IterativeSolver`, `LayoutTransform`).
2. **Performance-First Zero-Cost Abstraction**: Never sacrifice contiguous zero-copy CSR/SoA memory, cache locality, SIMD vectorization, or zero-allocation inner loops for unnecessary indirection. Keep hot loops on `&CsrGraph<W>` (`indptr: &[u32]`), and keep extension traits (`LayoutStorage`, `MetricSpace`, `RiemannianLayout`, `RepulsionBackend<L>`) open while keeping performance-coupled storage (`EuclideanLayout`) **sealed** so fast internal views remain tunable without breaking callers.
3. **Fail Fast**: Catch mismatches at compile time (`IterativeSolver { type Target: ?Sized }` instead of runtime `ActiveLayoutKind` switches). When runtime checks are necessary (dimension mismatch, `NaN`, unsupported options/dtypes such as WASM `weightDtype: "float64"`, or `run_steps(target, 0)`), fail immediately with structured errors (`SolverStatus::Failed` on fatal breakdown) without mutating targets or silently coercing inputs.
4. **No Ad-Hoc Branching or Silent Fallbacks**: Treat any urge to write ad-hoc geometry branches (`if active_dims == 2`), silent Cargo feature fallbacks (e.g., falling back from Delaunay to k-NN), or feature-gated `pub` struct fields as a defect in the underlying abstraction or dependency boundary—fix the design instead. Keep Cargo features strictly additive.
5. **Minimal Evolvable Surface**: Never leak external crate types (`petgraph`, `rand_core`, `ChaCha8Rng`, host handles), keep internal modules `pub(crate)` with curated `prelude` and `publish = false` on bindings/ABI, declare workspace `rust-version`, annotate evolvable public types with `#[non_exhaustive]` + `with_*` builders (no `Copy` on evolvable Config/Report/Stats), strip unused `W: Scalar` bounds from topological CSR functions, organize crates as a flat 1:1 DAG (`crates/<x>` <-> `egraph-<x>`), and avoid positional parameter lock-in in Python/WASM bindings.

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
| [`scripts/verify.sh`](../scripts/verify.sh) | Tiered verification runner (`--fast`, `--rust`, `--python`, `--wasm`, `--invariants`, `--parity`, `--docs`, `--all`, `--json-summary`) auto-configuring `sccache`, single-thread env vars, `CARGO_BUILD_JOBS` from `EGRAPH_PARALLEL_WORKERS`, `nextest`, and helper CLI [`scripts/verify_helpers.py`](../scripts/verify_helpers.py) (`<= ~500L`). |
| [`scripts/check_invariants.sh`](../scripts/check_invariants.sh) | Enforce `cargo tree` (`rayon`/`linfa`/`wgpu`=0), static source scan, `wasm32-unknown-unknown` build, and packaging metadata alignment (`--mode report` or `--mode strict`). |
| [`scripts/parity_harness.py`](../scripts/parity_harness.py) | Cross-target parity runner delegating to [`scripts/parity/`](../scripts/parity/__init__.py) (30 modules) for deterministic golden fixtures, reproducibility, adversarial non-panic input tests, and optional Node.js WASM parity checks. |
| [`clippy.toml`](../clippy.toml) | Blocks `std::thread::{spawn, scope, Builder, JoinHandle}` workspace-wide via Clippy. |

### Verification Efficiency & Adversarial Convergence Guardrails
- **One Full `--all` Per Unique Code Diff (Milestone, Rebase & Final Gate)**: Never run `./scripts/verify.sh --all --build-wasm --docs` redundantly across Worker, Reviewer, Auditor, and post-merge root checks for the same code diff. Run it **once** in the milestone/Phase-1 gate with `--json-summary`; reviewers/auditors consume that SHA-stamped summary alongside focused complementary checks (`check_invariants.sh`, `tsc`, Playwright, targeted adversarial probes). After a clean rebase onto unrelated docs/commits, verify patch equivalence via `diff -u <(git diff <old_base>..<verified_sha>) <(git diff <new_base>..HEAD)` and run only fast warm incremental smoke checks—do not repeat full `--all` rebuilds.
- **No Cold Full-Workspace `--release` Rebuilds in Isolated Probe Dirs**: Auditors and Challengers MUST NOT initialize an empty `CARGO_TARGET_DIR` (e.g., `/tmp/.../cargo_target`) to recompile all 27 workspace crates from scratch in `--release`. Run isolated adversarial probes either against a small standalone binary crate referencing only the target crate, or via warm targeted tests sharing the workspace `target/` and `sccache`.
- **Adversarial Iteration Cap & Practical Severity Triage**: Cap adversarial fix-and-reverify loops at **`<= 3` rounds** per milestone/tier, and **`<= 2` consecutive rounds** for synthetic extreme IEEE-754 floating-point edge cases. Panics, `NaN`/`Inf` outputs on finite inputs, API contract violations, and practical-scale defects block acceptance; finite, non-panicking precision drift occurring solely at extreme synthetic scales (`|coord| > 1e100`, `< 1e-300`, or `1e50+` component ratios) **MUST** be logged as non-blocking follow-up items rather than blocking milestone convergence. Whenever a numerical defect recurs in Round 2, perform a single-pass audit of all sibling functions in that module before editing.
- **Deliverable vs. Scratch Harness Boundary & Structured Sync**: Apply adversarial bypass hardening only to committed deliverables (cap scratch harness fixes at `<= 2` iterations; stop if `>= 2` iterations produce zero target-repo diffs), and verify crate/test/CI alignment via structured metadata (`Cargo.toml`, `scripts/verify_helpers.py`, `.github/workflows/*.yml`, `--json-summary`).
- **Disk Space & Multi-Repo Scope Isolation**: Check `df -h .` (`>= 25 GiB` free recommended before spawning parallel worktrees or Docker wheel builds), clean merged `.worktrees/<slug>` immediately, and when scoped to a single repository (e.g., `egraph-rs` only), leave sibling workspaces (`/Users/likr/graph-drawing-experiments`, `/Users/likr/papers`) unmodified (`git status --short` verified clean).

---

## 5. Skills Index

- [egraph-rs-architecture-principles](skills/egraph-rs-architecture-principles/SKILL.md): Minimal design axioms, trade-off decision matrix, SemVer hardening, zero-fallback discipline, crate DAG partitioning, and cross-target binding principles for `egraph-rs`.
- [egraph-rs-parallel-dev](skills/egraph-rs-parallel-dev/SKILL.md): Multi-agent + git worktree parallel execution playbook, 12-core/24-GiB resource budgeting, and Rust-core-first two-stage disjoint crate/binding ownership rules.
- [egraph-rs-verification](skills/egraph-rs-verification/SKILL.md): Tiered Rust, Python, WASM, invariant, and parity verification commands.
- [egraph-rs-system-patterns](skills/egraph-rs-system-patterns/SKILL.md): Core algorithmic and architectural patterns across layouts, kernels, clustering, and bindings.
- [egraph-rs-dev-process](skills/egraph-rs-dev-process/SKILL.md): General engineering workflow, comment standards, autonomous verified commit/push completion gate, and skill maintenance rules.
- [egraph-rs-commit](skills/egraph-rs-commit/SKILL.md): Conventional Commits specification and crate scoping rules.
- [egraph-rs-github](skills/egraph-rs-github/SKILL.md): Branch/worktree discipline, push & tag immutability guards, PR/Issue templates, and non-polling `gh` CI workflows.
- [egraph-rs-release](skills/egraph-rs-release/SKILL.md): Version-bump checklist (`Cargo.toml`, `package-lock.json`, docs), OIDC Trusted Publishing prerequisites, non-polling `gh run watch`, and live PyPI/npm clean-env smoke verification.
