# AGENTS.md - `egraph-rs` Operational Knowledge Base & Development Harness Index

## 1. Project Goal & Active Roadmap

`egraph-rs` is a high-performance graph drawing and layout library targeting **Native Rust, Python (PyO3), and WebAssembly (`wasm-bindgen` / TypeScript)** equally as first-class targets.

- **Active Release Plan**: See [`docs/V6_RELEASE_PLAN.md`](../docs/V6_RELEASE_PLAN.md). All planned architectural redesign and feature milestones (**Step 0 + M0–M6**) culminate in **`v6.0.0-alpha.4`** on PyPI (`6.0.0a4`) and npm (`6.0.0-alpha.4`, `--tag next`). Subsequent releases (`beta`, `rc`, `v6.0.0` GA, `crates.io`) are scoped after `alpha.4` completes.
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
4. **Language Rules**:
   - All files under `.agents/` and all code comments (Rust, Python, JS/TS) **MUST be written in English**.
   - All user-facing replies and reports **MUST be written in Japanese**.

---

## 3. Verification & Parallel Development Harness

| Tool / Script | Purpose |
|---|---|
| [`scripts/setup_venv.sh`](../scripts/setup_venv.sh) | Provision isolated worktree-local Python 3.12 `.venv` (`maturin`, `pytest`, `numpy`, `scipy`, `networkx`, `mypy`, `ruff`) via `uv`. |
| [`scripts/worktree.sh`](../scripts/worktree.sh) | Manage parallel git worktrees under `.worktrees/<slug>` (`wt/<slug>` branch + isolated `.venv`), verify inside worktrees, and safely clean up. |
| [`scripts/verify.sh`](../scripts/verify.sh) | Tiered verification runner (`--fast`, `--rust`, `--python`, `--wasm`, `--invariants`, `--parity`, `--all`, `--json-summary`) auto-configuring `sccache`, single-thread env vars, `CARGO_BUILD_JOBS` from `EGRAPH_PARALLEL_WORKERS`, and `nextest`. |
| [`scripts/check_invariants.sh`](../scripts/check_invariants.sh) | Enforce `cargo tree` (`rayon`/`linfa`/`wgpu`=0), static source scan, `wasm32-unknown-unknown` build, `.wasm` gzip budget (`<= 600 KiB`), and packaging metadata alignment (`--mode report` or `--mode strict`). |
| [`scripts/parity_harness.py`](../scripts/parity_harness.py) | Deterministic golden fixtures, reproducibility, adversarial non-panic input tests, and optional Node.js WASM parity checks. |
| [`clippy.toml`](../clippy.toml) | Blocks `std::thread::{spawn, scope, Builder, JoinHandle}` workspace-wide via Clippy. |

---

## 4. Skills Index

- [egraph-rs-parallel-dev](skills/egraph-rs-parallel-dev/SKILL.md): Multi-agent + git worktree parallel execution playbook, 12-core/24-GiB resource budgeting, and M0–M6 disjoint file/crate ownership matrix.
- [egraph-rs-verification](skills/egraph-rs-verification/SKILL.md): Tiered Rust, Python, WASM, invariant, and parity verification commands.
- [egraph-rs-system-patterns](skills/egraph-rs-system-patterns/SKILL.md): Core algorithmic and architectural patterns across layouts, kernels, clustering, and bindings.
- [egraph-rs-dev-process](skills/egraph-rs-dev-process/SKILL.md): General engineering workflow, comment standards, confirmation gate before committing, and skill maintenance rules.
- [egraph-rs-commit](skills/egraph-rs-commit/SKILL.md): Conventional Commits specification and crate scoping rules.
