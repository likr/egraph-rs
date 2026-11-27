---
name: egraph-rs-architecture-principles
description: >-
  First-principles architectural decision criteria, trade-off resolution matrix, trait/SemVer
  design rules, zero-fallback feature discipline, crate DAG partitioning, and Python/WASM/TS
  binding principles for egraph-rs. Activate whenever designing, reviewing, or refactoring public
  Rust traits, types, solver hierarchies, Cargo features, crate splits, or cross-target APIs.
---

# `egraph-rs` First-Principles Architecture & Design Criteria

Do not resolve architectural choices by local convenience or ad-hoc patches. Derive every design
decision from the **five minimal axioms** below, resolve trade-offs using the **Decision Matrix**,
and verify against the **Pre-Design Checklist**.

## 1. Minimal First Principles (The 5 Decision Axioms)

1. **Simple Over Easy (Orthogonality First)**
   - Do not conflate distinct concepts merely to make one call site shorter (avoid mega-structs with
     optional modes, pass-through umbrellas, implicit coordinate wrapping, or fake `1`-step loops).
   - Separate what a concept *is* from what it *holds*: data buffers (`Layout*`, `EdgePolylines<L>`)
     vs. execution state; one-shot generation vs. iterative optimization vs. in-place transformation.
   - *Litmus Test*: If adding a new algorithm or target geometry forces unrelated fields, dummy return
     values (`Option`/constants), or mode checks into an existing type, the abstraction is *Easy*, not *Simple*.

2. **Performance-First Zero-Cost Abstraction**
   - Single-threaded pure-CPU throughput (contiguous CSR/SoA buffers, cache locality, SIMD, zero
     per-step heap allocations) is non-negotiable.
   - Abstract statically (monomorphization + associated types) and never force indirect iterators or
     element-wise copies inside inner solver/metric hot loops.
   - *Litmus Test*: Does the abstraction preserve zero-overhead access to contiguous slices (`&CsrGraph<W>`,
     SoA `&[S]` / simultaneous multi-axis `&mut [S]`) without tying public contracts to private SIMD/padding details?

3. **Fail Fast**
   - Prefer compile-time rejection (types, associated `type Target`, minimal trait bounds) over runtime checks.
   - When validation runs at runtime, reject invalid dimensions, `NaN`/non-finite coordinates, mismatched
     zero-step targets (`run_steps(target, 0)`), or unsupported options/dtypes immediately without
     mutating the solver or target buffer. Fatal solver breakdowns must transition to an explicit
     terminal error state (`SolverStatus::Failed`).
   - *Litmus Test*: Can an invalid configuration, dtype, or mixed target silently coerce, degrade,
     or leave a stateful solver stuck in `Running`? If yes, fail faster.

4. **No Ad-Hoc Branching or Silent Fallbacks**
   - Never hide a broken abstraction or missing dependency behind runtime branch hacks (`if active_dims == 2`),
     undocumented silent defaults, or feature-dependent algorithmic fallbacks (e.g., substituting k-NN
     for Delaunay when a feature is disabled).
   - Never add `#[cfg(feature = ...)]` `pub` struct fields that break feature unification additivity.
   - *Litmus Test*: Whenever you feel tempted to write a fallback or special-case branch to keep
     mismatched code working, stop—fix the trait boundary, split the type, or gate/require the dependency cleanly.

5. **Minimal Evolvable Surface**
   - Expose the smallest coherent public API needed by callers. Hide internal directory layouts,
     invariants, PRNG engines, and third-party crates completely so internal optimization and `v6.x`
     feature growth remain SemVer-compatible by default.

## 2. Decision Criteria Matrix (Resolving Recurring Trade-Offs)

When two designs compete, apply these exact resolution rules derived from Axioms 1–5:

| Recurring Trade-Off | Resolution Criteria | Governing Axioms |
|---|---|---|
| **Open vs. Sealed Trait** | Keep domain extension points **open** (`LayoutStorage`, `MetricSpace`, `RiemannianLayout`, `RepulsionBackend<L>`, `LinearOperator`, `NearestNeighborIndex`) with `f64` math and minimal required methods. **Seal** internal memory-layout contracts (`EuclideanLayout` restricted to `Layout2d`/`Layout3d`/`LayoutNd`) when stable Rust lacks specialization for simultaneous multi-axis `&mut` borrows and future SIMD layout tuning. | #2 Performance + #5 Evolvable Surface |
| **Graph Abstraction Trait vs. Concrete `CsrGraph`** | Keep solvers directly on contiguous `&CsrGraph<W>` (`indptr: &[u32]`) because indirect iterator traits degrade inner-loop cache locality; instead, remove unnecessary `W: Scalar` bounds from unweighted topological APIs so `CsrGraph<()>` works cleanly. | #2 Performance + #1 Simple |
| **Unified Multi-Mode Type vs. Separate Types** | Enforce **1 algorithm = 1 type** and compile-time target binding (`IterativeSolver { type Target: ?Sized }`). Split whenever variants diverge in state, config, telemetry, or target geometry (`FieldLayoutSolver` -> `TFdpSolver`/`IbUmapSolver`, `SgdSolver` variants split, no `ActiveLayoutKind`). | #1 Simple + #3 Fail Fast + #4 No Ad-Hoc Branching |
| **1 Trait vs. Role-Specific Traits** | Separate orthogonal behaviors cleanly: `EdgePolylines<L>` is data, not solver state; processes split into `LayoutGenerator { type Output }`, `IterativeSolver { type Target: ?Sized }`, and `LayoutTransform { type Target: ?Sized }`. | #1 Simple + #4 No Ad-Hoc Branching |
| **Optional Dependency Fallback vs. Explicit Policy** | Never fall back silently between algorithms on a Cargo feature flag. Either make a lightweight pure-Rust WASM-safe dependency unconditional (`spade`) or feature-gate the entire dependent API cleanly; remove external graph libraries (`petgraph`) whose boundary can be handled cleanly in caller code/docs. | #4 No Fallbacks + #3 Fail Fast + #5 Minimal Surface |
| **Implicit Coercion / Tolerant API vs. Explicit Rejection** | Never silently wrap Euclidean `Layout2d` as `[0, 1)^2` torus (`TorusLayoutView` sealed / no unconditional `Layout2d` impl), never ignore unsupported WASM `weightDtype: "float64"`, and never merge unrelated metric budgets (`crossing_num_samples` vs. `np_num_samples`). | #3 Fail Fast + #1 Simple |
| **Thin Umbrellas / Wrapper Explosion vs. Direct Facade** | Eliminate 1-file pass-through umbrella crates (`spatial`, `layout`) and redundant `*_nd`/`*_manifold` wrapper functions; re-export split crates directly from `egraph` and unify functions generically over `MetricSpace` / `EuclideanLayout`. | #1 Simple + #5 Minimal Surface |
| **Crate Splitting Boundary** | Place crates flatly 1:1 (`crates/<name>` <-> `egraph-<name>`), split along both domain cohesion and acyclic dependency tiers so primary solver crates compile in parallel without cross-crate bottlenecks (`edge-routing`, `layout-embedding`, `layout-refine`, `layout-composite` separated from leaf `layout-multilevel`), while keeping externally linked deployment/Colab paths untouched. | #2 Performance + #1 Simple + #5 Minimal Surface |

## 3. Derived Concrete Guardrails

### 3.1 Rust Core API, Traits & External Isolation
- **Geometry & Solver Lifecycle**: Support open-ended/convergence iteration via `total_iterations() -> Option<usize>`, `#[non_exhaustive]` `SolverStatus` with `Failed`, `&mut T`/`Box<T>` blanket impls, and invariant-preserving validation before `run_steps(target, 0)` and `run(target)`.
- **Uniform Layout & ID Symmetry**: Support first-class `Layout2d<S>`, `Layout3d<S>`, and `LayoutNd<S>` with symmetric constructors/validators (`validate_finite` everywhere); unify discrete node/layer/cluster IDs as `u32` and prefer borrowed `&[T]` slice inputs on constructors.
- **Zero External Leakage**: Wrap RNG solely in `DeterministicRng` (no `as_rng_core_mut()` or public `impl RngCore` dependency on `rand_core`/`rand_chacha`); keep core dependencies pure Rust and `serde` strictly opt-in behind a feature flag.

### 3.2 SemVer Defense & Crate Visibility
- Mark every public enum, error (`CoreError`, `ErrorKind`), config, input, stats, stage, and report struct `#[non_exhaustive]`; equip evolvable configs with `with_*` builders; **never** derive `Copy` on evolvable Config/Report/Stats/Stage/Result structs.
- Encapsulate invariants (`SparseProbabilities`, `Sparse12HopPairs`, `EdgePolylines`) behind private fields and slice getters; never accept `#[doc(hidden)]` helper types in public constructors.
- Keep internal crate modules `pub(crate)` with flat crate-root `pub use` surfaces, curate `egraph::prelude` strictly (never dump `Scalar`, `VisitStamp`, internal views, or `#[doc(hidden)]` items), set `publish = false` on bindings/ABI/binary crates, and declare workspace `rust-version`.

### 3.3 Cross-Target Bindings (Python / WASM / TypeScript)
- **Python**: Put `*` keyword-only after essential arguments whenever an API has 4+ non-coordinate parameters; enforce strict `.pyi` `Protocol`, `TypedDict`, and `Literal` coverage; replace extensible positional tuple / untyped dict returns with structured classes or `TypedDict`; subclass exceptions from `ValueError`.
- **WASM / TS**: Take `{ ...options }` objects instead of long positional lists; enforce `camelCase` across all TS unions and JS properties; accept `number | bigint` for full 64-bit seeds; map structured `ErrorKind` to error `.code`; expose typed interfaces for layouts/polylines/solvers; explicitly reject unsupported float precisions rather than silently ignoring them.

## 4. Pre-Design Review Checklist

Before finalizing any trait, public type, solver, feature flag, crate boundary, or binding signature:
1. **Simple Check**: Does every type own exactly one algorithm/role with zero fake steps, dummy outputs, or hidden mode flags?
2. **Performance Check**: Do hot loops run over contiguous CSR/SoA memory without dynamic dispatch or allocation, while keeping SIMD/borrow tuning internal via sealed storage contracts?
3. **Fail-Fast Check**: Are geometry/target incompatibilities caught at compile time, and do runtime/FFI errors reject bad inputs (including `steps == 0` and unsupported dtypes) cleanly before mutating state?
4. **Zero-Fallback Check**: Are all `#[cfg(feature = ...)]` branches strictly additive with zero silent algorithmic switches and zero conditional `pub` struct fields?
5. **Surface Check**: Are external crate types absent from public APIs, internal modules `pub(crate)`, evolvable structs/enums `#[non_exhaustive]` + non-`Copy`, and bindings free of positional lock-in?
6. **DAG Check**: Does each crate satisfy flat 1:1 `crates/<x>` <-> `egraph-<x>` naming, omit thin single-file umbrellas, and compile in its proper acyclic tier without pulling unrelated solver families?
