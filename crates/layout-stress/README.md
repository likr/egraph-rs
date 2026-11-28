# `egraph-layout-stress`

Kamada-Kawai, stress majorization, and maxent-stress graph layout solvers for `egraph-rs`.

> **Recommended Entry Point**: `egraph-layout-stress` is a component crate of **`egraph-rs`**. Most Rust users should install **`egraph-rs`** (`cargo add egraph-rs@6.0.0-beta.3`) and import items via `use egraph::prelude::*;` (or `use egraph::layout::stress::...;`).

## Overview

Provides cooperative `IterativeSolver` implementations for stress-based graph drawing across Euclidean (`Layout2d`, `Layout3d`, `LayoutNd`) and Riemannian manifold (`ManifoldLayout2d`) spaces:

- `StressMajorizationSolver` (`EuclideanPcg`, `GeodesicGradient` strategies)
- `KamadaKawaiSolver`
- Optional `MaxentStressSolver` (`baselines` feature)

## Direct Dependency (Minimal / Advanced Use)

```bash
cargo add egraph-layout-stress@6.0.0-beta.3
```

## Feature Flags

| Feature | Purpose |
|---|---|
| `baselines` | Opt-in `MaxentStressSolver` with `Exact` and `BarnesHut` repulsion backends |

## Links

- **API Documentation**: [docs.rs/egraph-layout-stress](https://docs.rs/egraph-layout-stress)
- **Unified Facade Crate**: [docs.rs/egraph-rs](https://docs.rs/egraph-rs)
- **Source**: [crates/layout-stress on GitHub](https://github.com/likr/egraph-rs/tree/master/crates/layout-stress)
- **License**: MIT ([LICENSE](https://github.com/likr/egraph-rs/blob/master/crates/layout-stress/LICENSE))
