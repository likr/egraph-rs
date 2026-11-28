# `egraph-layout-refine`

Single-threaded post-layout spatial and density-field refinement algorithms for `egraph-rs`.

> **Recommended Entry Point**: `egraph-layout-refine` is a component crate of **`egraph-rs`**. Most Rust users should install **`egraph-rs`** (`cargo add egraph-rs@6.0.0-beta.3`) and import items via `use egraph::prelude::*;` (or `use egraph::layout::refine::...;`).

## Overview

Provides cooperative `IterativeSolver` post-layout refinement solvers:

- `SpatialIntruderRefiner` (2D spatial intruder removal maximizing 2-hop neighborhood preservation)
- `ScreenedPoissonRefiner` (FFT-based screened Poisson density-field refinement)
- Optional cluster-faithful refinement (`baselines` feature)

## Direct Dependency (Minimal / Advanced Use)

```bash
cargo add egraph-layout-refine@6.0.0-beta.3
```

## Feature Flags

| Feature | Purpose |
|---|---|
| `baselines` | Opt-in `ClusterRefineSolver` and `refine_cluster_layout` |

## Links

- **API Documentation**: [docs.rs/egraph-layout-refine](https://docs.rs/egraph-layout-refine)
- **Unified Facade Crate**: [docs.rs/egraph-rs](https://docs.rs/egraph-rs)
- **Source**: [crates/layout-refine on GitHub](https://github.com/likr/egraph-rs/tree/master/crates/layout-refine)
- **License**: MIT ([LICENSE](https://github.com/likr/egraph-rs/blob/master/crates/layout-refine/LICENSE))
