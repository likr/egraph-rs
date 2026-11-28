# `egraph-layout-neighbor`

Neighborhood-preserving `tsNET`, `GuMap`, `t-FDP`, and `ibUMAP` graph layout solvers for `egraph-rs`.

> **Recommended Entry Point**: `egraph-layout-neighbor` is a component crate of **`egraph-rs`**. Most Rust users should install **`egraph-rs`** (`cargo add egraph-rs@6.0.0-beta.3`) and import items via `use egraph::prelude::*;` (or `use egraph::layout::neighbor::...;`).

## Overview

Provides strictly single-threaded, cooperative `IterativeSolver` implementations for neighborhood-preserving graph layouts with compile-time typed repulsion backends (`Auto2d`, `Auto3d`, `FitFft`, `LinearFft`, `IbFft`, `BarnesHut`, `Exact`):

- `TsNetSolver` (Pure 1+2-hop, row-normalized random walk, $k$-NN, partial-BFS)
- `GuMapSolver` (Chebyshev DoG and graph-based UMAP)
- `TFdpSolver` and `IbUmapSolver`

## Direct Dependency (Minimal / Advanced Use)

```bash
cargo add egraph-layout-neighbor@6.0.0-beta.3
```

## Links

- **API Documentation**: [docs.rs/egraph-layout-neighbor](https://docs.rs/egraph-layout-neighbor)
- **Unified Facade Crate**: [docs.rs/egraph-rs](https://docs.rs/egraph-rs)
- **Source**: [crates/layout-neighbor on GitHub](https://github.com/likr/egraph-rs/tree/master/crates/layout-neighbor)
- **License**: MIT ([LICENSE](https://github.com/likr/egraph-rs/blob/master/crates/layout-neighbor/LICENSE))
