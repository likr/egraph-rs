# `egraph-layout-force`

Single-threaded force-directed graph layout solvers (`ForceAtlas2`, `Yifan Hu`, `Fruchterman-Reingold`, `LinLog`, `GEM`) for `egraph-rs`.

> **Recommended Entry Point**: `egraph-layout-force` is a component crate of **`egraph-rs`**. Most Rust users should install **`egraph-rs`** (`cargo add egraph-rs@6.0.0-beta.3`) and import items via `use egraph::prelude::*;` (or `use egraph::layout::force::...;`).

## Overview

Provides cooperative `IterativeSolver` force-directed layout solvers with compile-time typed repulsion backends (`Exact`, `BarnesHut`, `ForceBackend`):

- `ForceAtlas2Solver` (`Layout2d`, `Layout3d`, `LayoutNd`)
- `YifanHuSolver` (`Layout2d`, `Layout3d`, `LayoutNd`)
- Optional baseline force-directed solvers (`baselines` feature)

## Direct Dependency (Minimal / Advanced Use)

```bash
cargo add egraph-layout-force@6.0.0-beta.3
```

## Feature Flags

| Feature | Purpose |
|---|---|
| `baselines` | Opt-in comparative force-directed solvers (`FruchtermanReingoldSolver` with `Grid` backend, `LinLogSolver`, `GemSolver`) |

## Links

- **API Documentation**: [docs.rs/egraph-layout-force](https://docs.rs/egraph-layout-force)
- **Unified Facade Crate**: [docs.rs/egraph-rs](https://docs.rs/egraph-rs)
- **Source**: [crates/layout-force on GitHub](https://github.com/likr/egraph-rs/tree/master/crates/layout-force)
- **License**: MIT ([LICENSE](https://github.com/likr/egraph-rs/blob/master/crates/layout-force/LICENSE))
