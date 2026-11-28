# `egraph-linalg-amg`

Single-threaded algebraic multigrid PCG solvers and effective resistance curvature for `egraph-rs`.

> **Recommended Entry Point**: `egraph-linalg-amg` is a component crate of **`egraph-rs`**. Most Rust users should install **`egraph-rs`** (`cargo add egraph-rs@6.0.0-beta.3`) and import items via `use egraph::prelude::*;` (or `use egraph::linalg::...;`).

## Overview

Provides smoothed-aggregation algebraic multigrid preconditioned conjugate gradient solvers (`AmgPcgSolver`, `AmgPcgConfig`) for graph Laplacians, Johnson-Lindenstrauss effective resistance estimation (`compute_effective_resistances`), and resistance curvature flow (`resistance_curvature_flow`).

## Direct Dependency (Minimal / Advanced Use)

```bash
cargo add egraph-linalg-amg@6.0.0-beta.3
```

## Links

- **API Documentation**: [docs.rs/egraph-linalg-amg](https://docs.rs/egraph-linalg-amg)
- **Unified Facade Crate**: [docs.rs/egraph-rs](https://docs.rs/egraph-rs)
- **Source**: [crates/linalg-amg on GitHub](https://github.com/likr/egraph-rs/tree/master/crates/linalg-amg)
- **License**: MIT ([LICENSE](https://github.com/likr/egraph-rs/blob/master/crates/linalg-amg/LICENSE))
