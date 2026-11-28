# `egraph-layout-sgd`

Single-threaded stochastic gradient descent stress and sparse barrier layout solvers for `egraph-rs`.

> **Recommended Entry Point**: `egraph-layout-sgd` is a component crate of **`egraph-rs`**. Most Rust users should install **`egraph-rs`** (`cargo add egraph-rs@6.0.0-beta.3`) and import items via `use egraph::prelude::*;` (or `use egraph::layout::sgd::...;`).

## Overview

Provides cooperative `IterativeSolver` SGD layout optimizers across 2D, 3D, $N$-D, and non-Euclidean manifolds:

- `SgdSolver` (Full SGD, Ortmann sparse pivot stress, 3-tier Sparse Barrier Chebyshev DoG SGD)
- `Sgd2Solver` and `SgdPairBuffer`
- `SgdScheduleConfig` and streaming negative sampling helpers

## Direct Dependency (Minimal / Advanced Use)

```bash
cargo add egraph-layout-sgd@6.0.0-beta.3
```

## Links

- **API Documentation**: [docs.rs/egraph-layout-sgd](https://docs.rs/egraph-layout-sgd)
- **Unified Facade Crate**: [docs.rs/egraph-rs](https://docs.rs/egraph-rs)
- **Source**: [crates/layout-sgd on GitHub](https://github.com/likr/egraph-rs/tree/master/crates/layout-sgd)
- **License**: MIT ([LICENSE](https://github.com/likr/egraph-rs/blob/master/crates/layout-sgd/LICENSE))
