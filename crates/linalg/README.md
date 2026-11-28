# `egraph-linalg`

Single-threaded sparse linear algebra, spectral diffusion, and shortest-path primitives for `egraph-rs`.

> **Recommended Entry Point**: `egraph-linalg` is a component crate of **`egraph-rs`**. Most Rust users should install **`egraph-rs`** (`cargo add egraph-rs@6.0.0-beta.3`) and import items via `use egraph::prelude::*;` (or `use egraph::linalg::...;`).

## Overview

Provides pure-Rust, zero-FFI linear algebra and graph spectral primitives:

- `CsrMatrix` and sparse matrix-vector / Chebyshev polynomial expansion (`ChebyshevDogKernel`)
- Graph Laplacians (`combinatorial_laplacian`, `symmetric_normalized_laplacian`, `random_walk_laplacian`, `shifted_normalized_laplacian`)
- Single-source and all-pairs shortest paths (`DijkstraWorkspace`)
- Open `LinearOperator` trait and dense symmetric eigensolvers

## Direct Dependency (Minimal / Advanced Use)

```bash
cargo add egraph-linalg@6.0.0-beta.3
```

## Links

- **API Documentation**: [docs.rs/egraph-linalg](https://docs.rs/egraph-linalg)
- **Unified Facade Crate**: [docs.rs/egraph-rs](https://docs.rs/egraph-rs)
- **Source**: [crates/linalg on GitHub](https://github.com/likr/egraph-rs/tree/master/crates/linalg)
- **License**: MIT ([LICENSE](https://github.com/likr/egraph-rs/blob/master/crates/linalg/LICENSE))
