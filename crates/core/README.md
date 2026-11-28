# `egraph-core`

Core zero-copy CSR graph, layout, and Riemannian geometry primitives for `egraph-rs`.

> **Recommended Entry Point**: `egraph-core` is a component crate of **`egraph-rs`**. Most Rust users should install **`egraph-rs`** (`cargo add egraph-rs@6.0.0-beta.3`) and import items via `use egraph::prelude::*;` (or `use egraph::core::...;`).

## Overview

Provides strictly single-threaded, pure-Rust foundational types shared across all `egraph-rs` crates:

- `CsrGraph` and `CsrGraphBuilder` (`u32` CSR topology + optional `f64`/`f32` weights)
- Contiguous SoA layout buffers: `Layout2d`, `Layout3d`, `LayoutNd`, `ManifoldLayout2d`, `EdgePolylines`
- Geometry and solver traits: `LayoutStorage`, `MetricSpace`, `RiemannianLayout`, `EuclideanLayout`, `IterativeSolver`, `LayoutGenerator`, `LayoutTransform`, `RepulsionBackend`
- Deterministic seeded PRNG wrapper (`DeterministicRng`) and error hierarchy (`CoreError`, `CoreResult`)

## Direct Dependency (Minimal / Advanced Use)

```bash
cargo add egraph-core@6.0.0-beta.3
```

## Feature Flags

| Feature | Purpose |
|---|---|
| `serde` | Enables `serde::Serialize` and `serde::Deserialize` on core graph, layout, manifold, and solver status types |

## Links

- **API Documentation**: [docs.rs/egraph-core](https://docs.rs/egraph-core)
- **Unified Facade Crate**: [docs.rs/egraph-rs](https://docs.rs/egraph-rs)
- **Source**: [crates/core on GitHub](https://github.com/likr/egraph-rs/tree/master/crates/core)
- **License**: MIT ([LICENSE](https://github.com/likr/egraph-rs/blob/master/crates/core/LICENSE))
