# `egraph-edge-routing`

Single-threaded edge routing and edge bundling algorithms for `egraph-rs`.

> **Recommended Entry Point**: `egraph-edge-routing` is a component crate of **`egraph-rs`**. Most Rust users should install **`egraph-rs`** (`cargo add egraph-rs@6.0.0-beta.3`) and import items via `use egraph::prelude::*;` (or `use egraph::edge_routing::...;`).

## Overview

Provides force-directed edge bundling (`FdebSolver`, `FdebConfig`) implementing `IterativeSolver<Target = EdgePolylines<L>>` for 2D, 3D, and $N$-dimensional `EdgePolylines`.

## Direct Dependency (Minimal / Advanced Use)

```bash
cargo add egraph-edge-routing@6.0.0-beta.3
```

## Links

- **API Documentation**: [docs.rs/egraph-edge-routing](https://docs.rs/egraph-edge-routing)
- **Unified Facade Crate**: [docs.rs/egraph-rs](https://docs.rs/egraph-rs)
- **Source**: [crates/edge-routing on GitHub](https://github.com/likr/egraph-rs/tree/master/crates/edge-routing)
- **License**: MIT ([LICENSE](https://github.com/likr/egraph-rs/blob/master/crates/edge-routing/LICENSE))
