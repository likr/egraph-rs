# `egraph-layout-embedding`

Single-threaded sampling-based neighborhood graph embedding layouts for `egraph-rs`.

> **Recommended Entry Point**: `egraph-layout-embedding` is a component crate of **`egraph-rs`**. Most Rust users should install **`egraph-rs`** (`cargo add egraph-rs@6.0.0-beta.3`) and import items via `use egraph::prelude::*;` (or `use egraph::layout::embedding::...;`).

## Overview

Provides single-algorithm-per-type cooperative `IterativeSolver` implementations for sampling-based graph embedding:

- `PacmapGraphSolver` (3-tier PaCMAP-Graph)
- `SnapTFdpSolver` (streaming large-scale $t$-FDP)
- `LargeVisSolver`, `DrGraphSolver`, and `GraphCneSolver`

## Direct Dependency (Minimal / Advanced Use)

```bash
cargo add egraph-layout-embedding@6.0.0-beta.3
```

## Links

- **API Documentation**: [docs.rs/egraph-layout-embedding](https://docs.rs/egraph-layout-embedding)
- **Unified Facade Crate**: [docs.rs/egraph-rs](https://docs.rs/egraph-rs)
- **Source**: [crates/layout-embedding on GitHub](https://github.com/likr/egraph-rs/tree/master/crates/layout-embedding)
- **License**: MIT ([LICENSE](https://github.com/likr/egraph-rs/blob/master/crates/layout-embedding/LICENSE))
