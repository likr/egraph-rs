# `egraph-layout-multilevel`

Single-threaded multilevel coarse backbone, `GRIP`, and space-filling curve graph layouts for `egraph-rs`.

> **Recommended Entry Point**: `egraph-layout-multilevel` is a component crate of **`egraph-rs`**. Most Rust users should install **`egraph-rs`** (`cargo add egraph-rs@6.0.0-beta.3`) and import items via `use egraph::prelude::*;` (or `use egraph::layout::multilevel::...;`).

## Overview

Provides multilevel and space-filling initializers and solvers:

- `CoarseBackbone` (`LayoutGenerator` for Louvain area-reserved coarse backbone initialization)
- Optional comparative multilevel/SFC baselines (`baselines` feature)

## Direct Dependency (Minimal / Advanced Use)

```bash
cargo add egraph-layout-multilevel@6.0.0-beta.3
```

## Feature Flags

| Feature | Purpose |
|---|---|
| `baselines` | Opt-in `GripSolver` and `SfcLayout` space-filling curve layout generator |

## Links

- **API Documentation**: [docs.rs/egraph-layout-multilevel](https://docs.rs/egraph-layout-multilevel)
- **Unified Facade Crate**: [docs.rs/egraph-rs](https://docs.rs/egraph-rs)
- **Source**: [crates/layout-multilevel on GitHub](https://github.com/likr/egraph-rs/tree/master/crates/layout-multilevel)
- **License**: MIT ([LICENSE](https://github.com/likr/egraph-rs/blob/master/crates/layout-multilevel/LICENSE))
