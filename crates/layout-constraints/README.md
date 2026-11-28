# `egraph-layout-constraints`

Separation constraints, hierarchical layering, and 2D node overlap removal solvers for `egraph-rs`.

> **Recommended Entry Point**: `egraph-layout-constraints` is a component crate of **`egraph-rs`**. Most Rust users should install **`egraph-rs`** (`cargo add egraph-rs@6.0.0-beta.3`) and import items via `use egraph::prelude::*;` (or `use egraph::layout::constraints::...;`).

## Overview

Provides pure-Rust constrained layout and overlap removal primitives:

- $O(N \log N)$ Delaunay `PrismOverlapRemoval`
- `TwinDeclutterer` (`LayoutTransform` for twin-vertex decluttering)
- Hierarchical DAG layering (`assign_layers`, `remove_cycles`, `transitive_reduction`) and `SeparationConstraint` projection
- Optional comparative overlap removal baselines (`baselines` feature)

## Direct Dependency (Minimal / Advanced Use)

```bash
cargo add egraph-layout-constraints@6.0.0-beta.3
```

## Feature Flags

| Feature | Purpose |
|---|---|
| `baselines` | Opt-in comparative overlap removal solvers (`ForbidOverlapRemoval`, `GTreeOverlapRemoval`, `PfsPrimeOverlapRemoval`) |

## Links

- **API Documentation**: [docs.rs/egraph-layout-constraints](https://docs.rs/egraph-layout-constraints)
- **Unified Facade Crate**: [docs.rs/egraph-rs](https://docs.rs/egraph-rs)
- **Source**: [crates/layout-constraints on GitHub](https://github.com/likr/egraph-rs/tree/master/crates/layout-constraints)
- **License**: MIT ([LICENSE](https://github.com/likr/egraph-rs/blob/master/crates/layout-constraints/LICENSE))
