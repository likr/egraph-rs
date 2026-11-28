# `egraph-layout-composite`

Single-threaded composite multi-solver graph layout algorithms for `egraph-rs`.

> **Recommended Entry Point**: `egraph-layout-composite` is a component crate of **`egraph-rs`**. Most Rust users should install **`egraph-rs`** (`cargo add egraph-rs@6.0.0-beta.3`) and import items via `use egraph::prelude::*;` (or `use egraph::layout::composite::...;`).

## Overview

Provides composite multi-stage graph layout algorithms behind the opt-in `baselines` feature:

- `BackboneSolver` (`extract_quadrilateral_backbone`)
- `SelLayoutSolver`
- `GroupInABoxSolver` (`squarified_treemap` re-export)

## Direct Dependency (Minimal / Advanced Use)

```bash
cargo add egraph-layout-composite@6.0.0-beta.3 --features baselines
```

## Feature Flags

| Feature | Purpose |
|---|---|
| `baselines` | Enables `BackboneSolver`, `SelLayoutSolver`, `GroupInABoxSolver`, and `squarified_treemap` |

## Links

- **API Documentation**: [docs.rs/egraph-layout-composite](https://docs.rs/egraph-layout-composite)
- **Unified Facade Crate**: [docs.rs/egraph-rs](https://docs.rs/egraph-rs)
- **Source**: [crates/layout-composite on GitHub](https://github.com/likr/egraph-rs/tree/master/crates/layout-composite)
- **License**: MIT ([LICENSE](https://github.com/likr/egraph-rs/blob/master/crates/layout-composite/LICENSE))
