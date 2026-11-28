# `egraph-spatial-trees`

Single-threaded spatial indexing trees, rectangle overlap sweep, and 2D Delaunay triangulation for `egraph-rs`.

> **Recommended Entry Point**: `egraph-spatial-trees` is a component crate of **`egraph-rs`**. Most Rust users should install **`egraph-rs`** (`cargo add egraph-rs@6.0.0-beta.3`) and import items via `use egraph::prelude::*;` (or `use egraph::spatial::...;`).

## Overview

Provides pure-Rust spatial data structures and geometric utilities:

- `StaticKdTree`, `DynamicKdTree`, `ArenaQuadtree2d`, `ArenaOctree3d`, and `NearestNeighborIndex`
- 2D Delaunay triangulation (`Delaunay2d`)
- Rectangle overlap sweep-line detection and `squarified_treemap`
- `TorusView` periodic boundary queries

## Direct Dependency (Minimal / Advanced Use)

```bash
cargo add egraph-spatial-trees@6.0.0-beta.3
```

## Links

- **API Documentation**: [docs.rs/egraph-spatial-trees](https://docs.rs/egraph-spatial-trees)
- **Unified Facade Crate**: [docs.rs/egraph-rs](https://docs.rs/egraph-rs)
- **Source**: [crates/spatial-trees on GitHub](https://github.com/likr/egraph-rs/tree/master/crates/spatial-trees)
- **License**: MIT ([LICENSE](https://github.com/likr/egraph-rs/blob/master/crates/spatial-trees/LICENSE))
