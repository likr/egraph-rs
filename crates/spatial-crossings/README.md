# `egraph-spatial-crossings`

Sweep-line edge-crossing detection and non-Euclidean geodesic crossing geometry for `egraph-rs`.

> **Recommended Entry Point**: `egraph-spatial-crossings` is a component crate of **`egraph-rs`**. Most Rust users should install **`egraph-rs`** (`cargo add egraph-rs@6.0.0-beta.3`) and import items via `use egraph::prelude::*;` (or `use egraph::spatial::...;`).

## Overview

Provides deterministic, single-threaded segment intersection and edge-crossing kernels across Euclidean 2D (`sweep_line_crossings`), hyperbolic Poincaré disk geodesics, spherical great-circle arcs, and 2D flat torus sub-segments.

## Direct Dependency (Minimal / Advanced Use)

```bash
cargo add egraph-spatial-crossings@6.0.0-beta.3
```

## Links

- **API Documentation**: [docs.rs/egraph-spatial-crossings](https://docs.rs/egraph-spatial-crossings)
- **Unified Facade Crate**: [docs.rs/egraph-rs](https://docs.rs/egraph-rs)
- **Source**: [crates/spatial-crossings on GitHub](https://github.com/likr/egraph-rs/tree/master/crates/spatial-crossings)
- **License**: MIT ([LICENSE](https://github.com/likr/egraph-rs/blob/master/crates/spatial-crossings/LICENSE))
