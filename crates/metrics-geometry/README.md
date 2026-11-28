# `egraph-metrics-geometry`

Single-threaded geometric, stress, neighborhood, proximity, and crossing graph layout metrics for `egraph-rs`.

> **Recommended Entry Point**: `egraph-metrics-geometry` is a component crate of **`egraph-rs`**. Most Rust users should install **`egraph-rs`** (`cargo add egraph-rs@6.0.0-beta.3`) and import items via `use egraph::prelude::*;` (or `use egraph::metrics::...;`).

## Overview

Provides pure-Rust layout quality evaluation functions across 2D, 3D, $N$-D, and non-Euclidean manifolds:

- Exact and pivot-sampled Scale-Normalized Stress (`scale_normalized_stress`, `sampled_scale_normalized_stress`) and `geodesic_distortion`
- Exact and sampled 1-hop / 2-hop neighborhood preservation (`NP1`, `NP2`)
- Sweep-line, hyperbolic, spherical, and torus edge crossings and `crosslessness`
- Gabriel / RNG shape preservation (`S_GG`, `S_RNG`), `minimum_angle`, `angular_resolution`, `aspect_ratio`, `prism_distortion`, `layout_degeneracy`, and `landmark_stratified_rta`

## Direct Dependency (Minimal / Advanced Use)

```bash
cargo add egraph-metrics-geometry@6.0.0-beta.3
```

## Links

- **API Documentation**: [docs.rs/egraph-metrics-geometry](https://docs.rs/egraph-metrics-geometry)
- **Unified Facade Crate**: [docs.rs/egraph-rs](https://docs.rs/egraph-rs)
- **Source**: [crates/metrics-geometry on GitHub](https://github.com/likr/egraph-rs/tree/master/crates/metrics-geometry)
- **License**: MIT ([LICENSE](https://github.com/likr/egraph-rs/blob/master/crates/metrics-geometry/LICENSE))
