# `egraph-metrics-cluster`

Single-threaded cluster-area faithfulness, community, and clustering baseline metrics for `egraph-rs`.

> **Recommended Entry Point**: `egraph-metrics-cluster` is a component crate of **`egraph-rs`**. Most Rust users should install **`egraph-rs`** (`cargo add egraph-rs@6.0.0-beta.3`) and import items via `use egraph::prelude::*;` (or `use egraph::metrics::...;`).

## Overview

Provides cluster-level layout evaluation metrics across Euclidean and non-Euclidean layouts:

- Community coherence (`adjusted_patch_nig`, `adjusted_patch_nig_multiscale`)
- Grid-based Cluster Area Faithfulness (`cluster_area_faithfulness`, `G-CAF`)
- Optional baseline clustering and Node Overlap Removal metrics (`eval-baselines` feature)

## Direct Dependency (Minimal / Advanced Use)

```bash
cargo add egraph-metrics-cluster@6.0.0-beta.3
```

## Feature Flags

| Feature | Purpose |
|---|---|
| `eval-baselines` | Opt-in external/internal clustering metrics (`Silhouette`, `ARI`/`AMI`, `SepMe`, `Neighborhood Hit`, `Sprawlter-NN`, `NOR`) |

## Links

- **API Documentation**: [docs.rs/egraph-metrics-cluster](https://docs.rs/egraph-metrics-cluster)
- **Unified Facade Crate**: [docs.rs/egraph-rs](https://docs.rs/egraph-rs)
- **Source**: [crates/metrics-cluster on GitHub](https://github.com/likr/egraph-rs/tree/master/crates/metrics-cluster)
- **License**: MIT ([LICENSE](https://github.com/likr/egraph-rs/blob/master/crates/metrics-cluster/LICENSE))
