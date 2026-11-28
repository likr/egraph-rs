# `egraph-clustering`

Single-threaded graph community detection, K-Means++ clustering, and multilevel coarsening for `egraph-rs`.

> **Recommended Entry Point**: `egraph-clustering` is a component crate of **`egraph-rs`**. Most Rust users should install **`egraph-rs`** (`cargo add egraph-rs@6.0.0-beta.3`) and import items via `use egraph::prelude::*;` (or `use egraph::clustering::...;`).

## Overview

Provides deterministic, single-threaded community detection and coarsening algorithms:

- `Louvain`, `Leiden`, `LabelPropagation`, `SpectralClustering`, and `KMeansPlusPlus`
- Unified `CommunityDetector` and `Clusterer` traits with `ClusterResult`
- Graph contraction and multilevel coarsening (`coarsen_graph`)

## Direct Dependency (Minimal / Advanced Use)

```bash
cargo add egraph-clustering@6.0.0-beta.3
```

## Links

- **API Documentation**: [docs.rs/egraph-clustering](https://docs.rs/egraph-clustering)
- **Unified Facade Crate**: [docs.rs/egraph-rs](https://docs.rs/egraph-rs)
- **Source**: [crates/clustering on GitHub](https://github.com/likr/egraph-rs/tree/master/crates/clustering)
- **License**: MIT ([LICENSE](https://github.com/likr/egraph-rs/blob/master/crates/clustering/LICENSE))
