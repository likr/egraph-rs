# `egraph-neighborhood`

Single-threaded neighborhood graph, adaptive perplexity, fuzzy simplicial set, and Varadhan landmark primitives for `egraph-rs`.

> **Recommended Entry Point**: `egraph-neighborhood` is a component crate of **`egraph-rs`**. Most Rust users should install **`egraph-rs`** (`cargo add egraph-rs@6.0.0-beta.3`) and import items via `use egraph::prelude::*;` (or `use egraph::neighborhood::...;`).

## Overview

Provides sparse neighborhood and affinity representations for tsNET, UMAP, and Chebyshev DoG pipelines:

- Exact 1+2-hop CSR extraction (`Sparse12HopPairs`)
- Adaptive perplexity probability calibration (`SparseProbabilities`, `PerplexityConfig`)
- $k$-NN graph construction (`KnnGraph`) and `fuzzy_simplicial_set`
- Regularized Chebyshev DoG / rHGH Varadhan geodesic landmarks

## Direct Dependency (Minimal / Advanced Use)

```bash
cargo add egraph-neighborhood@6.0.0-beta.3
```

## Links

- **API Documentation**: [docs.rs/egraph-neighborhood](https://docs.rs/egraph-neighborhood)
- **Unified Facade Crate**: [docs.rs/egraph-rs](https://docs.rs/egraph-rs)
- **Source**: [crates/neighborhood on GitHub](https://github.com/likr/egraph-rs/tree/master/crates/neighborhood)
- **License**: MIT ([LICENSE](https://github.com/likr/egraph-rs/blob/master/crates/neighborhood/LICENSE))
