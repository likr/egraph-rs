# `egraph-linalg-mds`

Classical, Pivot, Landmark, Harel-Koren HDE, and Koren spectral multidimensional scaling for `egraph-rs`.

> **Recommended Entry Point**: `egraph-linalg-mds` is a component crate of **`egraph-rs`**. Most Rust users should install **`egraph-rs`** (`cargo add egraph-rs@6.0.0-beta.3`) and import items via `use egraph::prelude::*;` (or `use egraph::linalg::...;`).

## Overview

Provides deterministic `LayoutGenerator` implementations and functional helpers for fast initial graph layouts across 2D, 3D, and $N$-D spaces (`ClassicalMds`, `PivotMds`, `RdMds`, `Hde`, and `KorenSpectral`).

## Direct Dependency (Minimal / Advanced Use)

```bash
cargo add egraph-linalg-mds@6.0.0-beta.3
```

## Links

- **API Documentation**: [docs.rs/egraph-linalg-mds](https://docs.rs/egraph-linalg-mds)
- **Unified Facade Crate**: [docs.rs/egraph-rs](https://docs.rs/egraph-rs)
- **Source**: [crates/linalg-mds on GitHub](https://github.com/likr/egraph-rs/tree/master/crates/linalg-mds)
- **License**: MIT ([LICENSE](https://github.com/likr/egraph-rs/blob/master/crates/linalg-mds/LICENSE))
