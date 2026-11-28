# `egraph-dataset`

Embedded benchmark datasets and synthetic graph generators for `egraph-rs`.

> **Recommended Entry Point**: `egraph-dataset` is a component crate of **`egraph-rs`**. Most Rust users should install **`egraph-rs`** (`cargo add egraph-rs@6.0.0-beta.3`) and import items via `use egraph::prelude::*;` (or `use egraph::dataset::...;`).

## Overview

Provides strictly single-threaded, pure-Rust CSR dataset loaders, CSV parsers (`parse_csr_csv`, `parse_unweighted_csr_csv`), catalog metadata (`dataset_catalog`, `dataset_info`, `list_datasets`), and deterministic synthetic graph generators (`generate_sbm_graph`, `generate_layered_dag`, `generate_torus_grid`, `generate_spherical_icosphere`, `generate_spherical_grid`, `generate_hyperbolic_tree`, `generate_weighted_graph`).

## Direct Dependency (Minimal / Advanced Use)

```bash
cargo add egraph-dataset@6.0.0-beta.3
```

## Feature Flags

All embedded benchmark datasets are opt-in (`default = []`):

| Feature | Purpose |
|---|---|
| `embedded` | Enables all 9 embedded CSR benchmark dataset loaders |
| `1138_bus`, `3elt`, `dwt_1005`, `dwt_2680`, `poli`, `qh882`, `USpowerGrid`, `karate`, `lesmis` | Individual embedded CSV benchmark loaders (`dataset_*`) |

## Links

- **API Documentation**: [docs.rs/egraph-dataset](https://docs.rs/egraph-dataset)
- **Unified Facade Crate**: [docs.rs/egraph-rs](https://docs.rs/egraph-rs)
- **Source**: [crates/dataset on GitHub](https://github.com/likr/egraph-rs/tree/master/crates/dataset)
- **License**: MIT ([LICENSE](https://github.com/likr/egraph-rs/blob/master/crates/dataset/LICENSE))
