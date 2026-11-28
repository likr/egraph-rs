# `egraph-metrics`

Unified single-threaded graph layout evaluation metrics dispatcher for `egraph-rs`.

> **Recommended Entry Point**: `egraph-metrics` is a component crate of **`egraph-rs`**. Most Rust users should install **`egraph-rs`** (`cargo add egraph-rs@6.0.0-beta.3`) and import items via `use egraph::prelude::*;` (or `use egraph::metrics::...;`).

## Overview

Re-exports all geometric and cluster evaluation metrics alongside the target-selective `evaluate_metrics` dispatcher (`MetricTarget`, `MetricConfig`, `MetricInputs`, `MetricReport`, `ClusterBaselineConfig`, `ClusterBaselineReport`), computing only requested targets with shared spatial index reuse.

## Direct Dependency (Minimal / Advanced Use)

```bash
cargo add egraph-metrics@6.0.0-beta.3
```

## Feature Flags

| Feature | Purpose |
|---|---|
| `eval-baselines` | Opt-in cluster separation, visual separation (`SepMe`), `Sprawlter-NN`, and `NOR` baseline `MetricTarget` variants |

## Links

- **API Documentation**: [docs.rs/egraph-metrics](https://docs.rs/egraph-metrics)
- **Unified Facade Crate**: [docs.rs/egraph-rs](https://docs.rs/egraph-rs)
- **Source**: [crates/metrics on GitHub](https://github.com/likr/egraph-rs/tree/master/crates/metrics)
- **License**: MIT ([LICENSE](https://github.com/likr/egraph-rs/blob/master/crates/metrics/LICENSE))
