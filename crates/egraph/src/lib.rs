//! Unified single-threaded, pure-Rust facade for `egraph-rs` (library crate `egraph`).
//!
//! Install via Cargo as `egraph-rs` and import items via `use egraph::prelude::*;`.
//! `egraph-rs` re-exports `egraph-core` unconditionally as [`core`] and exposes
//! feature-gated domain modules ([`spatial`], [`edge_routing`], [`linalg`],
//! [`neighborhood`], [`layout`], [`clustering`], [`metrics`], [`dataset`])
//! alongside a curated [`prelude`] module for Native Rust, Python, and
//! WebAssembly consumers.
//!
//! By default (`default` features), `egraph-rs` enables pure-Rust spatial indexing,
//! edge routing, linear algebra, neighborhood construction, layout solvers,
//! clustering, metrics, and embedded CSR benchmark datasets while remaining
//! strictly single-threaded.

#![warn(missing_docs)]

pub use egraph_core as core;

#[cfg(feature = "spatial")]
pub mod spatial;

#[cfg(feature = "edge-routing")]
/// Single-threaded edge routing and edge bundling solvers (`FdebSolver`).
pub use egraph_edge_routing as edge_routing;

#[cfg(feature = "linalg")]
pub mod linalg;

#[cfg(feature = "neighborhood")]
/// Single-threaded neighborhood graph, adaptive perplexity, and fuzzy simplicial set primitives.
pub use egraph_neighborhood as neighborhood;

#[cfg(feature = "layout")]
pub mod layout;

#[cfg(feature = "clustering")]
/// Single-threaded graph clustering and coarsening primitives.
pub use egraph_clustering as clustering;

#[cfg(feature = "metrics")]
/// Unified layout quality evaluation metrics and target-selective dispatcher.
pub use egraph_metrics as metrics;

#[cfg(feature = "dataset")]
/// Pure-CSR benchmark and synthetic graph dataset loaders.
pub use egraph_dataset as dataset;

pub mod prelude;
