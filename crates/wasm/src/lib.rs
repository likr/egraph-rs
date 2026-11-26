//! WebAssembly binding for the `egraph` library.
//!
//! This crate provides JavaScript/WebAssembly bindings for `egraph`, exposing:
//!
//! * Zero-copy CSR graphs, linear-memory 2D/ND layouts, quality metrics, spatial/dataset
//!   utilities, spectral/neighborhood primitives, clustering, and cooperative iterative solvers
//!   ([`csr_graph`], [`dog_varadhan`], [`layout_buffers`], [`linalg_amg`], [`linalg_chebyshev`],
//!   [`linalg_matrix`], [`linalg_resistance`], [`linalg_spectral`], [`metrics_report`],
//!   [`metrics_crossings`], [`metrics_cluster_rta`], [`metrics_functions`], [`metrics_eval`],
//!   [`neighborhood_knn`], [`neighborhood_types`], [`spatial_dataset`], [`v6`])
//! * Opt-in legacy `petgraph`-based graph, drawing, layout, edge bundling, clustering,
//!   quality metrics, and RNG modules (behind the `legacy` feature flag).

#[cfg(feature = "legacy")]
pub mod clustering;
#[cfg(feature = "legacy")]
pub mod distance_matrix;
#[cfg(feature = "legacy")]
pub mod drawing;
#[cfg(feature = "legacy")]
pub mod edge_bundling;
#[cfg(feature = "legacy")]
pub mod graph;
#[cfg(feature = "legacy")]
pub mod layout;
#[cfg(feature = "legacy")]
pub mod quality_metrics;
#[cfg(feature = "legacy")]
pub mod rng;

pub mod csr_graph;
pub mod dog_varadhan;
mod js_convert;
mod js_options;
pub mod layout_buffers;
pub mod linalg_amg;
pub mod linalg_chebyshev;
pub mod linalg_matrix;
pub mod linalg_resistance;
pub mod linalg_spectral;
pub mod metrics_cluster_rta;
pub mod metrics_crossings;
pub mod metrics_eval;
pub mod metrics_functions;
pub mod metrics_report;
pub mod neighborhood_knn;
pub mod neighborhood_types;
pub mod spatial_dataset;

pub mod v6;
mod v6_m3;
mod v6_m4;
mod v6_m5;
mod v6_w1;
mod v6_w2;

use wasm_bindgen::prelude::*;

/// Installs `console_error_panic_hook` once so any Rust panic reports a readable stack trace
/// to the JavaScript console.
#[wasm_bindgen(js_name = setPanicHook)]
pub fn set_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen(start)]
fn wasm_start() {
    console_error_panic_hook::set_once();
}
