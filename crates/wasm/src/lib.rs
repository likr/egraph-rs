//! WebAssembly binding for the `egraph` library.
//!
//! This crate provides JavaScript/WebAssembly bindings for `egraph`, exposing:
//!
//! * Zero-copy CSR graphs, linear-memory 2D/ND layouts, quality metrics, spatial/dataset
//!   utilities, spectral/neighborhood primitives, clustering, coarsening, and cooperative
//!   iterative solvers ([`coarsen_backbone`], [`community`], [`constraints_layering`],
//!   [`csr_graph`], [`dog_varadhan`], [`fdeb`], [`force_multilevel`], [`gumap`], [`ib_umap`],
//!   [`layout_buffers`], [`linalg_amg`], [`linalg_chebyshev`], [`linalg_matrix`],
//!   [`linalg_resistance`], [`linalg_spectral`], [`manifold`], [`metrics_cluster_rta`],
//!   [`metrics_crossings`], [`metrics_eval`], [`metrics_functions`], [`metrics_report`],
//!   [`neighborhood_knn`], [`neighborhood_types`], [`overlap_prism`], [`rhgh_landmarks`],
//!   [`screened_poisson`], [`sgd`], [`sgd_barrier`], [`sgd_pacmap`], [`spatial_dataset`],
//!   [`spatial_refiner`], [`stress_kamada`], [`t_fdp`], [`tsnet`], [`v6`])
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

pub mod coarsen_backbone;
pub mod community;
pub mod constraints_layering;
pub mod csr_graph;
pub mod dog_varadhan;
pub mod fdeb;
pub mod force_multilevel;
pub mod gumap;
pub mod ib_umap;
mod js_convert;
mod js_options;
pub mod layout_buffers;
pub mod linalg_amg;
pub mod linalg_chebyshev;
pub mod linalg_matrix;
pub mod linalg_resistance;
pub mod linalg_spectral;
pub mod manifold;
pub mod metrics_cluster_rta;
pub mod metrics_crossings;
pub mod metrics_eval;
pub mod metrics_functions;
pub mod metrics_report;
pub mod neighborhood_knn;
pub mod neighborhood_types;
pub mod overlap_prism;
pub mod rhgh_landmarks;
pub mod screened_poisson;
pub mod sgd;
pub mod sgd_barrier;
pub mod sgd_pacmap;
mod sgd_sampling;
mod solver_runner;
pub mod spatial_dataset;
pub mod spatial_refiner;
pub mod stress_kamada;
pub mod t_fdp;
pub mod tsnet;
mod tsnet_options;

pub mod v6;
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
