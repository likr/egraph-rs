//! WebAssembly binding for the `egraph` library.
//!
//! This crate provides JavaScript/WebAssembly bindings for `egraph`, exposing zero-copy CSR graphs,
//! linear-memory 2D/N-D/Manifold layouts, quality metrics, spatial trees, datasets, spectral and
//! neighborhood primitives, clustering, coarsening, and cooperative iterative solvers.

pub mod backbone;
pub mod cluster_layout;
pub mod coarsen_backbone;
pub mod community;
pub mod community_leiden;
pub mod community_spectral;
pub mod constraints_layering;
pub mod constraints_twin_declutter;
pub mod csr_graph;
pub mod datasets_catalog;
pub mod dog_varadhan;
pub mod fdeb;
pub mod force_baselines;
pub mod force_classic;
pub mod force_multilevel;
pub mod gumap;
pub mod gumap_steps;
pub mod hde;
pub mod ib_umap;
mod js_convert;
mod js_options;
pub mod layout_buffers;
pub mod linalg_amg;
pub mod linalg_chebyshev;
pub mod linalg_chebyshev_kernel;
pub mod linalg_dijkstra;
pub mod linalg_laplacian;
pub mod linalg_matrix;
pub mod linalg_mds;
pub mod linalg_resistance;
pub mod linalg_spectral;
pub mod manifold;
pub mod manifold_geodesic;
pub mod metrics_cluster_baselines;
pub mod metrics_cluster_baselines_nd;
pub mod metrics_cluster_rta;
pub mod metrics_crossings;
pub mod metrics_eval;
pub mod metrics_eval_dispatch;
pub mod metrics_functions;
pub mod metrics_geometry;
pub mod metrics_nd_manifold;
pub mod metrics_non_euclidean_crossings;
pub mod metrics_report;
pub mod metrics_rta;
pub mod metrics_summaries;
pub mod neighborhood_knn;
pub mod neighborhood_types;
pub mod overlap_pfs_sfc;
pub mod overlap_prism;
pub mod overlap_tree_forbid;
pub mod rhgh_landmarks;
pub mod screened_poisson;
pub mod screened_poisson_refiner;
pub mod sgd;
pub mod sgd_barrier;
pub mod sgd_baselines;
pub mod sgd_pacmap;
mod sgd_sampling;
pub mod sgd_schedule;
pub mod sgd_stats;
mod solver_runner;
pub mod spatial_dataset;
pub mod spatial_refiner;
pub mod spatial_trees;
pub mod stress_kamada;
pub mod stress_kamada_kawai;
pub mod stress_multilevel_baselines;
pub mod t_fdp;
pub mod tsnet;
mod tsnet_options;
pub mod tsnet_stages;

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
