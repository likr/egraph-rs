//! WebAssembly binding for the `egraph` library.
//!
//! This crate provides JavaScript/WebAssembly bindings for `egraph`, exposing:
//!
//! * v6 zero-copy CSR graphs and linear-memory 2D layouts ([`v6`])
//! * Feature-gated legacy `petgraph`-based graph, drawing, layout, edge bundling, clustering,
//!   quality metrics, and RNG modules (`legacy` feature, enabled by default until M1-M4 replace
//!   them with unified `egraph-core` solvers).

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

pub mod v6;
mod v6_m2;

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
