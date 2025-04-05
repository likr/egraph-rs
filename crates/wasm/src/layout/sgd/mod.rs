//! SGD-based layout algorithms for WebAssembly.
//!
//! This module provides WebAssembly bindings for Stochastic Gradient Descent (SGD)
//! based graph layout algorithms, which are force-directed approaches that efficiently
//! handle large graphs by optimizing layouts through stochastic sampling of node pairs.
//!
//! The module includes several variants of SGD algorithms:
//! * Full SGD - computes accurate layouts using all node pairs
//! * Sparse SGD - uses pivot nodes to approximate distances for better scalability
//! * Distance-adjusted variants - modify distances dynamically to avoid node overlap

pub mod distance_adjusted;
pub mod full;
pub mod schedulers;
pub mod sparse;

// Re-export all SGD implementations for convenience
pub use self::distance_adjusted::{JsDistanceAdjustedFullSgd, JsDistanceAdjustedSparseSgd};
pub use self::full::JsFullSgd;
pub use self::sparse::JsSparseSgd;

// Common functionality for SGD implementations
use js_sys::{Array, Function};
use wasm_bindgen::JsValue;

/// Helper function to create a distance transform function from JavaScript
pub(crate) fn create_distance_transform(
    distance_fn: &Function,
) -> impl Fn(usize, usize, f32, f32) -> f32 + '_ {
    move |i: usize, j: usize, dij: f32, wij: f32| {
        let args = Array::new();
        args.push(&JsValue::from_f64(i as f64));
        args.push(&JsValue::from_f64(j as f64));
        args.push(&JsValue::from_f64(dij as f64));
        args.push(&JsValue::from_f64(wij as f64));
        distance_fn
            .apply(&JsValue::null(), &args)
            .unwrap()
            .as_f64()
            .unwrap() as f32
    }
}

/// Helper function to create a weight transform function from JavaScript
pub(crate) fn create_weight_transform(
    weight_fn: &Function,
) -> impl Fn(usize, usize, f32, f32) -> f32 + '_ {
    move |i: usize, j: usize, d: f32, w: f32| {
        let args = Array::new();
        args.push(&JsValue::from_f64(i as f64));
        args.push(&JsValue::from_f64(j as f64));
        args.push(&JsValue::from_f64(d as f64));
        args.push(&JsValue::from_f64(w as f64));
        weight_fn
            .apply(&JsValue::null(), &args)
            .unwrap()
            .as_f64()
            .unwrap() as f32
    }
}

/// Helper function to extract edge length information from JavaScript
pub(crate) fn extract_edge_lengths(
    graph: &petgraph::graph::Graph<crate::graph::Node, crate::graph::Edge, petgraph::Undirected>,
    length_fn: &Function,
) -> std::collections::HashMap<petgraph::graph::EdgeIndex, f32> {
    let mut length_map = std::collections::HashMap::new();
    for e in graph.edge_indices() {
        let c = length_fn
            .call1(&JsValue::null(), &JsValue::from_f64(e.index() as f64))
            .unwrap()
            .as_f64()
            .unwrap() as f32;
        length_map.insert(e, c);
    }
    length_map
}
