//! Sparse SGD layout algorithm for WebAssembly.
//!
//! This module provides WebAssembly bindings for the Sparse SGD layout algorithm,
//! which is a scalable force-directed approach that uses pivot nodes to approximate
//! distances, making it suitable for large graphs.

use crate::{graph::JsGraph, layout::sgd::sgd::JsSgd, rng::JsRng};
use js_sys::Function;
use petgraph::visit::EdgeRef;
use petgraph_layout_sgd::SparseSgd;
use wasm_bindgen::prelude::*;

use super::extract_edge_lengths;

/// WebAssembly binding for Sparse SGD layout algorithm.
///
/// Sparse SGD is an efficient variant of SGD that uses pivot-based
/// approximation of graph distances, making it suitable for large graphs
/// where computing all-pairs shortest paths would be too expensive.
#[wasm_bindgen(js_name = "SparseSgd")]
pub struct JsSparseSgd {
    builder: SparseSgd<f32>,
}

#[wasm_bindgen(js_class = "SparseSgd")]
impl JsSparseSgd {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            builder: SparseSgd::new(),
        }
    }

    pub fn build(&self, graph: &JsGraph, length: &Function, rng: &mut JsRng) -> JsSgd {
        let length_map = extract_edge_lengths(graph.graph(), length);
        JsSgd::new_with_sgd(self.builder.build(
            graph.graph(),
            |e| length_map[&e.id()],
            rng.get_mut(),
        ))
    }
}
