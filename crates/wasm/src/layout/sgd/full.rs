//! Full SGD layout algorithm for WebAssembly.
//!
//! This module provides WebAssembly bindings for the Full SGD layout algorithm,
//! which is a force-directed approach that uses all pairs of nodes for computing
//! the layout, providing accurate results but with higher computational complexity.

use crate::{graph::JsGraph, layout::sgd::sgd::JsSgd};
use js_sys::Function;
use petgraph::visit::EdgeRef;
use petgraph_layout_sgd::FullSgd;
use wasm_bindgen::prelude::*;

use super::extract_edge_lengths;

/// WebAssembly binding for Full SGD layout algorithm.
///
/// Full SGD is a force-directed layout algorithm that computes shortest-path
/// distances between all pairs of nodes. While accurate, it can be
/// computationally expensive for large graphs.
#[wasm_bindgen(js_name = "FullSgd")]
pub struct JsFullSgd {
    builder: FullSgd<f32>,
}

#[wasm_bindgen(js_class = "FullSgd")]
impl JsFullSgd {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            builder: FullSgd::new(),
        }
    }

    pub fn build(&self, graph: &JsGraph, length: &Function) -> JsSgd {
        let length_map = extract_edge_lengths(graph.graph(), length);
        JsSgd::new_with_sgd(self.builder.build(graph.graph(), |e| length_map[&e.id()]))
    }
}
