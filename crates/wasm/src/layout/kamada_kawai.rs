//! Kamada-Kawai graph layout algorithm for WebAssembly.
//!
//! This module provides a WebAssembly binding for the Kamada-Kawai force-directed
//! graph layout algorithm, which positions nodes based on graph-theoretic distances.

use crate::{drawing::JsDrawingEuclidean2d, graph::JsGraph};
use js_sys::{Function, Reflect};
use petgraph::visit::EdgeRef;
use petgraph_layout_kamada_kawai::KamadaKawai;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

/// WebAssembly binding for the Kamada-Kawai layout algorithm.
///
/// The Kamada-Kawai algorithm is a force-directed graph drawing method that
/// positions nodes to reflect their graph-theoretic distances. It aims to
/// place nodes such that the geometric distance between them corresponds
/// to their shortest path distance in the graph.
#[wasm_bindgen(js_name = KamadaKawai)]
pub struct JsKamadaKawai {
    kamada_kawai: KamadaKawai<f32>,
}

#[wasm_bindgen(js_class = KamadaKawai)]
impl JsKamadaKawai {
    /// Creates a new Kamada-Kawai layout instance for the given graph.
    ///
    /// @param {Graph} graph - The graph to layout
    /// @param {Function} f - A function that takes an edge index and returns an object with a "distance" property
    /// @returns {KamadaKawai} A new Kamada-Kawai layout instance
    /// @throws {Error} If any edge's distance is not a number
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph, f: &Function) -> Result<JsKamadaKawai, JsValue> {
        let mut distance = HashMap::new();
        for e in graph.graph().edge_indices() {
            let result = f.call1(&JsValue::null(), &JsValue::from_f64(e.index() as f64))?;
            let d = Reflect::get(&result, &"distance".into())?
                .as_f64()
                .ok_or_else(|| format!("links[{}].distance is not a Number.", e.index()))?;
            distance.insert(e, d as f32);
        }
        Ok(JsKamadaKawai {
            kamada_kawai: KamadaKawai::new(graph.graph(), |e| distance[&e.id()]),
        })
    }

    /// Selects the node with the maximum gradient in the current layout.
    ///
    /// This method identifies which node should be moved next to improve the layout.
    ///
    /// @param {DrawingEuclidean2d} drawing - The current drawing of the graph
    /// @returns {number|null} The index of the selected node, or null if no suitable node is found
    #[wasm_bindgen(js_name = selectNode)]
    pub fn select_node(&self, drawing: &JsDrawingEuclidean2d) -> Option<usize> {
        self.kamada_kawai.select_node(drawing.drawing())
    }

    /// Applies the Kamada-Kawai algorithm to adjust a single node's position.
    ///
    /// @param {number} m - The index of the node to adjust
    /// @param {DrawingEuclidean2d} drawing - The drawing to modify
    #[wasm_bindgen(js_name = applyToNode)]
    pub fn apply_to_node(&self, m: usize, drawing: &mut JsDrawingEuclidean2d) {
        self.kamada_kawai.apply_to_node(m, drawing.drawing_mut())
    }

    /// Runs the complete Kamada-Kawai algorithm on the drawing.
    ///
    /// This method iteratively selects nodes and adjusts their positions until
    /// the layout converges to a stable state.
    ///
    /// @param {DrawingEuclidean2d} drawing - The drawing to modify
    pub fn run(&self, drawing: &mut JsDrawingEuclidean2d) {
        self.kamada_kawai.run(drawing.drawing_mut())
    }

    /// Gets the convergence threshold (epsilon).
    ///
    /// The algorithm stops when the maximum gradient falls below this threshold.
    ///
    /// @returns {number} The current epsilon value
    #[wasm_bindgen(getter)]
    pub fn eps(&self) -> f32 {
        self.kamada_kawai.eps
    }

    /// Sets the convergence threshold (epsilon).
    ///
    /// A smaller value leads to more precise layouts but may require more iterations.
    ///
    /// @param {number} value - The new epsilon value
    #[wasm_bindgen(setter)]
    pub fn set_eps(&mut self, value: f32) {
        self.kamada_kawai.eps = value;
    }
}
