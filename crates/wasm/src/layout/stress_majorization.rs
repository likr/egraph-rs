//! Stress Majorization layout algorithm for WebAssembly.
//!
//! This module provides a WebAssembly binding for the Stress Majorization
//! algorithm, which is an optimization-based approach for graph drawing that
//! aims to minimize a stress energy function. This algorithm is effective at
//! creating layouts that accurately preserve graph-theoretic distances.

use crate::{drawing::JsDrawingEuclidean2d, graph::JsGraph};
use js_sys::{Function, Reflect};
use petgraph::visit::EdgeRef;
use petgraph_layout_stress_majorization::StressMajorization;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

/// WebAssembly binding for the Stress Majorization layout algorithm.
///
/// Stress Majorization is an iterative graph layout algorithm that minimizes a
/// stress energy function, which penalizes deviations between geometric distances
/// in the layout and graph-theoretic distances. It produces high-quality layouts
/// that preserve the graph's structure by solving a sequence of quadratic problems.
#[wasm_bindgen(js_name = StressMajorization)]
pub struct JsStressMajorization {
    stress_majorization: StressMajorization,
}

#[wasm_bindgen(js_class = StressMajorization)]
impl JsStressMajorization {
    /// Creates a new Stress Majorization layout instance.
    ///
    /// @param {Graph} graph - The graph to layout
    /// @param {DrawingEuclidean2d} drawing - The initial drawing to start from
    /// @param {Function} f - A function that takes an edge index and returns an object with a "distance" property
    /// @returns {StressMajorization} A new stress majorization instance
    /// @throws {Error} If any edge's distance is not a number
    #[wasm_bindgen(constructor)]
    pub fn new(
        graph: &JsGraph,
        drawing: &JsDrawingEuclidean2d,
        f: &Function,
    ) -> Result<JsStressMajorization, JsValue> {
        let mut distance = HashMap::new();
        for e in graph.graph().edge_indices() {
            let result = f.call1(&JsValue::null(), &JsValue::from_f64(e.index() as f64))?;
            let d = Reflect::get(&result, &"distance".into())?
                .as_f64()
                .ok_or_else(|| format!("links[{}].distance is not a Number.", e.index()))?;
            distance.insert(e, d as f32);
        }

        Ok(JsStressMajorization {
            stress_majorization: StressMajorization::new(graph.graph(), drawing.drawing(), |e| {
                distance[&e.id()]
            }),
        })
    }

    /// Performs a single iteration of the stress majorization algorithm.
    ///
    /// @param {DrawingEuclidean2d} drawing - The drawing to modify
    /// @returns {number} The current stress value (lower is better)
    pub fn apply(&mut self, drawing: &mut JsDrawingEuclidean2d) -> f32 {
        self.stress_majorization.apply(drawing.drawing_mut())
    }

    /// Runs the complete stress majorization algorithm until convergence.
    ///
    /// This method iteratively applies the stress majorization algorithm
    /// until the layout converges to a stable state.
    ///
    /// @param {DrawingEuclidean2d} drawing - The drawing to modify
    pub fn run(&mut self, drawing: &mut JsDrawingEuclidean2d) {
        self.stress_majorization.run(drawing.drawing_mut());
    }
}
