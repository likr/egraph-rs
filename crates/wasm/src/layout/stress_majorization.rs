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
    stress_majorization: StressMajorization<f32>,
}

#[wasm_bindgen(js_class = StressMajorization)]
impl JsStressMajorization {
    /// Creates a new Stress Majorization layout instance.
    ///
    /// This constructor initializes a layout optimizer that will iteratively adjust node positions
    /// to minimize the difference between geometric distances in the layout and desired distances
    /// in the graph. The algorithm is particularly effective for creating layouts where
    /// the visual distances accurately reflect the graph-theoretic distances.
    ///
    /// The distance function should take an edge index and return an object with a
    /// "distance" property representing the ideal length for that edge in the layout.
    /// A typical approach is to use graph-theoretic distance (shortest path) or some
    /// function of it.
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
    /// This method makes one pass of node position adjustments to reduce stress in the layout.
    /// It's useful when you want to control the layout process step by step, for example
    /// to provide animation or to monitor the stress value at each step.
    ///
    /// Modifies the drawing by adjusting node positions to reduce stress.
    /// Returns the current stress value (lower is better).
    pub fn apply(&mut self, drawing: &mut JsDrawingEuclidean2d) -> f32 {
        self.stress_majorization.apply(drawing.drawing_mut())
    }

    /// Runs the complete stress majorization algorithm until convergence.
    ///
    /// This method iteratively applies the stress majorization algorithm
    /// until the layout converges to a stable state. It's a convenience method
    /// that runs multiple iterations of `apply()` automatically until the stress
    /// value stops decreasing significantly or until a maximum number of iterations
    /// is reached.
    pub fn run(&mut self, drawing: &mut JsDrawingEuclidean2d) {
        self.stress_majorization.run(drawing.drawing_mut());
    }

    /// Gets the convergence threshold (epsilon).
    ///
    /// The algorithm stops when the relative change in stress falls below this threshold.
    ///
    /// @returns {number} The current epsilon value
    #[wasm_bindgen(getter)]
    pub fn epsilon(&self) -> f32 {
        self.stress_majorization.epsilon
    }

    /// Sets the convergence threshold (epsilon).
    ///
    /// A smaller value leads to more precise layouts but may require more iterations.
    ///
    /// @param {number} value - The new epsilon value
    #[wasm_bindgen(setter)]
    pub fn set_epsilon(&mut self, value: f32) {
        self.stress_majorization.epsilon = value;
    }

    /// Gets the maximum number of iterations.
    ///
    /// The algorithm will stop after this many iterations even if convergence is not reached.
    ///
    /// @returns {number} The current maximum iterations value
    #[wasm_bindgen(getter)]
    pub fn max_iterations(&self) -> u32 {
        self.stress_majorization.max_iterations as u32
    }

    /// Sets the maximum number of iterations.
    ///
    /// A larger value allows more iterations for potentially better convergence.
    ///
    /// @param {number} value - The new maximum iterations value
    #[wasm_bindgen(setter)]
    pub fn set_max_iterations(&mut self, value: u32) {
        self.stress_majorization.max_iterations = value as usize;
    }
}
