//! Full SGD layout algorithm for WebAssembly.
//!
//! This module provides WebAssembly bindings for the Full SGD layout algorithm,
//! which is a force-directed approach that uses all pairs of nodes for computing
//! the layout, providing accurate results but with higher computational complexity.

use crate::{
    drawing::{
        JsDrawingEuclidean, JsDrawingEuclidean2d, JsDrawingHyperbolic2d, JsDrawingSpherical2d,
        JsDrawingTorus2d,
    },
    graph::JsGraph,
    layout::sgd::schedulers::{
        JsSchedulerConstant, JsSchedulerExponential, JsSchedulerLinear, JsSchedulerQuadratic,
        JsSchedulerReciprocal,
    },
    rng::JsRng,
};
use js_sys::{Array, Function};
use petgraph::visit::EdgeRef;
use petgraph_layout_sgd::{FullSgd, Sgd};
use wasm_bindgen::prelude::*;

use super::extract_edge_lengths;

/// WebAssembly binding for Full SGD layout algorithm.
///
/// Full SGD is a force-directed layout algorithm that computes shortest-path
/// distances between all pairs of nodes. While accurate, it can be
/// computationally expensive for large graphs.
#[wasm_bindgen(js_name = "FullSgd")]
pub struct JsFullSgd {
    sgd: FullSgd<f32>,
}

#[wasm_bindgen(js_class = "FullSgd")]
impl JsFullSgd {
    /// Creates a new Full SGD layout instance for the given graph.
    ///
    /// Takes a graph and a length function that determines desired edge lengths.
    /// The length function should take an edge index and return its desired length.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph, length: &Function) -> JsFullSgd {
        let length_map = extract_edge_lengths(graph.graph(), length);
        JsFullSgd {
            sgd: FullSgd::new(graph.graph(), |e| length_map[&e.id()]),
        }
    }

    /// Shuffles the node pairs used for SGD updates to improve randomization.
    ///
    /// Uses the provided random number generator for shuffling operations.
    pub fn shuffle(&mut self, rng: &mut JsRng) {
        self.sgd.shuffle(rng.get_mut());
    }

    /// Applies a single SGD update step to a 2D Euclidean drawing.
    ///
    /// Modifies the drawing by adjusting node positions with the specified learning rate.
    #[wasm_bindgen(js_name = "applyWithDrawingEuclidean2d")]
    pub fn apply_with_drawing_euclidean_2d(&self, drawing: &mut JsDrawingEuclidean2d, eta: f32) {
        self.sgd.apply(drawing.drawing_mut(), eta);
    }

    /// Applies a single SGD update step to a n-dimensional Euclidean drawing.
    ///
    /// Modifies the drawing by adjusting node positions with the specified learning rate.
    #[wasm_bindgen(js_name = "applyWithDrawingEuclidean")]
    pub fn apply_with_drawing_euclidean(&self, drawing: &mut JsDrawingEuclidean, eta: f32) {
        self.sgd.apply(drawing.drawing_mut(), eta);
    }

    /// Applies a single SGD update step to a hyperbolic 2D drawing.
    ///
    /// Modifies the drawing by adjusting node positions with the specified learning rate.
    #[wasm_bindgen(js_name = "applyWithDrawingHyperbolic2d")]
    pub fn apply_with_drawing_hyperbolic_2d(&self, drawing: &mut JsDrawingHyperbolic2d, eta: f32) {
        self.sgd.apply(drawing.drawing_mut(), eta);
    }

    /// Applies a single SGD update step to a spherical 2D drawing.
    ///
    /// Modifies the drawing by adjusting node positions with the specified learning rate.
    #[wasm_bindgen(js_name = "applyWithDrawingSpherical2d")]
    pub fn apply_with_drawing_spherical_2d(&self, drawing: &mut JsDrawingSpherical2d, eta: f32) {
        self.sgd.apply(drawing.drawing_mut(), eta);
    }

    /// Applies a single SGD update step to a torus 2D drawing.
    ///
    /// Modifies the drawing by adjusting node positions with the specified learning rate.
    #[wasm_bindgen(js_name = "applyWithDrawingTorus2d")]
    pub fn apply_with_drawing_torus_2d(&self, drawing: &mut JsDrawingTorus2d, eta: f32) {
        self.sgd.apply(drawing.drawing_mut(), eta);
    }

    /// Creates an exponential decay scheduler for controlling the learning rate.
    ///
    /// This is the default scheduler type and works well for most graphs.
    ///
    /// @param {number} t_max - The maximum number of iterations
    /// @param {number} epsilon - The minimum learning rate
    /// @returns {SchedulerExponential} A learning rate scheduler
    pub fn scheduler(&self, t_max: usize, epsilon: f32) -> JsSchedulerExponential {
        self.scheduler_exponential(t_max, epsilon)
    }

    /// Creates a constant learning rate scheduler.
    ///
    /// @param {number} t_max - The maximum number of iterations
    /// @param {number} epsilon - The constant learning rate value
    /// @returns {SchedulerConstant} A learning rate scheduler
    #[wasm_bindgen(js_name = "schedulerConstant")]
    pub fn scheduler_constant(&self, t_max: usize, epsilon: f32) -> JsSchedulerConstant {
        JsSchedulerConstant {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    /// Creates a linear decay scheduler for controlling the learning rate.
    ///
    /// @param {number} t_max - The maximum number of iterations
    /// @param {number} epsilon - The minimum learning rate
    /// @returns {SchedulerLinear} A learning rate scheduler
    #[wasm_bindgen(js_name = "schedulerLinear")]
    pub fn scheduler_linear(&self, t_max: usize, epsilon: f32) -> JsSchedulerLinear {
        JsSchedulerLinear {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    /// Creates a quadratic decay scheduler for controlling the learning rate.
    ///
    /// @param {number} t_max - The maximum number of iterations
    /// @param {number} epsilon - The minimum learning rate
    /// @returns {SchedulerQuadratic} A learning rate scheduler
    #[wasm_bindgen(js_name = "schedulerQuadratic")]
    pub fn scheduler_quadratic(&self, t_max: usize, epsilon: f32) -> JsSchedulerQuadratic {
        JsSchedulerQuadratic {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    /// Creates an exponential decay scheduler for controlling the learning rate.
    ///
    /// @param {number} t_max - The maximum number of iterations
    /// @param {number} epsilon - The minimum learning rate
    /// @returns {SchedulerExponential} A learning rate scheduler
    #[wasm_bindgen(js_name = "schedulerExponential")]
    pub fn scheduler_exponential(&self, t_max: usize, epsilon: f32) -> JsSchedulerExponential {
        JsSchedulerExponential {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    /// Creates a reciprocal decay scheduler for controlling the learning rate.
    ///
    /// @param {number} t_max - The maximum number of iterations
    /// @param {number} epsilon - The minimum learning rate
    /// @returns {SchedulerReciprocal} A learning rate scheduler
    #[wasm_bindgen(js_name = "schedulerReciprocal")]
    pub fn scheduler_reciprocal(&self, t_max: usize, epsilon: f32) -> JsSchedulerReciprocal {
        JsSchedulerReciprocal {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    /// Updates the distance calculation function for node pairs.
    ///
    /// This allows customizing how graph-theoretic distances are transformed
    /// into target distances in the layout.
    ///
    /// Takes a function that receives (node_i, node_j, distance, weight) parameters
    /// and returns a new transformed distance value.
    #[wasm_bindgen(js_name = "updateDistance")]
    pub fn update_distance(&mut self, distance: &Function) {
        self.sgd.update_distance(|i, j, dij, wij| {
            let args = Array::new();
            args.push(&JsValue::from_f64(i as f64));
            args.push(&JsValue::from_f64(j as f64));
            args.push(&JsValue::from_f64(dij as f64));
            args.push(&JsValue::from_f64(wij as f64));
            distance
                .apply(&JsValue::null(), &args)
                .unwrap()
                .as_f64()
                .unwrap() as f32
        });
    }

    /// Updates the weight calculation function for node pairs.
    ///
    /// This allows customizing how much influence each node pair has on the layout.
    ///
    /// Takes a function that receives (node_i, node_j, distance, weight) parameters
    /// and returns a new weight value.
    #[wasm_bindgen(js_name = "updateWeight")]
    pub fn update_weight(&mut self, weight: &Function) {
        self.sgd.update_weight(|i, j, d, w| {
            let args = Array::new();
            args.push(&JsValue::from_f64(i as f64));
            args.push(&JsValue::from_f64(j as f64));
            args.push(&JsValue::from_f64(d as f64));
            args.push(&JsValue::from_f64(w as f64));
            weight
                .apply(&JsValue::null(), &args)
                .unwrap()
                .as_f64()
                .unwrap() as f32
        });
    }
}
