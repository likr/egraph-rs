//! Distance-adjusted SGD layout algorithms for WebAssembly.
//!
//! This module provides WebAssembly bindings for distance-adjusted SGD algorithms,
//! which modify distances dynamically to improve layout quality and avoid node overlap.
//! The module includes both full and sparse variants of distance-adjusted SGD.

use crate::{
    drawing::JsDrawingEuclidean2d, graph::JsGraph, layout::sgd::schedulers::JsSchedulerExponential,
    rng::JsRng,
};
use js_sys::Function;
use petgraph::visit::EdgeRef;
use petgraph_layout_sgd::{DistanceAdjustedSgd, FullSgd, Sgd, SparseSgd};
use wasm_bindgen::prelude::*;

use super::{create_distance_transform, create_weight_transform, extract_edge_lengths};

/// WebAssembly binding for Distance-Adjusted Full SGD layout algorithm.
///
/// This variant enhances the standard Full SGD by dynamically adjusting
/// distances during layout, which can help prevent node collision and
/// improve the overall aesthetic of the layout.
#[wasm_bindgen(js_name = "DistanceAdjustedFullSgd")]
pub struct JsDistanceAdjustedFullSgd {
    sgd: DistanceAdjustedSgd<FullSgd<f32>, f32>,
}

#[wasm_bindgen(js_class = "DistanceAdjustedFullSgd")]
impl JsDistanceAdjustedFullSgd {
    /// Creates a new Distance-Adjusted Full SGD layout instance.
    ///
    /// Takes a graph and a length function that determines desired edge lengths.
    /// The length function should take an edge index and return its desired length.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph, length: &Function) -> Self {
        let length_map = extract_edge_lengths(graph.graph(), length);
        Self {
            sgd: DistanceAdjustedSgd::new(FullSgd::new(graph.graph(), |e| length_map[&e.id()])),
        }
    }

    /// Shuffles the node pairs used for SGD updates to improve randomization.
    ///
    /// Uses the provided random number generator for shuffling operations.
    pub fn shuffle(&mut self, rng: &mut JsRng) {
        self.sgd.shuffle(rng.get_mut());
    }

    /// Applies a single SGD update step without distance adjustment.
    ///
    /// This method is provided for compatibility with the standard SGD interface.
    /// In most cases, you should use applyWithDistanceAdjustment instead.
    ///
    /// Modifies the drawing by adjusting node positions with the specified learning rate.
    pub fn apply(&self, drawing: &mut JsDrawingEuclidean2d, eta: f32) {
        self.sgd.apply(drawing.drawing_mut(), eta);
    }

    /// Applies a single distance-adjusted SGD update step to the layout.
    ///
    /// This variant automatically adjusts distances to avoid node collisions.
    /// Modifies the drawing by adjusting node positions with the specified learning rate.
    #[wasm_bindgen(js_name = "applyWithDistanceAdjustment")]
    pub fn apply_with_distance_adjustment(&self, drawing: &mut JsDrawingEuclidean2d, eta: f32) {
        self.sgd.apply(drawing.drawing_mut(), eta);
    }

    /// Creates an exponential decay scheduler for controlling the learning rate.
    ///
    /// @param {number} t_max - The maximum number of iterations
    /// @param {number} epsilon - The minimum learning rate
    /// @returns {SchedulerExponential} A learning rate scheduler
    pub fn scheduler(&self, t_max: usize, epsilon: f32) -> JsSchedulerExponential {
        JsSchedulerExponential {
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
        self.sgd
            .update_distance(create_distance_transform(distance));
    }

    /// Updates the weight calculation function for node pairs.
    ///
    /// This allows customizing how much influence each node pair has on the layout.
    ///
    /// Takes a function that receives (node_i, node_j, distance, weight) parameters
    /// and returns a new weight value.
    #[wasm_bindgen(js_name = "updateWeight")]
    pub fn update_weight(&mut self, weight: &Function) {
        self.sgd.update_weight(create_weight_transform(weight));
    }

    /// The alpha parameter that controls the strength of distance adjustment.
    ///
    /// Higher values result in stronger distance adjustments.
    #[wasm_bindgen(getter)]
    pub fn alpha(&self) -> f32 {
        self.sgd.alpha
    }

    /// Sets the alpha parameter that controls the strength of distance adjustment.
    ///
    /// @param {number} value - The new alpha value
    #[wasm_bindgen(setter)]
    pub fn set_alpha(&mut self, value: f32) {
        self.sgd.alpha = value;
    }

    /// The minimum distance parameter for the distance adjustment.
    ///
    /// This distance serves as a lower bound for how close nodes can get.
    #[wasm_bindgen(getter, js_name = "minimumDistance")]
    pub fn minimum_distance(&self) -> f32 {
        self.sgd.minimum_distance
    }

    /// Sets the minimum distance parameter for the distance adjustment.
    ///
    /// @param {number} value - The new minimum distance
    #[wasm_bindgen(setter, js_name = "minimumDistance")]
    pub fn set_minimum_distance(&mut self, value: f32) {
        self.sgd.minimum_distance = value;
    }
}

/// WebAssembly binding for Distance-Adjusted Sparse SGD layout algorithm.
///
/// This variant combines the efficiency of Sparse SGD with distance adjustment
/// to improve layout quality for large graphs while preventing node overlap.
#[wasm_bindgen(js_name = "DistanceAdjustedSparseSgd")]
pub struct JsDistanceAdjustedSparseSgd {
    sgd: DistanceAdjustedSgd<SparseSgd<f32>, f32>,
}

#[wasm_bindgen(js_class = "DistanceAdjustedSparseSgd")]
impl JsDistanceAdjustedSparseSgd {
    /// Creates a new Distance-Adjusted Sparse SGD layout instance.
    ///
    /// Takes a graph, a length function, number of pivot nodes, and a random number generator.
    /// The length function should take an edge index and return its desired length.
    /// More pivot nodes give better accuracy but slower performance.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph, length: &Function, h: usize, rng: &mut JsRng) -> Self {
        let length_map = extract_edge_lengths(graph.graph(), length);
        Self {
            sgd: DistanceAdjustedSgd::new(SparseSgd::new_with_rng(
                graph.graph(),
                |e| length_map[&e.id()],
                h,
                rng.get_mut(),
            )),
        }
    }

    /// Shuffles the node pairs used for SGD updates to improve randomization.
    ///
    /// Uses the provided random number generator for shuffling operations.
    pub fn shuffle(&mut self, rng: &mut JsRng) {
        self.sgd.shuffle(rng.get_mut());
    }

    /// Applies a single SGD update step without distance adjustment.
    ///
    /// This method is provided for compatibility with the standard SGD interface.
    /// In most cases, you should use applyWithDistanceAdjustment instead.
    ///
    /// Modifies the drawing by adjusting node positions with the specified learning rate.
    pub fn apply(&self, drawing: &mut JsDrawingEuclidean2d, eta: f32) {
        self.sgd.apply(drawing.drawing_mut(), eta);
    }

    /// Applies a single distance-adjusted SGD update step to the layout.
    ///
    /// This variant automatically adjusts distances to avoid node collisions.
    /// Modifies the drawing by adjusting node positions with the specified learning rate.
    #[wasm_bindgen(js_name = "applyWithDistanceAdjustment")]
    pub fn apply_with_distance_adjustment(&self, drawing: &mut JsDrawingEuclidean2d, eta: f32) {
        self.sgd.apply(drawing.drawing_mut(), eta);
    }

    /// Creates an exponential decay scheduler for controlling the learning rate.
    ///
    /// @param {number} t_max - The maximum number of iterations
    /// @param {number} epsilon - The minimum learning rate
    /// @returns {SchedulerExponential} A learning rate scheduler
    pub fn scheduler(&self, t_max: usize, epsilon: f32) -> JsSchedulerExponential {
        JsSchedulerExponential {
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
        self.sgd
            .update_distance(create_distance_transform(distance));
    }

    /// Updates the weight calculation function for node pairs.
    ///
    /// This allows customizing how much influence each node pair has on the layout.
    ///
    /// Takes a function that receives (node_i, node_j, distance, weight) parameters
    /// and returns a new weight value.
    #[wasm_bindgen(js_name = "updateWeight")]
    pub fn update_weight(&mut self, weight: &Function) {
        self.sgd.update_weight(create_weight_transform(weight));
    }

    /// The alpha parameter that controls the strength of distance adjustment.
    ///
    /// Higher values result in stronger distance adjustments.
    #[wasm_bindgen(getter)]
    pub fn alpha(&self) -> f32 {
        self.sgd.alpha
    }

    /// Sets the alpha parameter that controls the strength of distance adjustment.
    ///
    /// @param {number} value - The new alpha value
    #[wasm_bindgen(setter)]
    pub fn set_alpha(&mut self, value: f32) {
        self.sgd.alpha = value;
    }

    /// The minimum distance parameter for the distance adjustment.
    ///
    /// This distance serves as a lower bound for how close nodes can get.
    #[wasm_bindgen(getter, js_name = "minimumDistance")]
    pub fn minimum_distance(&self) -> f32 {
        self.sgd.minimum_distance
    }

    /// Sets the minimum distance parameter for the distance adjustment.
    ///
    /// @param {number} value - The new minimum distance
    #[wasm_bindgen(setter, js_name = "minimumDistance")]
    pub fn set_minimum_distance(&mut self, value: f32) {
        self.sgd.minimum_distance = value;
    }
}
