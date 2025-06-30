//! Sparse SGD layout algorithm for WebAssembly.
//!
//! This module provides WebAssembly bindings for the Sparse SGD layout algorithm,
//! which is a scalable force-directed approach that uses pivot nodes to approximate
//! distances, making it suitable for large graphs.

use crate::{
    drawing::{
        JsDrawingEuclidean, JsDrawingEuclidean2d, JsDrawingHyperbolic2d, JsDrawingSpherical2d,
        JsDrawingTorus2d,
    },
    rng::JsRng,
};
use js_sys::Function;
use petgraph_layout_sgd::Sgd;
use wasm_bindgen::prelude::*;

use super::{create_distance_transform, create_weight_transform};

/// WebAssembly binding for Sparse SGD layout algorithm.
///
/// Sparse SGD is an efficient variant of SGD that uses pivot-based
/// approximation of graph distances, making it suitable for large graphs
/// where computing all-pairs shortest paths would be too expensive.
#[wasm_bindgen(js_name = "Sgd")]
pub struct JsSgd {
    sgd: Sgd<f32>,
}

impl JsSgd {
    pub fn new_with_sgd(sgd: Sgd<f32>) -> Self {
        Self { sgd }
    }
}

#[wasm_bindgen(js_class = "Sgd")]
impl JsSgd {
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
}
