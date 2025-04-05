//! Node overlap removal for WebAssembly.
//!
//! This module provides a WebAssembly binding for the overlap removal algorithm,
//! which adjusts node positions to resolve overlaps between nodes that are
//! represented as circles with defined radii. This is useful for improving the
//! readability of graph visualizations by ensuring nodes don't overlap.

use js_sys::Function;
use petgraph_layout_overwrap_removal::OverwrapRemoval;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

use crate::{
    drawing::{
        JsDrawingEuclidean, JsDrawingEuclidean2d, JsDrawingHyperbolic2d, JsDrawingSpherical2d,
        JsDrawingTorus2d,
    },
    graph::JsGraph,
};

/// WebAssembly binding for the node overlap removal algorithm.
///
/// This struct provides a JavaScript interface to an algorithm that adjusts node
/// positions to eliminate overlaps between nodes of varying sizes. It's typically
/// used as a post-processing step after the main layout algorithm to improve
/// visual clarity.
#[wasm_bindgen(js_name = OverwrapRemoval)]
pub struct JsOverwrapRemoval {
    overwrap_removal: OverwrapRemoval<f32>,
}

#[wasm_bindgen(js_class = OverwrapRemoval)]
impl JsOverwrapRemoval {
    /// Creates a new overlap removal instance for the given graph.
    ///
    /// Takes a graph and a radius function that determines node sizes.
    /// The radius function should take a node index and return its radius.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph, radius: &Function) -> JsOverwrapRemoval {
        let mut radius_map = HashMap::new();
        for u in graph.graph().node_indices() {
            let r = radius
                .call1(&JsValue::null(), &JsValue::from_f64(u.index() as f64))
                .unwrap()
                .as_f64()
                .unwrap() as f32;
            radius_map.insert(u, r);
        }
        JsOverwrapRemoval {
            overwrap_removal: OverwrapRemoval::new(graph.graph(), |u| radius_map[&u]),
        }
    }

    /// Applies the overlap removal algorithm to a 2D Euclidean drawing.
    ///
    /// This method adjusts node positions to eliminate overlaps while attempting
    /// to preserve the overall layout structure.
    #[wasm_bindgen(js_name = "applyWithDrawingEuclidean2d")]
    pub fn apply_with_drawing_euclidean_2d(&self, drawing: &mut JsDrawingEuclidean2d) {
        self.overwrap_removal.apply(drawing.drawing_mut());
    }

    /// Applies the overlap removal algorithm to an n-dimensional Euclidean drawing.
    #[wasm_bindgen(js_name = "applyWithDrawingEuclidean")]
    pub fn apply_with_drawing_euclidean(&self, drawing: &mut JsDrawingEuclidean) {
        self.overwrap_removal.apply(drawing.drawing_mut());
    }

    /// Applies the overlap removal algorithm to a 2D hyperbolic drawing.
    #[wasm_bindgen(js_name = "applyWithDrawingHyperbolic2d")]
    pub fn apply_with_drawing_hyperbolic_2d(&self, drawing: &mut JsDrawingHyperbolic2d) {
        self.overwrap_removal.apply(drawing.drawing_mut());
    }

    /// Applies the overlap removal algorithm to a 2D spherical drawing.
    #[wasm_bindgen(js_name = "applyWithDrawingSpherical2d")]
    pub fn apply_with_drawing_spherical_2d(&self, drawing: &mut JsDrawingSpherical2d) {
        self.overwrap_removal.apply(drawing.drawing_mut());
    }

    /// Applies the overlap removal algorithm to a 2D torus drawing.
    #[wasm_bindgen(js_name = "applyWithDrawingTorus2d")]
    pub fn apply_with_drawing_torus_2d(&self, drawing: &mut JsDrawingTorus2d) {
        self.overwrap_removal.apply(drawing.drawing_mut());
    }

    /// The strength parameter that controls how aggressively overlaps are removed.
    #[wasm_bindgen(getter)]
    pub fn get_strength(&self) -> f32 {
        self.overwrap_removal.strength
    }

    /// Sets the strength parameter that controls how aggressively overlaps are removed.
    ///
    /// Higher values result in faster but potentially more disruptive adjustments.
    #[wasm_bindgen(setter)]
    pub fn set_strength(&mut self, value: f32) {
        self.overwrap_removal.strength = value;
    }

    /// The number of iterations the algorithm performs.
    #[wasm_bindgen(getter)]
    pub fn get_iterations(&self) -> usize {
        self.overwrap_removal.iterations
    }

    /// Sets the number of iterations the algorithm performs.
    ///
    /// More iterations may lead to better results but take longer to compute.
    #[wasm_bindgen(setter)]
    pub fn set_iterations(&mut self, value: usize) {
        self.overwrap_removal.iterations = value;
    }

    /// The minimum distance that should be maintained between nodes.
    #[wasm_bindgen(getter = minDistance)]
    pub fn get_min_distance(&self) -> f32 {
        self.overwrap_removal.min_distance
    }

    /// Sets the minimum distance that should be maintained between nodes.
    ///
    /// This can be used to add extra padding beyond the node radii.
    #[wasm_bindgen(setter = minDistance)]
    pub fn set_min_distance(&mut self, value: f32) {
        self.overwrap_removal.min_distance = value;
    }
}
