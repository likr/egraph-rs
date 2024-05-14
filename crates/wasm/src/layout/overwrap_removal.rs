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

#[wasm_bindgen(js_name = OverwrapRemoval)]
pub struct JsOverwrapRemoval {
    overwrap_removal: OverwrapRemoval<f32>,
}

#[wasm_bindgen(js_class = OverwrapRemoval)]
impl JsOverwrapRemoval {
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

    #[wasm_bindgen(js_name = "applyWithDrawingEuclidean2d")]
    pub fn apply_with_drawing_euclidean_2d(&self, drawing: &mut JsDrawingEuclidean2d) {
        self.overwrap_removal.apply(drawing.drawing_mut());
    }

    #[wasm_bindgen(js_name = "applyWithDrawingEuclidean")]
    pub fn apply_with_drawing_euclidean(&self, drawing: &mut JsDrawingEuclidean) {
        self.overwrap_removal.apply(drawing.drawing_mut());
    }

    #[wasm_bindgen(js_name = "applyWithDrawingHyperbolic2d")]
    pub fn apply_with_drawing_hyperbolic_2d(&self, drawing: &mut JsDrawingHyperbolic2d) {
        self.overwrap_removal.apply(drawing.drawing_mut());
    }

    #[wasm_bindgen(js_name = "applyWithDrawingSpherical2d")]
    pub fn apply_with_drawing_spherical_2d(&self, drawing: &mut JsDrawingSpherical2d) {
        self.overwrap_removal.apply(drawing.drawing_mut());
    }

    #[wasm_bindgen(js_name = "applyWithDrawingTorus2d")]
    pub fn apply_with_drawing_torus_2d(&self, drawing: &mut JsDrawingTorus2d) {
        self.overwrap_removal.apply(drawing.drawing_mut());
    }

    #[wasm_bindgen(getter)]
    pub fn get_strength(&self) -> f32 {
        self.overwrap_removal.strength
    }

    #[wasm_bindgen(setter)]
    pub fn set_strength(&mut self, value: f32) {
        self.overwrap_removal.strength = value;
    }

    #[wasm_bindgen(getter)]
    pub fn get_iterations(&self) -> usize {
        self.overwrap_removal.iterations
    }

    #[wasm_bindgen(setter)]
    pub fn set_iterations(&mut self, value: usize) {
        self.overwrap_removal.iterations = value;
    }

    #[wasm_bindgen(getter = minDistance)]
    pub fn get_min_distance(&self) -> f32 {
        self.overwrap_removal.min_distance
    }

    #[wasm_bindgen(setter = minDistance)]
    pub fn set_min_distance(&mut self, value: f32) {
        self.overwrap_removal.min_distance = value;
    }
}
