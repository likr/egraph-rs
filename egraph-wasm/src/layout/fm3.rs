use super::super::graph::Graph;
use js_sys::{Array, Object, Reflect};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct FM3 {
    #[wasm_bindgen(js_name = minSize)]
    pub min_size: usize,
    #[wasm_bindgen(js_name = stepIteration)]
    pub step_iteration: usize,
    #[wasm_bindgen(js_name = unitEdgeLength)]
    pub unit_edge_length: f64,
    #[wasm_bindgen(js_name = positionForceStrength)]
    pub position_force_strength: f64,
}

#[wasm_bindgen]
impl FM3 {
    #[wasm_bindgen(constructor)]
    pub fn new() -> FM3 {
        let fm3 = egraph::layout::fm3::FM3::new();
        FM3 {
            min_size: fm3.min_size,
            step_iteration: fm3.step_iteration,
            unit_edge_length: fm3.unit_edge_length as f64,
            position_force_strength: fm3.position_force_strength as f64,
        }
    }

    pub fn call(&self, graph: &Graph) -> JsValue {
        let array = Array::new();
        let mut fm3 = egraph::layout::fm3::FM3::new();
        fm3.min_size = self.min_size;
        fm3.step_iteration = self.step_iteration;
        fm3.unit_edge_length = self.unit_edge_length as f32;
        fm3.position_force_strength = self.position_force_strength as f32;
        let points = fm3.call(&graph.graph());
        for point in points.iter() {
            let obj = Object::new();
            Reflect::set(&obj, &"x".into(), &point.x.into())
                .ok()
                .unwrap();
            Reflect::set(&obj, &"y".into(), &point.y.into())
                .ok()
                .unwrap();
            array.push(&obj);
        }
        array.into()
    }
}
