use crate::{drawing::JsDrawing, graph::JsGraph};
use js_sys::{Function, Reflect};
use petgraph::visit::EdgeRef;
use petgraph_layout_stress_majorization::StressMajorization;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = StressMajorization)]
pub struct JsStressMajorization {
    stress_majorization: StressMajorization,
}

#[wasm_bindgen(js_class = StressMajorization)]
impl JsStressMajorization {
    #[wasm_bindgen(constructor)]
    pub fn new(
        graph: &JsGraph,
        drawing: &JsDrawing,
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

    pub fn apply(&mut self, drawing: &mut JsDrawing) -> f32 {
        self.stress_majorization.apply(drawing.drawing_mut())
    }

    pub fn run(&mut self, drawing: &mut JsDrawing) {
        self.stress_majorization.run(drawing.drawing_mut());
    }
}
