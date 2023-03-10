use crate::{drawing::JsDrawing, graph::JsGraph};
use js_sys::{Function, Reflect};
use petgraph::visit::EdgeRef;
use petgraph_layout_kamada_kawai::KamadaKawai;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = KamadaKawai)]
pub struct JsKamadaKawai {
    kamada_kawai: KamadaKawai,
}

#[wasm_bindgen(js_class = KamadaKawai)]
impl JsKamadaKawai {
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

    #[wasm_bindgen(js_name = selectNode)]
    pub fn select_node(&self, drawing: &JsDrawing) -> Option<usize> {
        self.kamada_kawai.select_node(drawing.drawing())
    }

    #[wasm_bindgen(js_name = applyToNode)]
    pub fn apply_to_node(&self, m: usize, drawing: &mut JsDrawing) {
        self.kamada_kawai.apply_to_node(m, drawing.drawing_mut());
    }

    pub fn run(&self, drawing: &mut JsDrawing) {
        self.kamada_kawai.run(drawing.drawing_mut());
    }

    #[wasm_bindgen(getter)]
    pub fn eps(&self) -> f32 {
        self.kamada_kawai.eps
    }

    #[wasm_bindgen(setter)]
    pub fn set_eps(&mut self, value: f32) {
        self.kamada_kawai.eps = value;
    }
}
