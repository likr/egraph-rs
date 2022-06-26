use crate::graph::JsGraph;
use crate::layout::force_simulation::coordinates::JsCoordinates;
use js_sys::Function;
use petgraph::visit::EdgeRef;
use petgraph_layout_sgd::Sgd;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = "Sgd")]
pub struct JsSgd {
    sgd: Sgd,
}

#[wasm_bindgen(js_class = "Sgd")]
impl JsSgd {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph, length: &Function) -> JsSgd {
        let mut length_map = HashMap::new();
        for e in graph.graph().edge_indices() {
            let c = length
                .call1(&JsValue::null(), &JsValue::from_f64(e.index() as f64))
                .unwrap()
                .as_f64()
                .unwrap() as f32;
            length_map.insert(e, c);
        }
        JsSgd {
            sgd: Sgd::new(graph.graph(), &mut |e| length_map[&e.id()]),
        }
    }

    pub fn apply(&mut self, coordinates: &mut JsCoordinates) {
        self.sgd.apply(coordinates.coordinates_mut());
    }
}
