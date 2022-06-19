use crate::graph::JsGraph;
use crate::layout::force_simulation::coordinates::JsCoordinates;
use js_sys::{Array, Function};
use petgraph::graph::node_index;
use petgraph_layout_mds::{ClassicalMds, PivotMds};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = "ClassicalMds")]
pub struct JsClassicalMds {
    classical_mds: ClassicalMds,
}

#[wasm_bindgen(js_class = "ClassicalMds")]
impl JsClassicalMds {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsClassicalMds {
        JsClassicalMds {
            classical_mds: ClassicalMds::new(),
        }
    }

    pub fn run(&self, graph: &JsGraph, length: &Function) -> JsCoordinates {
        let mut length_map = HashMap::new();
        for e in graph.graph().edge_indices() {
            let c = length
                .call1(&JsValue::null(), &JsValue::from_f64(e.index() as f64))
                .unwrap()
                .as_f64()
                .unwrap() as f32;
            length_map.insert(e, c);
        }
        let coordinates = self
            .classical_mds
            .run(graph.graph(), &mut |_, e| length_map[&e]);
        JsCoordinates::new(coordinates)
    }
}

#[wasm_bindgen(js_name = "PivotMds")]
pub struct JsPivotMds {
    pivot_mds: PivotMds,
}

#[wasm_bindgen(js_class = "PivotMds")]
impl JsPivotMds {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsPivotMds {
        JsPivotMds {
            pivot_mds: PivotMds::new(),
        }
    }

    pub fn run(&self, graph: &JsGraph, length: &Function, sources: &Array) -> JsCoordinates {
        let sources = sources
            .iter()
            .map(|item| node_index(item.as_f64().unwrap() as usize))
            .collect::<Vec<_>>();
        let mut length_map = HashMap::new();
        for e in graph.graph().edge_indices() {
            let c = length
                .call1(&JsValue::null(), &JsValue::from_f64(e.index() as f64))
                .unwrap()
                .as_f64()
                .unwrap() as f32;
            length_map.insert(e, c);
        }
        let coordinates = self
            .pivot_mds
            .run(graph.graph(), &mut |_, e| length_map[&e], &sources);
        JsCoordinates::new(coordinates)
    }
}
