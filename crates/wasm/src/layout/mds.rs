use crate::{
    drawing::{JsDrawingEuclidean, JsDrawingEuclidean2d},
    graph::JsGraph,
};
use js_sys::{Array, Function};
use petgraph::{graph::node_index, stable_graph::NodeIndex, visit::EdgeRef};
use petgraph_layout_mds::{ClassicalMds, PivotMds};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = "ClassicalMds")]
pub struct JsClassicalMds {
    mds: ClassicalMds<NodeIndex>,
}

#[wasm_bindgen(js_class = "ClassicalMds")]
impl JsClassicalMds {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph, length: &Function) -> JsClassicalMds {
        let mut length_map = HashMap::new();
        for e in graph.graph().edge_indices() {
            let c = length
                .call1(&JsValue::null(), &JsValue::from_f64(e.index() as f64))
                .unwrap()
                .as_f64()
                .unwrap() as f32;
            length_map.insert(e, c);
        }
        JsClassicalMds {
            mds: ClassicalMds::new(graph.graph(), |e| length_map[&e.id()]),
        }
    }

    #[wasm_bindgen(js_name = "run2d")]
    pub fn run_2d(&self) -> JsDrawingEuclidean2d {
        JsDrawingEuclidean2d::new(self.mds.run_2d())
    }

    pub fn run(&self, d: usize) -> JsDrawingEuclidean {
        JsDrawingEuclidean::new(self.mds.run(d))
    }
}

#[wasm_bindgen(js_name = "PivotMds")]
pub struct JsPivotMds {
    mds: PivotMds<NodeIndex>,
}

#[wasm_bindgen(js_class = "PivotMds")]
impl JsPivotMds {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph, length: &Function, sources: &Array) -> JsPivotMds {
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
        JsPivotMds {
            mds: PivotMds::new(graph.graph(), |e| length_map[&e.id()], &sources),
        }
    }

    #[wasm_bindgen(js_name = "run2d")]
    pub fn run_2d(&self) -> JsDrawingEuclidean2d {
        JsDrawingEuclidean2d::new(self.mds.run_2d())
    }
}
