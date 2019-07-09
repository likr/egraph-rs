pub mod biclustering;

use egraph::algorithm::connected_components;
use egraph_wasm_adapter::{JsGraph, JsGraphAdapter};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = connected_components)]
pub fn js_connected_components(graph: JsGraph) -> JsValue {
    let graph = JsGraphAdapter::new(graph);
    let components = connected_components(&graph);
    JsValue::from_serde(&components).unwrap()
}
