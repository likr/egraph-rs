use egraph::algorithm::ranking::{longest_path, min_width};
use egraph_wasm_adapter::{JsGraph, JsGraphAdapter};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = longestPathRanking)]
pub fn js_longest_path(graph: JsGraph) -> JsValue {
    let graph = JsGraphAdapter::new(graph);
    let result = longest_path(&graph);
    JsValue::from_serde(&result).unwrap()
}

#[wasm_bindgen(js_name = minWidthRanking)]
pub fn js_min_width(graph: JsGraph, size: usize) -> JsValue {
    let graph = JsGraphAdapter::new(graph);
    let result = min_width(&graph, size);
    JsValue::from_serde(&result).unwrap()
}
