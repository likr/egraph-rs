use crate::{drawing::JsDrawingEuclidean2d, graph::JsGraph};
use petgraph_edge_bundling_fdeb::{fdeb, EdgeBundlingOptions};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = fdeb)]
pub fn js_fdeb(graph: &JsGraph, drawing: JsDrawingEuclidean2d) -> JsValue {
    let options = EdgeBundlingOptions::<f32>::new();
    let bends = fdeb(graph.graph(), drawing.drawing(), &options)
        .into_iter()
        .map(|(e, lines)| (e.index(), lines))
        .collect::<HashMap<_, _>>();
    serde_wasm_bindgen::to_value(&bends).unwrap()
}
