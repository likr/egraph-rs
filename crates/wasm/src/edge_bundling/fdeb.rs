use crate::graph::JsGraph;
use petgraph::graph::node_index;
use petgraph_edge_bundling_fdeb::{fdeb, EdgeBundlingOptions};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = fdeb)]
pub fn js_fdeb(graph: &JsGraph, coordinates: JsValue) -> JsValue {
    let coordinates: HashMap<usize, (f32, f32)> =
        serde_wasm_bindgen::from_value(coordinates).unwrap();
    let coordinates = coordinates
        .into_iter()
        .map(|(u, xy)| (node_index(u), xy))
        .collect::<HashMap<_, _>>();
    let options = EdgeBundlingOptions::new();
    let bends = fdeb(graph.graph(), &coordinates, &options)
        .into_iter()
        .map(|(e, lines)| (e.index(), lines))
        .collect::<HashMap<_, _>>();
    serde_wasm_bindgen::to_value(&bends).unwrap()
}
