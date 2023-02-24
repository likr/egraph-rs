#[allow(unused_imports)]
use egraph_wasm::*;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize)]
struct NodeData {
    id: usize,
    group: usize,
}

#[derive(Serialize, Deserialize)]
struct LinkData {
    source: usize,
    target: usize,
}

#[derive(Serialize, Deserialize)]
struct GraphData {
    nodes: Vec<NodeData>,
    links: Vec<LinkData>,
}

pub fn example_data() -> JsValue {
    let s = include_str!("./miserables.json");
    let data = serde_json::from_str::<GraphData>(&s).unwrap();
    serde_wasm_bindgen::to_value(&data).ok().unwrap()
}
