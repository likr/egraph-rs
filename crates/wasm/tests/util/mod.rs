#![allow(dead_code)]

pub mod fixtures;

#[cfg(feature = "legacy")]
#[allow(unused_imports)]
use egraph_wasm::*;
#[cfg(feature = "legacy")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "legacy")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "legacy")]
#[derive(Serialize, Deserialize)]
struct NodeData {
    id: usize,
    group: usize,
}

#[cfg(feature = "legacy")]
#[derive(Serialize, Deserialize)]
struct LinkData {
    source: usize,
    target: usize,
}

#[cfg(feature = "legacy")]
#[derive(Serialize, Deserialize)]
struct GraphData {
    nodes: Vec<NodeData>,
    links: Vec<LinkData>,
}

#[cfg(feature = "legacy")]
pub fn example_data() -> JsValue {
    let s = include_str!("./miserables.json");
    let data = serde_json::from_str::<GraphData>(s).unwrap();
    serde_wasm_bindgen::to_value(&data).ok().unwrap()
}
