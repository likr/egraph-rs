#[allow(unused_imports)]
use egraph_wasm::*;
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize)]
struct NodeData {
  id: usize,
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
  JsValue::from_serde(&data).ok().unwrap()
}
