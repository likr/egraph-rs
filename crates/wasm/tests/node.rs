#[macro_use]
extern crate serde_derive;

#[allow(unused_imports)]
use egraph_wasm::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

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

#[wasm_bindgen(module = "tests/node.js")]
extern "C" {
	#[wasm_bindgen(js_name = "testConstructGraph")]
	fn test_construct_graph(data: JsValue);
}

#[wasm_bindgen_test]
pub fn construct_graph() {
	let s = include_str!("./miserables.json");
	let data = serde_json::from_str::<GraphData>(&s).unwrap();
	let data = JsValue::from_serde(&data).ok().unwrap();
	test_construct_graph(data);
}
