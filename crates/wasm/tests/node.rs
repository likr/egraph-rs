#[macro_use]
extern crate serde_derive;

mod util;

#[allow(unused_imports)]
use egraph_wasm::*;
use util::example_data;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

#[wasm_bindgen(module = "tests/node.js")]
extern "C" {
  #[wasm_bindgen(js_name = "testConstructGraph")]
  fn test_construct_graph(data: JsValue);
  #[wasm_bindgen(js_name = "testSimulation")]
  fn test_simulation(data: JsValue);
  #[wasm_bindgen(js_name = "testCenterForce")]
  fn test_center_force(data: JsValue);
  #[wasm_bindgen(js_name = "testCollideForce")]
  fn test_collide_force(data: JsValue);
  #[wasm_bindgen(js_name = "testLinkForce")]
  fn test_link_force(data: JsValue);
  #[wasm_bindgen(js_name = "testManyBodyForce")]
  fn test_many_body_force(data: JsValue);
  #[wasm_bindgen(js_name = "testPositionForce")]
  fn test_position_force(data: JsValue);
  #[wasm_bindgen(js_name = "testRadialForce")]
  fn test_radial_force(data: JsValue);
  #[wasm_bindgen(js_name = "testKamadaKawai")]
  fn test_kamada_kawai(data: JsValue);
  #[wasm_bindgen(js_name = "testStressMajorization")]
  fn test_stress_majorization(data: JsValue);
}

#[wasm_bindgen_test]
pub fn construct_graph() {
  let data = example_data();
  test_construct_graph(data);
}

#[wasm_bindgen_test]
pub fn simulation() {
  let data = example_data();
  test_simulation(data);
}

#[wasm_bindgen_test]
pub fn center_force() {
  let data = example_data();
  test_center_force(data);
}
#[wasm_bindgen_test]
pub fn collide_force() {
  let data = example_data();
  test_collide_force(data);
}

#[wasm_bindgen_test]
pub fn link_force() {
  let data = example_data();
  test_link_force(data);
}

#[wasm_bindgen_test]
pub fn many_body_force() {
  let data = example_data();
  test_many_body_force(data);
}

#[wasm_bindgen_test]
pub fn position_force() {
  let data = example_data();
  test_position_force(data);
}

#[wasm_bindgen_test]
pub fn radial_force() {
  let data = example_data();
  test_radial_force(data);
}

#[wasm_bindgen_test]
pub fn kamada_kawai() {
  let data = example_data();
  test_kamada_kawai(data);
}

#[wasm_bindgen_test]
pub fn stress_majorization() {
  let data = example_data();
  test_stress_majorization(data);
}
