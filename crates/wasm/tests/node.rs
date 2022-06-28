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
    #[wasm_bindgen(js_name = "testCoordinates")]
    fn test_coordinates(data: JsValue);
    #[wasm_bindgen(js_name = "testSimulation")]
    fn test_simulation(data: JsValue);
    #[wasm_bindgen(js_name = "testForceDirectedLayout")]
    fn test_force_directed_layout(data: JsValue);
    #[wasm_bindgen(js_name = "testHyperbolicForceDirectedLayout")]
    fn test_hypaerbolic_force_directed_layout(data: JsValue);
    #[wasm_bindgen(js_name = "testCollideForce")]
    fn test_collide_force(data: JsValue);
    #[wasm_bindgen(js_name = "testLinkForce")]
    fn test_link_force(data: JsValue);
    #[wasm_bindgen(js_name = "testManyBodyForce")]
    fn test_many_body_force(data: JsValue);
    #[wasm_bindgen(js_name = "testPositionForce")]
    fn test_position_force(data: JsValue);
    #[wasm_bindgen(js_name = "testGroupLinkForce")]
    fn test_group_link_force(data: JsValue);
    #[wasm_bindgen(js_name = "testGroupManyBodyForce")]
    fn test_group_many_body_force(data: JsValue);
    #[wasm_bindgen(js_name = "testGroupPositionForce")]
    fn test_group_position_force(data: JsValue);
    #[wasm_bindgen(js_name = "testRadialForce")]
    fn test_radial_force(data: JsValue);
    #[wasm_bindgen(js_name = "testKamadaKawai")]
    fn test_kamada_kawai(data: JsValue);
    #[wasm_bindgen(js_name = "testStressMajorization")]
    fn test_stress_majorization(data: JsValue);
    #[wasm_bindgen(js_name = "testClassicalMds")]
    fn test_classical_mds(data: JsValue);
    #[wasm_bindgen(js_name = "testPivotMds")]
    fn test_pivot_mds(data: JsValue);
    #[wasm_bindgen(js_name = "testFullSgd")]
    fn test_full_sgd(data: JsValue);
    #[wasm_bindgen(js_name = "testSparseSgd")]
    fn test_sparse_sgd(data: JsValue);
    #[wasm_bindgen(js_name = "testCoarsen")]
    fn test_coarsen(data: JsValue);
    #[wasm_bindgen(js_name = "testNumberOfCrossings")]
    fn test_number_of_crossings(data: JsValue);
    #[wasm_bindgen(js_name = "testShapeQuality")]
    fn test_shape_quality(data: JsValue);
    #[wasm_bindgen(js_name = "testStress")]
    fn test_stress(data: JsValue);
}

#[wasm_bindgen_test]
pub fn construct_graph() {
    let data = example_data();
    test_construct_graph(data);
}

#[wasm_bindgen_test]
pub fn coordinates() {
    let data = example_data();
    test_coordinates(data);
}

#[wasm_bindgen_test]
pub fn simulation() {
    let data = example_data();
    test_simulation(data);
}

#[wasm_bindgen_test]
pub fn force_directed_layout() {
    let data = example_data();
    test_force_directed_layout(data);
}

#[wasm_bindgen_test]
pub fn hyperbolic_force_directed_layout() {
    let data = example_data();
    test_hypaerbolic_force_directed_layout(data);
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
pub fn group_link_force() {
    let data = example_data();
    test_group_link_force(data);
}

#[wasm_bindgen_test]
pub fn group_many_body_force() {
    let data = example_data();
    test_group_many_body_force(data);
}

#[wasm_bindgen_test]
pub fn group_position_force() {
    let data = example_data();
    test_group_position_force(data);
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

#[wasm_bindgen_test]
pub fn classical_mds() {
    let data = example_data();
    test_classical_mds(data);
}

#[wasm_bindgen_test]
pub fn pivot_mds() {
    let data = example_data();
    test_pivot_mds(data);
}

#[wasm_bindgen_test]
pub fn full_sgd() {
    let data = example_data();
    test_full_sgd(data);
}

#[wasm_bindgen_test]
pub fn sparse_sgd() {
    let data = example_data();
    test_sparse_sgd(data);
}

#[wasm_bindgen_test]
pub fn coarsen() {
    let data = example_data();
    test_coarsen(data);
}

#[wasm_bindgen_test]
pub fn number_of_crossings() {
    let data = example_data();
    test_number_of_crossings(data);
}

#[wasm_bindgen_test]
pub fn shape_quality() {
    let data = example_data();
    test_shape_quality(data);
}

#[wasm_bindgen_test]
pub fn stress() {
    let data = example_data();
    test_stress(data);
}
