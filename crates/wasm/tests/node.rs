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
    #[wasm_bindgen(js_name = "testCrossingNumber")]
    fn test_crossing_number(data: JsValue);
    #[wasm_bindgen(js_name = "testNeighborhoodPreservation")]
    fn test_neighborhood_preservation(data: JsValue);
    #[wasm_bindgen(js_name = "testStress")]
    fn test_stress(data: JsValue);
}

#[wasm_bindgen_test]
pub fn construct_graph() {
    let data = example_data();
    test_construct_graph(data);
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
pub fn crossing_number() {
    let data = example_data();
    test_crossing_number(data);
}

#[wasm_bindgen_test]
pub fn neighborhood_preservation() {
    let data = example_data();
    test_neighborhood_preservation(data);
}

#[wasm_bindgen_test]
pub fn stress() {
    let data = example_data();
    test_stress(data);
}
