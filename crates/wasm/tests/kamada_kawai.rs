mod util;

#[allow(unused_imports)]
use egraph_wasm::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

#[wasm_bindgen(module = "tests/kamada_kawai.js")]
extern "C" {
    #[wasm_bindgen(js_name = "testKamadaKawaiConstructor")]
    fn test_kamada_kawai_constructor();
    #[wasm_bindgen(js_name = "testKamadaKawaiEpsilon")]
    fn test_kamada_kawai_epsilon();
    #[wasm_bindgen(js_name = "testKamadaKawaiSelectNode")]
    fn test_kamada_kawai_select_node();
    #[wasm_bindgen(js_name = "testKamadaKawaiApplyToNode")]
    fn test_kamada_kawai_apply_to_node();
    #[wasm_bindgen(js_name = "testKamadaKawaiRun")]
    fn test_kamada_kawai_run();
    #[wasm_bindgen(js_name = "testKamadaKawaiIntegration")]
    fn test_kamada_kawai_integration();
}

/// Test basic instantiation of KamadaKawai class
#[wasm_bindgen_test]
pub fn kamada_kawai_constructor() {
    test_kamada_kawai_constructor();
}

/// Test epsilon parameter getter and setter
#[wasm_bindgen_test]
pub fn kamada_kawai_epsilon() {
    test_kamada_kawai_epsilon();
}

/// Test node selection functionality
#[wasm_bindgen_test]
pub fn kamada_kawai_select_node() {
    test_kamada_kawai_select_node();
}

/// Test applying the algorithm to a single node
#[wasm_bindgen_test]
pub fn kamada_kawai_apply_to_node() {
    test_kamada_kawai_apply_to_node();
}

/// Test running the complete algorithm
#[wasm_bindgen_test]
pub fn kamada_kawai_run() {
    test_kamada_kawai_run();
}

/// Test integration with other components
#[wasm_bindgen_test]
pub fn kamada_kawai_integration() {
    test_kamada_kawai_integration();
}
