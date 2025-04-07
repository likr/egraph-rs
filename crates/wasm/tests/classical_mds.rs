mod util;

#[allow(unused_imports)]
use egraph_wasm::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

#[wasm_bindgen(module = "tests/classical_mds.js")]
extern "C" {
    #[wasm_bindgen(js_name = "testClassicalMdsConstructor")]
    fn test_classical_mds_constructor();
    #[wasm_bindgen(js_name = "testClassicalMdsRun2d")]
    fn test_classical_mds_run_2d();
    #[wasm_bindgen(js_name = "testClassicalMdsRun")]
    fn test_classical_mds_run();
    #[wasm_bindgen(js_name = "testClassicalMdsWithDifferentGraphs")]
    fn test_classical_mds_with_different_graphs();
    #[wasm_bindgen(js_name = "testClassicalMdsWithCustomLengthFunction")]
    fn test_classical_mds_with_custom_length_function();
    #[wasm_bindgen(js_name = "testClassicalMdsHandlesHighDimensions")]
    fn test_classical_mds_handles_high_dimensions();
    #[wasm_bindgen(js_name = "testClassicalMdsIntegration")]
    fn test_classical_mds_integration();
}

/// Test basic instantiation of ClassicalMds class
#[wasm_bindgen_test]
pub fn classical_mds_constructor() {
    test_classical_mds_constructor();
}

/// Test run2d method for 2D layout generation
#[wasm_bindgen_test]
pub fn classical_mds_run_2d() {
    test_classical_mds_run_2d();
}

/// Test run method for n-dimensional layout generation
#[wasm_bindgen_test]
pub fn classical_mds_run() {
    test_classical_mds_run();
}

/// Test with different graph structures
#[wasm_bindgen_test]
pub fn classical_mds_with_different_graphs() {
    test_classical_mds_with_different_graphs();
}

/// Test with custom length function
#[wasm_bindgen_test]
pub fn classical_mds_with_custom_length_function() {
    test_classical_mds_with_custom_length_function();
}

/// Test handling of high-dimensional embeddings
#[wasm_bindgen_test]
pub fn classical_mds_handles_high_dimensions() {
    test_classical_mds_handles_high_dimensions();
}

/// Test integration with other components
#[wasm_bindgen_test]
pub fn classical_mds_integration() {
    test_classical_mds_integration();
}
