mod util;

#[allow(unused_imports)]
use egraph_wasm::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

#[wasm_bindgen(module = "tests/edge_bundling.js")]
extern "C" {
    #[wasm_bindgen(js_name = "testFdebBasic")]
    fn test_fdeb_basic();
    #[wasm_bindgen(js_name = "testFdebWithComplexGraph")]
    fn test_fdeb_with_complex_graph();
    #[wasm_bindgen(js_name = "testFdebResultStructure")]
    fn test_fdeb_result_structure();
    #[wasm_bindgen(js_name = "testFdebIntegration")]
    fn test_fdeb_integration();
}

/// Test basic functionality of the FDEB algorithm
#[wasm_bindgen_test]
pub fn fdeb_basic() {
    test_fdeb_basic();
}

/// Test FDEB with a more complex graph
#[wasm_bindgen_test]
pub fn fdeb_with_complex_graph() {
    test_fdeb_with_complex_graph();
}

/// Test the structure of the FDEB result
#[wasm_bindgen_test]
pub fn fdeb_result_structure() {
    test_fdeb_result_structure();
}

/// Test integration of FDEB with other components
#[wasm_bindgen_test]
pub fn fdeb_integration() {
    test_fdeb_integration();
}
