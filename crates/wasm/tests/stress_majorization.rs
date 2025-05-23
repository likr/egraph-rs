mod util;

#[allow(unused_imports)]
use egraph_wasm::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

#[wasm_bindgen(module = "tests/stress_majorization.js")]
extern "C" {
    #[wasm_bindgen(js_name = "testStressMajorizationConstructor")]
    fn test_stress_majorization_constructor();
    #[wasm_bindgen(js_name = "testStressMajorizationApply")]
    fn test_stress_majorization_apply();
    #[wasm_bindgen(js_name = "testStressMajorizationRun")]
    fn test_stress_majorization_run();
    #[wasm_bindgen(js_name = "testStressMajorizationIntegration")]
    fn test_stress_majorization_integration();
    #[wasm_bindgen(js_name = "testStressMajorizationParameters")]
    fn test_stress_majorization_parameters();
}

/// Test basic instantiation of StressMajorization class
#[wasm_bindgen_test]
pub fn stress_majorization_constructor() {
    test_stress_majorization_constructor();
}

/// Test applying a single iteration of the stress majorization algorithm
#[wasm_bindgen_test]
pub fn stress_majorization_apply() {
    test_stress_majorization_apply();
}

/// Test running the complete stress majorization algorithm
#[wasm_bindgen_test]
pub fn stress_majorization_run() {
    test_stress_majorization_run();
}

/// Test integration with other components and stress reduction
#[wasm_bindgen_test]
pub fn stress_majorization_integration() {
    test_stress_majorization_integration();
}

/// Test getter and setter methods for epsilon and max_iterations
#[wasm_bindgen_test]
pub fn stress_majorization_parameters() {
    test_stress_majorization_parameters();
}
