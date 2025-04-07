mod util;

#[allow(unused_imports)]
use egraph_wasm::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

#[wasm_bindgen(module = "tests/quality_metrics.js")]
extern "C" {
    #[wasm_bindgen(js_name = "testStress")]
    fn test_stress();
    #[wasm_bindgen(js_name = "testCrossingNumber")]
    fn test_crossing_number();
    #[wasm_bindgen(js_name = "testCrossingNumberWithDrawingTorus2d")]
    fn test_crossing_number_with_drawing_torus_2d();
    #[wasm_bindgen(js_name = "testNeighborhoodPreservation")]
    fn test_neighborhood_preservation();
    #[wasm_bindgen(js_name = "testQualityMetricsIntegration")]
    fn test_quality_metrics_integration();
}

/// Test the stress metric calculation
#[wasm_bindgen_test]
pub fn stress() {
    test_stress();
}

/// Test the crossing number calculation in Euclidean 2D space
#[wasm_bindgen_test]
pub fn crossing_number() {
    test_crossing_number();
}

/// Test the crossing number calculation in torus 2D space
#[wasm_bindgen_test]
pub fn crossing_number_with_drawing_torus_2d() {
    test_crossing_number_with_drawing_torus_2d();
}

/// Test the neighborhood preservation metric
#[wasm_bindgen_test]
pub fn neighborhood_preservation() {
    test_neighborhood_preservation();
}

/// Test integration of quality metrics with layout algorithms
#[wasm_bindgen_test]
pub fn quality_metrics_integration() {
    test_quality_metrics_integration();
}
