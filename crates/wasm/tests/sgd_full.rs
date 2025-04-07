mod util;

#[allow(unused_imports)]
use egraph_wasm::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

#[wasm_bindgen(module = "tests/sgd_full.js")]
extern "C" {
    #[wasm_bindgen(js_name = "testFullSgdConstructor")]
    fn test_full_sgd_constructor();
    #[wasm_bindgen(js_name = "testFullSgdSchedulers")]
    fn test_full_sgd_schedulers();
    #[wasm_bindgen(js_name = "testFullSgdWithEuclidean2d")]
    fn test_full_sgd_with_euclidean_2d();
    #[wasm_bindgen(js_name = "testFullSgdWithHyperbolic2d")]
    fn test_full_sgd_with_hyperbolic_2d();
    #[wasm_bindgen(js_name = "testFullSgdWithSpherical2d")]
    fn test_full_sgd_with_spherical_2d();
    #[wasm_bindgen(js_name = "testFullSgdWithTorus2d")]
    fn test_full_sgd_with_torus_2d();
    #[wasm_bindgen(js_name = "testFullSgdWithEuclidean")]
    fn test_full_sgd_with_euclidean();
    #[wasm_bindgen(js_name = "testFullSgdUpdateDistance")]
    fn test_full_sgd_update_distance();
    #[wasm_bindgen(js_name = "testFullSgdUpdateWeight")]
    fn test_full_sgd_update_weight();
    #[wasm_bindgen(js_name = "testFullSgdShuffle")]
    fn test_full_sgd_shuffle();
    #[wasm_bindgen(js_name = "testFullSgdIntegration")]
    fn test_full_sgd_integration();
}

/// Test basic instantiation of FullSgd class
#[wasm_bindgen_test]
pub fn full_sgd_constructor() {
    test_full_sgd_constructor();
}

/// Test scheduler creation methods
#[wasm_bindgen_test]
pub fn full_sgd_schedulers() {
    test_full_sgd_schedulers();
}

/// Test applying SGD to Euclidean 2D drawings
#[wasm_bindgen_test]
pub fn full_sgd_with_euclidean_2d() {
    test_full_sgd_with_euclidean_2d();
}

/// Test applying SGD to Hyperbolic 2D drawings
#[wasm_bindgen_test]
pub fn full_sgd_with_hyperbolic_2d() {
    test_full_sgd_with_hyperbolic_2d();
}

/// Test applying SGD to Spherical 2D drawings
#[wasm_bindgen_test]
pub fn full_sgd_with_spherical_2d() {
    test_full_sgd_with_spherical_2d();
}

/// Test applying SGD to Torus 2D drawings
#[wasm_bindgen_test]
pub fn full_sgd_with_torus_2d() {
    test_full_sgd_with_torus_2d();
}

/// Test applying SGD to n-dimensional Euclidean drawings
#[wasm_bindgen_test]
pub fn full_sgd_with_euclidean() {
    test_full_sgd_with_euclidean();
}

/// Test updating distance function
#[wasm_bindgen_test]
pub fn full_sgd_update_distance() {
    test_full_sgd_update_distance();
}

/// Test updating weight function
#[wasm_bindgen_test]
pub fn full_sgd_update_weight() {
    test_full_sgd_update_weight();
}

/// Test shuffling node pairs
#[wasm_bindgen_test]
pub fn full_sgd_shuffle() {
    test_full_sgd_shuffle();
}

/// Test integration with other components
#[wasm_bindgen_test]
pub fn full_sgd_integration() {
    test_full_sgd_integration();
}
