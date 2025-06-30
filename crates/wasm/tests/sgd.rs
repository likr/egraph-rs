mod util;

#[allow(unused_imports)]
use egraph_wasm::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

#[wasm_bindgen(module = "tests/sgd.js")]
extern "C" {
    #[wasm_bindgen(js_name = "testSgdSchedulers")]
    fn test_sgd_schedulers();
    #[wasm_bindgen(js_name = "testSgdWithEuclidean2d")]
    fn test_sgd_with_euclidean_2d();
    #[wasm_bindgen(js_name = "testSgdWithHyperbolic2d")]
    fn test_sgd_with_hyperbolic_2d();
    #[wasm_bindgen(js_name = "testSgdWithSpherical2d")]
    fn test_sgd_with_spherical_2d();
    #[wasm_bindgen(js_name = "testSgdWithTorus2d")]
    fn test_sgd_with_torus_2d();
    #[wasm_bindgen(js_name = "testSgdWithEuclidean")]
    fn test_sgd_with_euclidean();
    #[wasm_bindgen(js_name = "testSgdUpdateDistance")]
    fn test_sgd_update_distance();
    #[wasm_bindgen(js_name = "testSgdUpdateWeight")]
    fn test_sgd_update_weight();
    #[wasm_bindgen(js_name = "testSgdShuffle")]
    fn test_sgd_shuffle();
}

/// Test scheduler creation methods
#[wasm_bindgen_test]
pub fn sgd_schedulers() {
    test_sgd_schedulers();
}

/// Test applying SGD to Euclidean 2D drawings
#[wasm_bindgen_test]
pub fn sgd_with_euclidean_2d() {
    test_sgd_with_euclidean_2d();
}

/// Test applying SGD to Hyperbolic 2D drawings
#[wasm_bindgen_test]
pub fn sgd_with_hyperbolic_2d() {
    test_sgd_with_hyperbolic_2d();
}

/// Test applying SGD to Spherical 2D drawings
#[wasm_bindgen_test]
pub fn sgd_with_spherical_2d() {
    test_sgd_with_spherical_2d();
}

/// Test applying SGD to Torus 2D drawings
#[wasm_bindgen_test]
pub fn sgd_with_torus_2d() {
    test_sgd_with_torus_2d();
}

/// Test applying SGD to n-dimensional Euclidean drawings
#[wasm_bindgen_test]
pub fn sgd_with_euclidean() {
    test_sgd_with_euclidean();
}

/// Test updating distance function
#[wasm_bindgen_test]
pub fn sgd_update_distance() {
    test_sgd_update_distance();
}

/// Test updating weight function
#[wasm_bindgen_test]
pub fn sgd_update_weight() {
    test_sgd_update_weight();
}

/// Test shuffling node pairs
#[wasm_bindgen_test]
pub fn sgd_shuffle() {
    test_sgd_shuffle();
}
