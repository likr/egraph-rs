mod util;

#[allow(unused_imports)]
use egraph_wasm::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

#[wasm_bindgen(module = "tests/sgd_sparse.js")]
extern "C" {
    #[wasm_bindgen(js_name = "testSparseSgdConstructor")]
    fn test_sparse_sgd_constructor();
    #[wasm_bindgen(js_name = "testSparseSgdWithDifferentPivots")]
    fn test_sparse_sgd_with_different_pivots();
    #[wasm_bindgen(js_name = "testSparseSgdSchedulers")]
    fn test_sparse_sgd_schedulers();
    #[wasm_bindgen(js_name = "testSparseSgdWithEuclidean2d")]
    fn test_sparse_sgd_with_euclidean_2d();
    #[wasm_bindgen(js_name = "testSparseSgdWithHyperbolic2d")]
    fn test_sparse_sgd_with_hyperbolic_2d();
    #[wasm_bindgen(js_name = "testSparseSgdWithSpherical2d")]
    fn test_sparse_sgd_with_spherical_2d();
    #[wasm_bindgen(js_name = "testSparseSgdWithTorus2d")]
    fn test_sparse_sgd_with_torus_2d();
    #[wasm_bindgen(js_name = "testSparseSgdWithEuclidean")]
    fn test_sparse_sgd_with_euclidean();
    #[wasm_bindgen(js_name = "testSparseSgdUpdateDistance")]
    fn test_sparse_sgd_update_distance();
    #[wasm_bindgen(js_name = "testSparseSgdUpdateWeight")]
    fn test_sparse_sgd_update_weight();
    #[wasm_bindgen(js_name = "testSparseSgdShuffle")]
    fn test_sparse_sgd_shuffle();
    #[wasm_bindgen(js_name = "testSparseSgdIntegration")]
    fn test_sparse_sgd_integration();
}

/// Test basic instantiation of SparseSgd class
#[wasm_bindgen_test]
pub fn sparse_sgd_constructor() {
    test_sparse_sgd_constructor();
}

/// Test SparseSgd with different numbers of pivot nodes
#[wasm_bindgen_test]
pub fn sparse_sgd_with_different_pivots() {
    test_sparse_sgd_with_different_pivots();
}

/// Test scheduler creation methods
#[wasm_bindgen_test]
pub fn sparse_sgd_schedulers() {
    test_sparse_sgd_schedulers();
}

/// Test applying SGD to Euclidean 2D drawings
#[wasm_bindgen_test]
pub fn sparse_sgd_with_euclidean_2d() {
    test_sparse_sgd_with_euclidean_2d();
}

/// Test applying SGD to Hyperbolic 2D drawings
#[wasm_bindgen_test]
pub fn sparse_sgd_with_hyperbolic_2d() {
    test_sparse_sgd_with_hyperbolic_2d();
}

/// Test applying SGD to Spherical 2D drawings
#[wasm_bindgen_test]
pub fn sparse_sgd_with_spherical_2d() {
    test_sparse_sgd_with_spherical_2d();
}

/// Test applying SGD to Torus 2D drawings
#[wasm_bindgen_test]
pub fn sparse_sgd_with_torus_2d() {
    test_sparse_sgd_with_torus_2d();
}

/// Test applying SGD to n-dimensional Euclidean drawings
#[wasm_bindgen_test]
pub fn sparse_sgd_with_euclidean() {
    test_sparse_sgd_with_euclidean();
}

/// Test updating distance function
#[wasm_bindgen_test]
pub fn sparse_sgd_update_distance() {
    test_sparse_sgd_update_distance();
}

/// Test updating weight function
#[wasm_bindgen_test]
pub fn sparse_sgd_update_weight() {
    test_sparse_sgd_update_weight();
}

/// Test shuffling node pairs
#[wasm_bindgen_test]
pub fn sparse_sgd_shuffle() {
    test_sparse_sgd_shuffle();
}

/// Test integration with other components
#[wasm_bindgen_test]
pub fn sparse_sgd_integration() {
    test_sparse_sgd_integration();
}
