mod util;

#[allow(unused_imports)]
use egraph_wasm::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

#[wasm_bindgen(module = "tests/sgd_sparse.js")]
extern "C" {
    #[wasm_bindgen(js_name = "testSparseSgdConstructor")]
    fn test_sparse_sgd_constructor();
}

/// Test basic instantiation of SparseSgd class
#[wasm_bindgen_test]
pub fn sparse_sgd_constructor() {
    test_sparse_sgd_constructor();
}
