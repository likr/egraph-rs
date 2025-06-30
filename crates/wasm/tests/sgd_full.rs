mod util;

#[allow(unused_imports)]
use egraph_wasm::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

#[wasm_bindgen(module = "tests/sgd_full.js")]
extern "C" {
    #[wasm_bindgen(js_name = "testFullSgdConstructor")]
    fn test_full_sgd_constructor();
}

/// Test basic instantiation of FullSgd class
#[wasm_bindgen_test]
pub fn full_sgd_constructor() {
    test_full_sgd_constructor();
}
