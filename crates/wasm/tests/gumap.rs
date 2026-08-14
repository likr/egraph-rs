mod util;

#[allow(unused_imports)]
use egraph_wasm::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

#[wasm_bindgen(module = "tests/gumap.js")]
extern "C" {
    #[wasm_bindgen(js_name = "testGuMapConstructor")]
    fn test_gumap_constructor();

    #[wasm_bindgen(js_name = "testGuMapRun")]
    fn test_gumap_run();
}

#[wasm_bindgen_test]
pub fn gumap_constructor() {
    test_gumap_constructor();
}

#[wasm_bindgen_test]
pub fn gumap_run() {
    test_gumap_run();
}
