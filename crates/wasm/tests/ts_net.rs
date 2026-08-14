mod util;

#[allow(unused_imports)]
use egraph_wasm::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

#[wasm_bindgen(module = "tests/ts_net.js")]
extern "C" {
    #[wasm_bindgen(js_name = "testTsNetConstructor")]
    fn test_ts_net_constructor();

    #[wasm_bindgen(js_name = "testTsNetRun")]
    fn test_ts_net_run();

    #[wasm_bindgen(js_name = "testTsNetWithDiffusionDistance")]
    fn test_ts_net_with_diffusion_distance();

    #[wasm_bindgen(js_name = "testTsNetWithEmbeddingDistance")]
    fn test_ts_net_with_embedding_distance();

    #[wasm_bindgen(js_name = "testTsNetWithKernelDistance")]
    fn test_ts_net_with_kernel_distance();

    #[wasm_bindgen(js_name = "testBhTsNetConstructor")]
    fn test_bh_ts_net_constructor();

    #[wasm_bindgen(js_name = "testBhTsNetRun")]
    fn test_bh_ts_net_run();
}

#[wasm_bindgen_test]
pub fn ts_net_constructor() {
    test_ts_net_constructor();
}

#[wasm_bindgen_test]
pub fn ts_net_run() {
    test_ts_net_run();
}

#[wasm_bindgen_test]
pub fn ts_net_with_diffusion_distance() {
    test_ts_net_with_diffusion_distance();
}

#[wasm_bindgen_test]
pub fn ts_net_with_embedding_distance() {
    test_ts_net_with_embedding_distance();
}

#[wasm_bindgen_test]
pub fn ts_net_with_kernel_distance() {
    test_ts_net_with_kernel_distance();
}

#[wasm_bindgen_test]
pub fn bh_ts_net_constructor() {
    test_bh_ts_net_constructor();
}

#[wasm_bindgen_test]
pub fn bh_ts_net_run() {
    test_bh_ts_net_run();
}
