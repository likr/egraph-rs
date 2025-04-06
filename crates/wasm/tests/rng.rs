mod util;

#[allow(unused_imports)]
use egraph_wasm::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

#[wasm_bindgen(module = "tests/rng.js")]
extern "C" {
    #[wasm_bindgen(js_name = "testRngConstructor")]
    fn test_rng_constructor();
    #[wasm_bindgen(js_name = "testRngSeedFrom")]
    fn test_rng_seed_from();
    #[wasm_bindgen(js_name = "testRngDifferentSeeds")]
    fn test_rng_different_seeds();
    #[wasm_bindgen(js_name = "testRngWithSgdLayout")]
    fn test_rng_with_sgd_layout();
}

/// Test basic instantiation of Rng class
#[wasm_bindgen_test]
pub fn rng_constructor() {
    test_rng_constructor();
}

/// Test seeded random number generation
/// This test verifies that the same seed produces the same sequence of random numbers
#[wasm_bindgen_test]
pub fn rng_seed_from() {
    test_rng_seed_from();
}

/// Test that different seeds produce different sequences
#[wasm_bindgen_test]
pub fn rng_different_seeds() {
    test_rng_different_seeds();
}

/// Test integration with SGD layout algorithm
#[wasm_bindgen_test]
pub fn rng_with_sgd_layout() {
    test_rng_with_sgd_layout();
}
