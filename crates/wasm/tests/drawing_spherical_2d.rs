mod util;

#[allow(unused_imports)]
use egraph_wasm::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

#[wasm_bindgen(module = "tests/drawing_spherical_2d.js")]
extern "C" {
    #[wasm_bindgen(js_name = "testDrawingSpherical2dConstructor")]
    fn test_drawing_spherical_2d_constructor();
    #[wasm_bindgen(js_name = "testNodeCoordinates")]
    fn test_node_coordinates();
    #[wasm_bindgen(js_name = "testDrawingWithGraph")]
    fn test_drawing_with_graph();
}

/// Test basic instantiation of DrawingSpherical2d class
#[wasm_bindgen_test]
pub fn drawing_spherical_2d_constructor() {
    test_drawing_spherical_2d_constructor();
}

/// Test node coordinate operations (get/set longitude,latitude)
#[wasm_bindgen_test]
pub fn node_coordinates() {
    test_node_coordinates();
}

/// Test integration with Graph class
#[wasm_bindgen_test]
pub fn drawing_with_graph() {
    test_drawing_with_graph();
}
