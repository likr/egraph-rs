mod util;

#[allow(unused_imports)]
use egraph_wasm::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

#[wasm_bindgen(module = "tests/drawing_euclidean_2d.js")]
extern "C" {
    #[wasm_bindgen(js_name = "testDrawingEuclidean2dConstructor")]
    fn test_drawing_euclidean_2d_constructor();
    #[wasm_bindgen(js_name = "testNodeCoordinates")]
    fn test_node_coordinates();
    #[wasm_bindgen(js_name = "testDrawingManipulation")]
    fn test_drawing_manipulation();
    #[wasm_bindgen(js_name = "testEdgeSegments")]
    fn test_edge_segments();
    #[wasm_bindgen(js_name = "testDrawingWithGraph")]
    fn test_drawing_with_graph();
}

/// Test basic instantiation of DrawingEuclidean2d class
#[wasm_bindgen_test]
pub fn drawing_euclidean_2d_constructor() {
    test_drawing_euclidean_2d_constructor();
}

/// Test node coordinate operations (get/set x,y)
#[wasm_bindgen_test]
pub fn node_coordinates() {
    test_node_coordinates();
}

/// Test drawing manipulation (centralize, clamp_region)
#[wasm_bindgen_test]
pub fn drawing_manipulation() {
    test_drawing_manipulation();
}

/// Test edge segment representation
#[wasm_bindgen_test]
pub fn edge_segments() {
    test_edge_segments();
}

/// Test integration with Graph class
#[wasm_bindgen_test]
pub fn drawing_with_graph() {
    test_drawing_with_graph();
}
