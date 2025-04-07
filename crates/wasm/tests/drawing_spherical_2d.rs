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
    #[wasm_bindgen(js_name = "testEdgeSegments")]
    fn test_edge_segments();
    #[wasm_bindgen(js_name = "testGreatCircleDistance")]
    fn test_great_circle_distance();
    #[wasm_bindgen(js_name = "testCoordinateValidation")]
    fn test_coordinate_validation();
    #[wasm_bindgen(js_name = "testLayoutIntegration")]
    fn test_layout_integration();
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

/// Test edge segment representation on a spherical surface
#[wasm_bindgen_test]
pub fn edge_segments() {
    test_edge_segments();
}

/// Test great circle distance calculations between nodes
#[wasm_bindgen_test]
pub fn great_circle_distance() {
    test_great_circle_distance();
}

/// Test coordinate validation (longitude normalization, latitude clamping)
#[wasm_bindgen_test]
pub fn coordinate_validation() {
    test_coordinate_validation();
}

/// Test integration with layout algorithms
#[wasm_bindgen_test]
pub fn layout_integration() {
    test_layout_integration();
}
