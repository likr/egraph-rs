mod util;

#[allow(unused_imports)]
use egraph_wasm::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

#[wasm_bindgen(module = "tests/drawing_hyperbolic_2d.js")]
extern "C" {
    #[wasm_bindgen(js_name = "testDrawingHyperbolic2dConstructor")]
    fn test_drawing_hyperbolic_2d_constructor();
    #[wasm_bindgen(js_name = "testNodeCoordinates")]
    fn test_node_coordinates();
    #[wasm_bindgen(js_name = "testDrawingWithGraph")]
    fn test_drawing_with_graph();
    #[wasm_bindgen(js_name = "testHyperbolicDistance")]
    fn test_hyperbolic_distance();
    #[wasm_bindgen(js_name = "testPoincareDiscConstraint")]
    fn test_poincare_disc_constraint();
    #[wasm_bindgen(js_name = "testCoordinateValidation")]
    fn test_coordinate_validation();
    #[wasm_bindgen(js_name = "testLayoutIntegration")]
    fn test_layout_integration();
}

/// Test basic instantiation of DrawingHyperbolic2d class
#[wasm_bindgen_test]
pub fn drawing_hyperbolic_2d_constructor() {
    test_drawing_hyperbolic_2d_constructor();
}

/// Test node coordinate operations (get/set x,y)
#[wasm_bindgen_test]
pub fn node_coordinates() {
    test_node_coordinates();
}

/// Test integration with Graph class
#[wasm_bindgen_test]
pub fn drawing_with_graph() {
    test_drawing_with_graph();
}

/// Test hyperbolic distance calculations between nodes
#[wasm_bindgen_test]
pub fn hyperbolic_distance() {
    test_hyperbolic_distance();
}

/// Test Poincaré disc model constraint (|x^2 + y^2| < 1)
#[wasm_bindgen_test]
pub fn poincare_disc_constraint() {
    test_poincare_disc_constraint();
}

/// Test coordinate validation and normalization
#[wasm_bindgen_test]
pub fn coordinate_validation() {
    test_coordinate_validation();
}

/// Test integration with layout algorithms
#[wasm_bindgen_test]
pub fn layout_integration() {
    test_layout_integration();
}
