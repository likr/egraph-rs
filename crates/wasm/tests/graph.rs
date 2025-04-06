mod util;

#[allow(unused_imports)]
use egraph_wasm::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

#[wasm_bindgen(module = "tests/graph.js")]
extern "C" {
    #[wasm_bindgen(js_name = "testGraphConstructor")]
    fn test_graph_constructor();
    #[wasm_bindgen(js_name = "testNodeOperations")]
    fn test_node_operations();
    #[wasm_bindgen(js_name = "testEdgeOperations")]
    fn test_edge_operations();
    #[wasm_bindgen(js_name = "testGraphTraversal")]
    fn test_graph_traversal();
    #[wasm_bindgen(js_name = "testGraphWithDrawing")]
    fn test_graph_with_drawing();
}

/// Test basic instantiation of Graph class
#[wasm_bindgen_test]
pub fn graph_constructor() {
    test_graph_constructor();
}

/// Test node operations (add, remove, get)
#[wasm_bindgen_test]
pub fn node_operations() {
    test_node_operations();
}

/// Test edge operations (add, remove, get)
#[wasm_bindgen_test]
pub fn edge_operations() {
    test_edge_operations();
}

/// Test graph traversal and iteration
#[wasm_bindgen_test]
pub fn graph_traversal() {
    test_graph_traversal();
}

/// Test integration with drawing component
#[wasm_bindgen_test]
pub fn graph_with_drawing() {
    test_graph_with_drawing();
}
