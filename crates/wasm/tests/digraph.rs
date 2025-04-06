mod util;

#[allow(unused_imports)]
use egraph_wasm::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

#[wasm_bindgen(module = "tests/digraph.js")]
extern "C" {
    #[wasm_bindgen(js_name = "testDiGraphConstructor")]
    fn test_digraph_constructor();
    #[wasm_bindgen(js_name = "testNodeOperations")]
    fn test_node_operations();
    #[wasm_bindgen(js_name = "testEdgeOperations")]
    fn test_edge_operations();
    #[wasm_bindgen(js_name = "testGraphTraversal")]
    fn test_graph_traversal();
    #[wasm_bindgen(js_name = "testInOutNeighbors")]
    fn test_in_out_neighbors();
    #[wasm_bindgen(js_name = "testDiGraphWithDrawing")]
    fn test_digraph_with_drawing();
}

/// Test basic instantiation of DiGraph class
#[wasm_bindgen_test]
pub fn digraph_constructor() {
    test_digraph_constructor();
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

/// Test directed-specific functionality (in/out neighbors)
#[wasm_bindgen_test]
pub fn in_out_neighbors() {
    test_in_out_neighbors();
}

/// Test integration with drawing component
#[wasm_bindgen_test]
pub fn digraph_with_drawing() {
    test_digraph_with_drawing();
}
