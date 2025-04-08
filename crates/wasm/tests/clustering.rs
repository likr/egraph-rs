mod util;

#[allow(unused_imports)]
use egraph_wasm::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

#[wasm_bindgen(module = "tests/clustering.js")]
extern "C" {
    #[wasm_bindgen(js_name = "testBasicCoarsening")]
    fn test_basic_coarsening();
    #[wasm_bindgen(js_name = "testComplexGraphCoarsening")]
    fn test_complex_graph_coarsening();
    #[wasm_bindgen(js_name = "testCustomNodeAndEdgeMerging")]
    fn test_custom_node_and_edge_merging();
    #[wasm_bindgen(js_name = "testClusteringIntegration")]
    fn test_clustering_integration();
}

/// Test basic graph coarsening functionality
#[wasm_bindgen_test]
pub fn basic_coarsening() {
    test_basic_coarsening();
}

/// Test coarsening with a more complex graph
#[wasm_bindgen_test]
pub fn complex_graph_coarsening() {
    test_complex_graph_coarsening();
}

/// Test custom node and edge merging functions
#[wasm_bindgen_test]
pub fn custom_node_and_edge_merging() {
    test_custom_node_and_edge_merging();
}

/// Test integration of clustering with other components
#[wasm_bindgen_test]
pub fn clustering_integration() {
    test_clustering_integration();
}
