use egraph_wasm::csr_graph::JsCsrGraph;
use egraph_wasm::layout_buffers::{JsLayout2d, JsLayoutNd, wasm_memory};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn csr_graph_from_csr_and_sanitization() {
    let indptr = [0u32, 1, 3, 4];
    let indices = [1u32, 0, 2, 1];
    let weights = [1.5f32, 1.5, 2.5, 2.5];

    let g = JsCsrGraph::from_csr(
        &indptr,
        &indices,
        Some(weights.to_vec()),
        &JsValue::from_bool(false),
        Some(false),
    )
    .expect("valid undirected weighted CSR");
    assert_eq!(g.num_nodes(), 3);
    assert_eq!(g.num_edges(), 2);
    assert!(!g.is_directed());
    assert_eq!(g.indptr(), vec![0, 1, 3, 4]);
    assert_eq!(g.indices(), vec![1, 0, 2, 1]);
    assert_eq!(g.weights(), Some(vec![1.5, 1.5, 2.5, 2.5]));

    // Unsorted + self-loop + duplicate + missing reverse edge should sanitize cleanly.
    let dirty_indptr = [0u32, 4, 5, 5];
    let dirty_indices = [0u32, 2, 1, 2, 0];
    let dirty_weights = [9.0f32, 2.0, 1.5, 4.0, 1.5];

    assert!(
        JsCsrGraph::from_csr(
            &dirty_indptr,
            &dirty_indices,
            Some(dirty_weights.to_vec()),
            &JsValue::from_bool(false),
            Some(false),
        )
        .is_err()
    );

    let sanitized = JsCsrGraph::from_csr(
        &dirty_indptr,
        &dirty_indices,
        Some(dirty_weights.to_vec()),
        &JsValue::UNDEFINED,
        None,
    )
    .expect("default options sanitize dirty CSR");
    assert_eq!(sanitized.num_nodes(), 3);
    assert_eq!(sanitized.num_edges(), 2);
    assert_eq!(sanitized.indptr(), vec![0, 2, 3, 4]);
    assert_eq!(sanitized.indices(), vec![1, 2, 0, 0]);
    assert_eq!(sanitized.weights(), Some(vec![1.5, 4.0, 1.5, 4.0]));
}

#[wasm_bindgen_test]
fn csr_graph_from_edge_list_and_options_object() {
    let opts = js_sys::Object::new();
    js_sys::Reflect::set(&opts, &"directed".into(), &JsValue::from_bool(true)).unwrap();
    js_sys::Reflect::set(&opts, &"sanitize".into(), &JsValue::from_bool(false)).unwrap();

    let dg = JsCsrGraph::from_edge_list(3.0, &[0, 1, 0, 2], None, &opts, None)
        .expect("directed unweighted edge list");
    assert!(dg.is_directed());
    assert_eq!(dg.num_nodes(), 3);
    assert_eq!(dg.num_edges(), 2);
    assert_eq!(dg.indptr(), vec![0, 2, 2, 2]);
    assert_eq!(dg.indices(), vec![1, 2]);
    assert!(dg.weights().is_none());
}

#[wasm_bindgen_test]
fn csr_graph_adversarial_errors_return_js_error() {
    // Empty indptr
    assert!(JsCsrGraph::from_csr(&[], &[], None, &JsValue::UNDEFINED, None).is_err());
    // Out-of-bounds neighbor index
    assert!(JsCsrGraph::from_csr(&[0, 1, 1], &[5], None, &JsValue::UNDEFINED, None).is_err());
    // Non-finite weight
    assert!(
        JsCsrGraph::from_csr(
            &[0, 1, 2],
            &[1, 0],
            Some(vec![f32::NAN, 1.0]),
            &JsValue::UNDEFINED,
            None,
        )
        .is_err()
    );
    // Odd edge list length
    assert!(JsCsrGraph::from_edge_list(3.0, &[0, 1, 2], None, &JsValue::UNDEFINED, None).is_err());
    // Invalid numNodes (negative, fractional, NaN)
    assert!(JsCsrGraph::from_edge_list(-1.0, &[], None, &JsValue::UNDEFINED, None).is_err());
    assert!(JsCsrGraph::from_edge_list(1.5, &[], None, &JsValue::UNDEFINED, None).is_err());
    assert!(JsCsrGraph::from_edge_list(f64::NAN, &[], None, &JsValue::UNDEFINED, None).is_err());
    // Invalid options type
    assert!(JsCsrGraph::from_csr(&[0, 0], &[], None, &JsValue::from_str("bad"), None).is_err());
}

#[wasm_bindgen_test]
fn layout2d_lifecycle_linear_memory_and_errors() {
    let mut layout = JsLayout2d::zeros(3.0).expect("zeros(3)");
    assert_eq!(layout.len(), 3);
    assert!(!layout.is_empty());
    assert_eq!(layout.coords_len(), 6);
    assert!(layout.is_all_finite());

    layout.set_x(1.0, 4.5).unwrap();
    layout.set_y(1.0, -2.25).unwrap();
    assert_eq!(layout.x(1.0).unwrap(), 4.5);
    assert_eq!(layout.y(1.0).unwrap(), -2.25);

    // Inspect and mutate the underlying WASM linear memory slice directly.
    let mem = wasm_memory();
    let view = js_sys::Float32Array::new_with_byte_offset_and_length(
        &mem.buffer(),
        layout.coords_ptr() as u32,
        layout.coords_len() as u32,
    );
    assert_eq!(view.get_index(2), 4.5);
    assert_eq!(view.get_index(3), -2.25);
    view.set_index(4, 9.25);
    view.set_index(5, -7.5);
    assert_eq!(layout.x(2.0).unwrap(), 9.25);
    assert_eq!(layout.y(2.0).unwrap(), -7.5);

    // Non-finite check
    layout.set_x(0.0, f32::NAN).unwrap();
    assert!(!layout.is_all_finite());

    // Round-trip fromCoords
    let from_coords = JsLayout2d::from_coords(&[1.0, 2.0, 3.0, 4.0]).expect("even coords");
    assert_eq!(from_coords.len(), 2);
    assert_eq!(from_coords.x(1.0).unwrap(), 3.0);
    assert_eq!(from_coords.y(1.0).unwrap(), 4.0);

    // Adversarial inputs
    assert!(JsLayout2d::from_coords(&[1.0, 2.0, 3.0]).is_err());
    assert!(JsLayout2d::zeros(-1.0).is_err());
    assert!(JsLayout2d::zeros(2.5).is_err());
    assert!(layout.x(3.0).is_err());
    assert!(layout.y(-1.0).is_err());
    assert!(layout.set_x(1.5, 0.0).is_err());
    assert!(layout.set_y(99.0, 0.0).is_err());
}

#[wasm_bindgen_test]
fn csr_graph_degrees_subgraph_and_lcc() {
    // Disconnected graph: component {0,1} and larger component {2,3,4,5}
    let edges = [0u32, 1, 2, 3, 3, 4, 4, 5, 5, 2];
    let weights = [1.5f32, 2.0, 2.0, 2.0, 2.0];
    let g = JsCsrGraph::from_edge_list(
        6.0,
        &edges,
        Some(weights.to_vec()),
        &JsValue::from_bool(false),
        Some(true),
    )
    .unwrap();

    assert_eq!(g.nnz(), 10);
    assert!(g.is_weighted());
    assert_eq!(g.degree(2.0).unwrap(), 2);
    assert_eq!(g.neighbors(2.0).unwrap(), vec![3, 5]);
    assert!(g.degree(99.0).is_err());

    let sub = g.induced_subgraph(&[2, 3, 4]).unwrap();
    assert_eq!(sub.num_nodes(), 3);
    assert_eq!(sub.num_edges(), 2);

    let lcc = g.largest_connected_component();
    assert_eq!(lcc.graph().num_nodes(), 4);
    assert_eq!(lcc.original_node_ids(), vec![2, 3, 4, 5]);
    assert_eq!(lcc.node_ids(), vec![2, 3, 4, 5]);
}

#[wasm_bindgen_test]
fn layout_2d_and_nd_lifecycle() {
    let mut l2 = JsLayout2d::init(8.0, Some(123.0), None).unwrap();
    assert_eq!(l2.num_nodes(), 8);
    assert!(!l2.is_empty());
    assert!(l2.is_all_finite());
    assert_eq!(l2.copy_coords().len(), 16);

    l2.normalize_std(Some(2.0)).unwrap();
    let coords = l2.copy_coords();
    let mut mean_x = 0.0f64;
    let mut mean_y = 0.0f64;
    for i in 0..8 {
        mean_x += coords[2 * i] as f64 / 8.0;
        mean_y += coords[2 * i + 1] as f64 / 8.0;
    }
    assert!(mean_x.abs() < 1e-4 && mean_y.abs() < 1e-4);

    let nd = JsLayoutNd::zeros(4.0, 3.0).unwrap();
    assert_eq!(nd.num_nodes(), 4);
    assert_eq!(nd.dims(), 3);
    assert_eq!(nd.len(), 4);
    assert_eq!(nd.coords_len(), 12);
    assert!(nd.coords_ptr() > 0);
    assert!(!nd.is_empty());
    assert!(nd.is_all_finite());

    let nd2 = JsLayoutNd::from_coords(2.0, 3.0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert_eq!(nd2.copy_coords(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert!(JsLayoutNd::from_coords(2.0, 3.0, &[1.0]).is_err());
}
