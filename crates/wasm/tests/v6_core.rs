use egraph_wasm::v6::{JsCsrGraph, JsLayout2d, wasm_memory};
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
