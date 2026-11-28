#![allow(dead_code)]

use egraph_wasm::csr_graph::JsCsrGraph;
use egraph_wasm::layout_buffers::JsLayout2d;
use wasm_bindgen::JsValue;

pub fn make_grid(rows: u32, cols: u32) -> JsCsrGraph {
    let mut edges = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let u = r * cols + c;
            if c + 1 < cols {
                edges.push(u);
                edges.push(u + 1);
            }
            if r + 1 < rows {
                edges.push(u);
                edges.push(u + cols);
            }
        }
    }
    JsCsrGraph::from_edge_list(
        (rows * cols) as f64,
        &edges,
        None,
        &JsValue::UNDEFINED,
        None,
    )
    .expect("grid graph")
}

pub fn make_grid_4x4_with_layout() -> (JsCsrGraph, JsLayout2d) {
    let mut edges = Vec::new();
    let mut coords = Vec::with_capacity(32);
    for r in 0..4u32 {
        for c in 0..4u32 {
            let id = r * 4 + c;
            coords.push(c as f32);
            coords.push(r as f32);
            if c + 1 < 4 {
                edges.push(id);
                edges.push(id + 1);
            }
            if r + 1 < 4 {
                edges.push(id);
                edges.push(id + 4);
            }
        }
    }
    let graph = JsCsrGraph::from_edge_list(16.0, &edges, None, &JsValue::from_bool(false), None)
        .expect("grid CSR");
    let layout = JsLayout2d::from_coords(&coords).expect("grid layout");
    (graph, layout)
}

pub fn make_path_graph(n: usize) -> JsCsrGraph {
    let mut edges = Vec::with_capacity((n - 1) * 2);
    for u in 0..(n as u32 - 1) {
        edges.push(u);
        edges.push(u + 1);
    }
    JsCsrGraph::from_edge_list(
        n as f64,
        &edges,
        None,
        &JsValue::from_bool(false),
        Some(true),
    )
    .expect("path graph")
}

pub fn make_barbell(clique_size: u32) -> JsCsrGraph {
    let mut edges = Vec::new();
    for off in [0u32, clique_size] {
        for i in 0..clique_size {
            for j in (i + 1)..clique_size {
                edges.push(off + i);
                edges.push(off + j);
            }
        }
    }
    edges.push(clique_size - 1);
    edges.push(clique_size);
    JsCsrGraph::from_edge_list(
        (clique_size * 2) as f64,
        &edges,
        None,
        &JsValue::UNDEFINED,
        None,
    )
    .expect("barbell graph")
}

pub fn make_two_cluster_graph() -> (JsCsrGraph, Vec<u32>) {
    // Two 4-cliques (0..4 and 4..8) connected by a single bridge (3, 4).
    let mut edges = Vec::new();
    for off in [0u32, 4u32] {
        for i in 0..4 {
            for j in (i + 1)..4 {
                edges.push(off + i);
                edges.push(off + j);
            }
        }
    }
    edges.push(3);
    edges.push(4);
    let graph = JsCsrGraph::from_edge_list(8.0, &edges, None, &JsValue::UNDEFINED, None)
        .expect("two-cluster graph");
    let labels = vec![0, 0, 0, 0, 1, 1, 1, 1];
    (graph, labels)
}

pub fn make_star_graph(leaves: u32) -> JsCsrGraph {
    let mut edges = Vec::new();
    for i in 1..=leaves {
        edges.push(0);
        edges.push(i);
    }
    JsCsrGraph::from_edge_list((leaves + 1) as f64, &edges, None, &JsValue::UNDEFINED, None)
        .expect("star graph")
}

pub fn make_disconnected_graph() -> JsCsrGraph {
    // Two disconnected triangles plus an isolated node 6 (N=7).
    let edges = [0, 1, 1, 2, 0, 2, 3, 4, 4, 5, 3, 5];
    JsCsrGraph::from_edge_list(7.0, &edges, None, &JsValue::UNDEFINED, None)
        .expect("disconnected graph")
}

pub fn make_hub_chain_graph() -> JsCsrGraph {
    // Hub 0 connected to 1..=8; chain 8-9-10-11; isolated node 12 (N=13).
    let mut edges = Vec::new();
    for i in 1..=8u32 {
        edges.push(0);
        edges.push(i);
    }
    edges.extend([8, 9, 9, 10, 10, 11]);
    JsCsrGraph::from_edge_list(13.0, &edges, None, &JsValue::UNDEFINED, None)
        .expect("hub chain graph")
}

pub fn make_circle_layout(n: usize, radius: f32) -> JsLayout2d {
    let mut coords = Vec::with_capacity(n * 2);
    for i in 0..n {
        let ang = std::f32::consts::TAU * (i as f32) / (n.max(1) as f32);
        let (s, c) = ang.sin_cos();
        coords.push(radius * c);
        coords.push(radius * s);
    }
    JsLayout2d::from_coords(&coords).expect("circle layout")
}

pub fn make_separated_cluster_layout() -> JsLayout2d {
    JsLayout2d::from_coords(&[
        -5.0, 0.0, -4.5, 0.5, -5.5, 0.5, -5.0, 1.0, 5.0, 0.0, 5.5, 0.5, 4.5, 0.5, 5.0, 1.0,
    ])
    .expect("separated layout")
}

pub fn make_overlapping_layout() -> JsLayout2d {
    JsLayout2d::from_coords(&[
        0.0, 0.0, 0.1, 0.05, -0.08, 0.07, 0.05, -0.06, 0.12, 0.1, -0.1, -0.08, 0.02, 0.09, -0.04,
        0.02,
    ])
    .expect("overlapping layout")
}

pub fn make_imbalanced_cluster_fixture() -> (JsCsrGraph, JsLayout2d, Vec<f32>, Vec<u32>, JsValue) {
    let mut imb_coords = Vec::with_capacity(48);
    let mut imb_labels = Vec::with_capacity(24);
    for i in 0..24u32 {
        let x = if i < 20 {
            -2.0 + 0.5 * (i as f32 / 19.0)
        } else {
            1.5 + 0.5 * ((i - 20) as f32 / 3.0)
        };
        let y = (i as f32 * 0.7).sin() * 0.2;
        imb_coords.push(x);
        imb_coords.push(y);
        imb_labels.push(if i < 20 { 0 } else { 1 });
    }
    let imb_layout = JsLayout2d::from_coords(&imb_coords).unwrap();
    let imb_labels_js = u32_typed_js(&imb_labels);
    let path_edges: Vec<u32> = (0..23u32).flat_map(|i| [i, i + 1]).collect();
    let imb_graph =
        JsCsrGraph::from_edge_list(24.0, &path_edges, None, &JsValue::from_bool(false), None)
            .unwrap();
    (imb_graph, imb_layout, imb_coords, imb_labels, imb_labels_js)
}

pub fn make_opts(entries: &[(&str, JsValue)]) -> js_sys::Object {
    let obj = js_sys::Object::new();
    for (k, v) in entries {
        js_sys::Reflect::set(&obj, &JsValue::from_str(k), v).unwrap();
    }
    obj
}

pub fn u32_array_js(vals: &[u32]) -> JsValue {
    let arr = js_sys::Array::new();
    for &v in vals {
        arr.push(&JsValue::from_f64(v as f64));
    }
    arr.into()
}

pub fn u32_typed_js(vals: &[u32]) -> JsValue {
    js_sys::Uint32Array::from(vals).into()
}

pub fn f64_js(vals: &[f64]) -> JsValue {
    let arr = js_sys::Array::new();
    for &v in vals {
        arr.push(&JsValue::from_f64(v));
    }
    arr.into()
}

pub fn assert_layouts_bit_exact(a: &JsLayout2d, b: &JsLayout2d) {
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        let idx = i as f64;
        assert_eq!(a.x(idx).unwrap().to_bits(), b.x(idx).unwrap().to_bits());
        assert_eq!(a.y(idx).unwrap().to_bits(), b.y(idx).unwrap().to_bits());
    }
}

pub fn count_rect_overlaps(layout: &JsLayout2d, hw: &[f32], hh: &[f32]) -> usize {
    let n = layout.len();
    let mut count = 0usize;
    for i in 0..n {
        let xi = layout.x(i as f64).unwrap();
        let yi = layout.y(i as f64).unwrap();
        for j in (i + 1)..n {
            let xj = layout.x(j as f64).unwrap();
            let yj = layout.y(j as f64).unwrap();
            if (xi - xj).abs() < hw[i] + hw[j] && (yi - yj).abs() < hh[i] + hh[j] {
                count += 1;
            }
        }
    }
    count
}

pub fn layout_global_std(layout: &JsLayout2d) -> f64 {
    let n = layout.len();
    assert!(n >= 2);
    let mut sx = 0.0_f64;
    let mut sy = 0.0_f64;
    for i in 0..n {
        sx += layout.x(i as f64).unwrap() as f64;
        sy += layout.y(i as f64).unwrap() as f64;
    }
    let mx = sx / (n as f64);
    let my = sy / (n as f64);
    let mut sum_sq = 0.0_f64;
    for i in 0..n {
        let dx = (layout.x(i as f64).unwrap() as f64) - mx;
        let dy = (layout.y(i as f64).unwrap() as f64) - my;
        sum_sq += dx * dx + dy * dy;
    }
    (sum_sq / (2.0 * (n as f64))).sqrt()
}
