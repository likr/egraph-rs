use crate::util;

use egraph_wasm::csr_graph::JsCsrGraph;
use egraph_wasm::layout_buffers::JsLayout2d;
use egraph_wasm::spatial_refiner::JsSpatialIntruderRefiner;
use util::fixtures::{assert_layouts_bit_exact, make_grid, make_opts};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn spatial_intruder_refiner_k_cap_separation_and_errors() {
    // Clean grid converges early untouched; intruder pushed apart + step == run
    let small_grid = make_grid(3, 3);
    let mut clean_coords = Vec::new();
    for r in 0..3 {
        for c in 0..3 {
            clean_coords.push(c as f32);
            clean_coords.push(r as f32);
        }
    }
    let clean_orig = JsLayout2d::from_coords(&clean_coords).unwrap();
    let mut clean_layout = clean_orig.clone();
    let mut ref_clean = JsSpatialIntruderRefiner::new(
        &small_grid,
        &make_opts(&[("iterations", JsValue::from_f64(4.0))]),
    )
    .unwrap();
    assert_eq!(ref_clean.run(&mut clean_layout).unwrap(), "converged");
    assert_eq!(ref_clean.stats().rounds_executed(), 0);
    assert!(ref_clean.stats().round_stats()[0].skipped());
    assert_layouts_bit_exact(&clean_orig, &clean_layout);

    let mut intr_coords = clean_coords;
    intr_coords[16] = 0.08;
    intr_coords[17] = 0.08;
    let mut intr_run = JsLayout2d::from_coords(&intr_coords).unwrap();
    let mut intr_step = intr_run.clone();
    let dist_before = (intr_run.x(0.0).unwrap() - intr_run.x(8.0).unwrap())
        .hypot(intr_run.y(0.0).unwrap() - intr_run.y(8.0).unwrap());

    let r_opts = make_opts(&[
        ("iterations", JsValue::from_f64(3.0)),
        ("wFp", JsValue::from_f64(0.2)),
        ("hingeFp", JsValue::from_bool(true)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut r_run = JsSpatialIntruderRefiner::new(&small_grid, &r_opts).unwrap();
    let mut r_step = JsSpatialIntruderRefiner::new(&small_grid, &r_opts).unwrap();
    assert!(r_run.run(&mut intr_run).is_ok());
    r_step.step(&mut intr_step).unwrap();
    r_step.run(&mut intr_step).unwrap();
    assert_layouts_bit_exact(&intr_run, &intr_step);
    let r_stats = r_run.stats();
    assert!(r_stats.rounds_executed() >= 1);
    assert!(r_stats.n_fp_total() > 0);
    assert_eq!(r_stats.n_sep_total(), 0);
    assert!(!r_stats.round_stats().is_empty());
    let dist_after = (intr_run.x(0.0).unwrap() - intr_run.x(8.0).unwrap())
        .hypot(intr_run.y(0.0).unwrap() - intr_run.y(8.0).unwrap());
    assert!(dist_after > dist_before * 2.0);

    // SpatialIntruderRefiner dynamic kCap + wAnchor2 + setKCap
    let dyn_opts = make_opts(&[
        ("iterations", JsValue::from_f64(4.0)),
        ("kCap", JsValue::from_f64(8.0)),
        ("wAnchor2", JsValue::from_f64(0.5)),
        ("kCapDecay", JsValue::from_f64(0.5)),
        ("kCapMin", JsValue::from_f64(2.0)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut r_dyn = JsSpatialIntruderRefiner::new(&small_grid, &dyn_opts).unwrap();
    assert_eq!(r_dyn.current_k_cap(), 8);
    let mut l_dyn = JsLayout2d::from_coords(&intr_coords).unwrap();
    r_dyn.step(&mut l_dyn).unwrap();
    assert_eq!(r_dyn.current_k_cap(), 4);
    r_dyn.set_k_cap(6.0).unwrap();
    assert_eq!(r_dyn.current_k_cap(), 6);
    r_dyn.run(&mut l_dyn).unwrap();
    assert!(l_dyn.is_all_finite());
    assert!(r_dyn.set_k_cap(0.0).is_err());

    // SpatialIntruderRefiner opt-in separation hinge (minSepRatio + wSep) cures closed-twin collapse
    let mut twin_edges: Vec<u32> = Vec::new();
    for r in 0..3u32 {
        for c in 0..3u32 {
            let u = r * 3 + c;
            if c + 1 < 3 {
                twin_edges.extend([u, u + 1]);
            }
            if r + 1 < 3 {
                twin_edges.extend([u, u + 3]);
            }
        }
    }
    twin_edges.extend([0, 9, 0, 10, 9, 10]);
    let twin_graph =
        JsCsrGraph::from_edge_list(11.0, &twin_edges, None, &JsValue::from_bool(false), None)
            .unwrap();
    let mut twin_coords = Vec::with_capacity(22);
    for r in 0..3 {
        for c in 0..3 {
            twin_coords.push(c as f32);
            twin_coords.push(r as f32);
        }
    }
    twin_coords.extend([0.0_f32, 0.0, 0.0, 0.0]);
    let mut twin_layout = JsLayout2d::from_coords(&twin_coords).unwrap();
    let sep_opts = make_opts(&[
        ("minSepRatio", JsValue::from_f64(0.15)),
        ("wSep", JsValue::from_f64(1.0)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut r_sep = JsSpatialIntruderRefiner::new(&twin_graph, &sep_opts).unwrap();
    r_sep.run(&mut twin_layout).unwrap();
    let sep_stats = r_sep.stats();
    assert!(sep_stats.n_sep_total() > 0);
    assert!(sep_stats.round_stats()[0].n_sep() > 0);
    let d9_10 = (twin_layout.x(9.0).unwrap() - twin_layout.x(10.0).unwrap())
        .hypot(twin_layout.y(9.0).unwrap() - twin_layout.y(10.0).unwrap());
    assert!(d9_10 > 0.05);

    // Adversarial error check
    assert!(
        JsSpatialIntruderRefiner::new(
            &small_grid,
            &make_opts(&[
                ("iterations", JsValue::from_f64(2.0)),
                ("kCap", JsValue::from_f64(0.0)),
            ]),
        )
        .is_err()
    );
}
