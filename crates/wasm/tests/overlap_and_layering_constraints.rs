mod util;

use egraph_wasm::constraints_layering::{
    JsSeparationConstraint, js_assign_layers, js_cycle_edges, js_generate_layered_constraints,
    js_project_1d, js_project_clustered_rectangle_no_overlap_2d, js_project_layout_axis,
    js_project_rectangle_no_overlap_2d, js_remove_cycles,
};
use egraph_wasm::csr_graph::JsCsrGraph;
use egraph_wasm::layout_buffers::JsLayout2d;
use egraph_wasm::overlap_prism::JsPrismOverlapRemoval;
use egraph_wasm::overlap_tree_forbid::{JsForbidOverlapRemoval, JsGTreeOverlapRemoval};
use util::fixtures::{
    assert_layouts_bit_exact, count_rect_overlaps, make_grid, make_opts, make_overlapping_layout,
};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn prism_overlap_removal_zero_overlaps_and_step_eq_run() {
    let hw = vec![0.6f32; 9];
    let hh = vec![0.4f32; 9];
    let mut init_coords = Vec::new();
    for r in 0..3 {
        for c in 0..3 {
            init_coords.push((c as f32) * 0.3);
            init_coords.push((r as f32) * 0.25);
        }
    }
    let init = JsLayout2d::from_coords(&init_coords).unwrap();
    assert!(count_rect_overlaps(&init, &hw, &hh) > 0);

    let opts = make_opts(&[
        ("maxIterations", JsValue::from_f64(50.0)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut p_run = JsPrismOverlapRemoval::new(&hw, &hh, &opts).unwrap();
    let mut p_step = JsPrismOverlapRemoval::new(&hw, &hh, &opts).unwrap();

    let mut l_run = init.clone();
    let mut l_step = init;
    assert_eq!(p_run.run(&mut l_run).unwrap(), "converged");
    assert_eq!(p_run.last_overlap_count(), 0);
    assert_eq!(count_rect_overlaps(&l_run, &hw, &hh), 0);

    p_step.step(&mut l_step).unwrap();
    while !p_step.is_finished() {
        p_step.step(&mut l_step).unwrap();
    }
    assert_eq!(p_step.last_overlap_count(), 0);
    assert_layouts_bit_exact(&l_run, &l_step);

    // fromRadii on already separated boxes converges immediately
    let radii = vec![0.25f32; 4];
    let mut sep_layout =
        JsLayout2d::from_coords(&[0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0]).unwrap();
    let mut p_rad = JsPrismOverlapRemoval::from_radii(&radii, &JsValue::UNDEFINED).unwrap();
    assert_eq!(p_rad.run(&mut sep_layout).unwrap(), "converged");
    assert_eq!(p_rad.last_overlap_count(), 0);

    // Adversarial checks
    assert!(JsPrismOverlapRemoval::new(&[0.5, 0.5], &[0.5], &JsValue::UNDEFINED).is_err());
    assert!(JsPrismOverlapRemoval::from_radii(&[-0.1, 0.5], &JsValue::UNDEFINED).is_err());
    assert!(
        JsPrismOverlapRemoval::from_radii(
            &radii,
            &make_opts(&[("maxScaleFactor", JsValue::from_f64(1.0))])
        )
        .is_err()
    );
}

#[wasm_bindgen_test]
fn separation_constraints_and_rectangle_projection() {
    // 1. SeparationConstraint + project1d
    let c0 = JsSeparationConstraint::new(0.0, 1.0, 2.0).unwrap();
    let c1 = JsSeparationConstraint::new(1.0, 2.0, 1.5).unwrap();
    assert_eq!(c0.left(), 0);
    assert_eq!(c0.right(), 1);
    assert_eq!(c0.gap(), 2.0);

    let arr = js_sys::Array::new();
    arr.push(&JsValue::from(c0));
    arr.push(&JsValue::from(c1));

    let mut coords = [3.0f32, 1.0, 0.0];
    js_project_1d(&mut coords, &arr).unwrap();
    assert!(coords[1] - coords[0] >= 2.0 - 1e-5);
    assert!(coords[2] - coords[1] >= 1.5 - 1e-5);

    // 2. projectLayoutAxis + generateLayeredConstraints
    let dag = JsCsrGraph::from_edge_list(
        4.0,
        &[0, 1, 0, 2, 1, 3, 2, 3],
        None,
        &JsValue::from_bool(true),
        None,
    )
    .unwrap();
    let layers = js_assign_layers(&dag, &JsValue::from_str("longestPath")).unwrap();
    let layered_cs = js_generate_layered_constraints(&dag, &layers, Some(3.0)).unwrap();
    let js_cs = js_sys::Array::new();
    for c in layered_cs {
        js_cs.push(&JsValue::from(c));
    }
    let mut layout = JsLayout2d::zeros(4.0).unwrap();
    js_project_layout_axis(&mut layout, &JsValue::from_str("y"), &js_cs).unwrap();
    assert!(layout.y(1.0).unwrap() - layout.y(0.0).unwrap() >= 3.0 - 1e-5);
    assert!(layout.y(3.0).unwrap() - layout.y(1.0).unwrap() >= 3.0 - 1e-5);

    // 3. projectRectangleNoOverlap2d
    let hw = [0.6f32; 4];
    let hh = [0.6f32; 4];
    let mut rect_l = JsLayout2d::from_coords(&[0.0, 0.0, 0.2, 0.1, 0.1, 0.2, -0.1, 0.0]).unwrap();
    js_project_rectangle_no_overlap_2d(&mut rect_l, &hw, &hh).unwrap();
    assert_eq!(count_rect_overlaps(&rect_l, &hw, &hh), 0);

    // 4. projectClusteredRectangleNoOverlap2d rigidly preserves intra-cluster deltas
    let mut cl_l = JsLayout2d::from_coords(&[0.0, 0.0, 0.2, 0.1, 0.1, 0.0, 0.3, 0.1]).unwrap();
    let dx0_before = cl_l.x(1.0).unwrap() - cl_l.x(0.0).unwrap();
    let dy0_before = cl_l.y(1.0).unwrap() - cl_l.y(0.0).unwrap();
    let cluster_ids = [0u32, 0, 1, 1];
    let c_hw = [0.5f32; 4];
    let c_hh = [0.5f32; 4];
    js_project_clustered_rectangle_no_overlap_2d(&mut cl_l, &cluster_ids, &c_hw, &c_hh).unwrap();
    let dx0_after = cl_l.x(1.0).unwrap() - cl_l.x(0.0).unwrap();
    let dy0_after = cl_l.y(1.0).unwrap() - cl_l.y(0.0).unwrap();
    assert!((dx0_after - dx0_before).abs() < 1e-5);
    assert!((dy0_after - dy0_before).abs() < 1e-5);

    // Adversarial checks
    let cyc_cs = js_sys::Array::new();
    cyc_cs.push(&JsValue::from(
        JsSeparationConstraint::new(0.0, 1.0, 1.0).unwrap(),
    ));
    cyc_cs.push(&JsValue::from(
        JsSeparationConstraint::new(1.0, 0.0, 1.0).unwrap(),
    ));
    let mut bad_c = [0.0f32, 0.0];
    assert!(js_project_1d(&mut bad_c, &cyc_cs).is_err());
    assert!(js_project_layout_axis(&mut layout, &JsValue::from_f64(2.0), &js_cs).is_err());
    assert!(JsSeparationConstraint::new(0.0, 1.0, f32::NAN).is_err());
}

#[wasm_bindgen_test]
fn layering_cycle_edges_remove_cycles_and_assign_layers() {
    // Cyclic graph: 0 -> 1 -> 2 -> 0, plus 2 -> 3
    let cyc = JsCsrGraph::from_edge_list(
        4.0,
        &[0, 1, 1, 2, 2, 0, 2, 3],
        None,
        &JsValue::from_bool(true),
        None,
    )
    .unwrap();
    let back = js_cycle_edges(&cyc).unwrap();
    assert_eq!(back, vec![2, 0]);

    let dag = js_remove_cycles(&cyc).unwrap();
    assert!(dag.is_directed());
    assert!(js_cycle_edges(&dag).unwrap().is_empty());

    // Multi-span DAG: 0 -> 1 -> 2 -> 3 and 0 -> 3; plus 4 -> 3 (network simplex pulls 4 to layer 2)
    let span_dag = JsCsrGraph::from_edge_list(
        5.0,
        &[0, 1, 1, 2, 2, 3, 0, 3, 4, 3],
        None,
        &JsValue::from_bool(true),
        None,
    )
    .unwrap();
    let lp = js_assign_layers(&span_dag, &JsValue::from_str("longestPath")).unwrap();
    let ns = js_assign_layers(&span_dag, &JsValue::from_str("networkSimplex")).unwrap();
    let ns_obj = js_assign_layers(
        &span_dag,
        &make_opts(&[("method", JsValue::from_str("networkSimplex"))]),
    )
    .unwrap();
    let lp_default_obj = js_assign_layers(&span_dag, &make_opts(&[])).unwrap();
    assert_eq!(lp.len(), 5);
    assert_eq!(ns.len(), 5);
    assert_eq!(ns_obj, ns);
    assert_eq!(lp_default_obj, lp);
    assert_eq!(*ns.iter().min().unwrap(), 0);
    assert_eq!(lp[4], 0);
    assert_eq!(ns[4], 2);

    // Adversarial: undirected graph rejected, snake_case / bogus method / invalid arg type rejected cleanly
    let undir = make_grid(2, 2);
    assert!(js_cycle_edges(&undir).is_err());
    assert!(js_remove_cycles(&undir).is_err());
    assert!(js_assign_layers(&undir, &JsValue::UNDEFINED).is_err());
    let assert_invalid_input = |err: JsValue| {
        assert_eq!(
            js_sys::Reflect::get(&err, &"code".into())
                .unwrap()
                .as_string(),
            Some("InvalidInput".to_owned())
        );
    };
    assert_invalid_input(js_assign_layers(&span_dag, &JsValue::from_str("badMethod")).unwrap_err());
    assert_invalid_input(
        js_assign_layers(
            &span_dag,
            &make_opts(&[("method", JsValue::from_str("longest_path"))]),
        )
        .unwrap_err(),
    );
    assert_invalid_input(
        js_assign_layers(&span_dag, &make_opts(&[("method", JsValue::from_f64(1.0))])).unwrap_err(),
    );
    assert_invalid_input(js_assign_layers(&span_dag, &JsValue::from_f64(42.0)).unwrap_err());
}

#[wasm_bindgen_test]
fn gtree_and_forbid_overlap_removal_step_eq_run() {
    let init = make_overlapping_layout();
    let radii = vec![0.4_f32; 8];
    let hw = vec![0.4_f32; 8];
    let hh = vec![0.3_f32; 8];

    let gtree_opts = make_opts(&[
        ("maxIterations", JsValue::from_f64(25.0)),
        ("padding", JsValue::from_f64(0.02)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut gt_step = JsGTreeOverlapRemoval::new(&hw, &hh, &gtree_opts).unwrap();
    let mut gt_run = JsGTreeOverlapRemoval::new(&hw, &hh, &gtree_opts).unwrap();
    let mut l_gt_step = init.clone();
    let mut l_gt_run = init.clone();
    while !gt_step.is_finished() {
        gt_step.step(&mut l_gt_step).unwrap();
    }
    gt_run.run(&mut l_gt_run).unwrap();
    assert_layouts_bit_exact(&l_gt_step, &l_gt_run);
    assert_eq!(gt_run.last_overlap_count(), 0);
    assert_eq!(count_rect_overlaps(&l_gt_run, &hw, &hh), 0);

    let mut gt_radii = JsGTreeOverlapRemoval::from_radii(&radii, &gtree_opts).unwrap();
    let mut l_gt_rad = init.clone();
    gt_radii.run(&mut l_gt_rad).unwrap();
    assert_eq!(count_rect_overlaps(&l_gt_rad, &radii, &radii), 0);

    for variant in ["forbid", "forbidPrime", "sparseForbid"] {
        let f_opts = make_opts(&[
            ("variant", JsValue::from_str(variant)),
            ("maxScaleIters", JsValue::from_f64(20.0)),
            ("sgdIterations", JsValue::from_f64(15.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]);
        let mut fb_step = JsForbidOverlapRemoval::from_radii(&radii, &f_opts).unwrap();
        let mut fb_run = JsForbidOverlapRemoval::from_radii(&radii, &f_opts).unwrap();
        let mut l_fb_step = init.clone();
        let mut l_fb_run = init.clone();
        while !fb_step.is_finished() {
            fb_step.step(&mut l_fb_step).unwrap();
        }
        fb_run.run(&mut l_fb_run).unwrap();
        assert_layouts_bit_exact(&l_fb_step, &l_fb_run);
        assert_eq!(fb_run.last_overlap_count(), 0);
        assert_eq!(count_rect_overlaps(&l_fb_run, &radii, &radii), 0);
    }
}
