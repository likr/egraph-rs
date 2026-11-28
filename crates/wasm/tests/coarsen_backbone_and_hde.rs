mod util;

use egraph_wasm::coarsen_backbone::{
    js_align_clustered_centroids, js_coarse_backbone_init, js_louvain_coarsen,
    js_project_clustered_layout_with_radii,
};
use egraph_wasm::csr_graph::JsCsrGraph;
use egraph_wasm::hde::{js_hde, js_hde_nd};
use egraph_wasm::layout_buffers::JsLayout2d;
use egraph_wasm::rhgh_landmarks::js_compute_pivoted_rhgh_landmarks;
use egraph_wasm::sgd::JsSgdSolver;
use egraph_wasm::sgd_pacmap::JsPacmapGraphSolver;
use util::fixtures::{
    assert_layouts_bit_exact, make_grid, make_opts, make_path_graph, make_two_cluster_graph,
    u32_array_js,
};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn coarsen_rhgh_and_coarse_backbone() {
    // 1. Three-cluster chain A(6) - B(16) - C(6)
    let small = 6usize;
    let big = 16usize;
    let n = 2 * small + big;
    let mut edges = Vec::new();
    for i in 0..small {
        for j in (i + 1)..small {
            edges.push(i as u32);
            edges.push(j as u32);
        }
    }
    for i in 0..big {
        for j in (i + 1)..big {
            edges.push((small + i) as u32);
            edges.push((small + j) as u32);
        }
    }
    let c0 = small + big;
    for i in 0..small {
        for j in (i + 1)..small {
            edges.push((c0 + i) as u32);
            edges.push((c0 + j) as u32);
        }
    }
    edges.push((small - 1) as u32);
    edges.push(small as u32);
    edges.push((small + big - 1) as u32);
    edges.push(c0 as u32);
    let chain = JsCsrGraph::from_edge_list(
        n as f64,
        &edges,
        None,
        &JsValue::from_bool(false),
        Some(true),
    )
    .unwrap();

    let h = js_louvain_coarsen(
        &chain,
        &make_opts(&[
            ("maxCoarseNodes", JsValue::from_f64(3.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    assert!(h.num_clusters() <= 3);
    assert!(h.num_levels() >= 1);
    assert_eq!(h.node_to_coarse().len(), n);

    // 2. Pivoted RHGH landmarks on path(10)
    let path = make_path_graph(10);
    let rhgh = js_compute_pivoted_rhgh_landmarks(
        &path,
        &make_opts(&[
            ("numLandmarks", JsValue::from_f64(10.0)),
            ("minDegree", JsValue::from_f64(1.0)),
            ("numProbes", JsValue::from_f64(16.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    assert!(rhgh.all_converged());
    let d_alt = rhgh.landmarks().alt_distances(&[0, 0], &[1, 9]).unwrap();
    assert!(d_alt[1] > d_alt[0]);

    // 3. Coarse backbone init (works under both default and --no-default-features)
    let bb = js_coarse_backbone_init(
        &chain,
        &make_opts(&[
            ("maxCoarseNodes", JsValue::from_f64(3.0)),
            ("c0", JsValue::from_f64(0.8)),
            ("kappa", JsValue::from_f64(1.0)),
            ("muOv", JsValue::from_f64(2.0)),
            ("mGap", JsValue::from_f64(0.1)),
            (
                "edgeLengthMode",
                JsValue::from_str("resistanceCurvatureFlow"),
            ),
            ("distanceBackend", JsValue::from_str("rhgh")),
            ("coarseTMax", JsValue::from_f64(30.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    assert_eq!(bb.num_fine_nodes(), n);
    assert_eq!(bb.num_coarse_nodes(), 3);
    assert_eq!(bb.coarse_overlap_pairs_final(), 0);
    assert!(bb.fine_layout().is_all_finite());

    // 4. Centroid alignment + radii projection
    let coarse = JsLayout2d::from_coords(&[5.0, -3.0, -5.0, 2.0]).unwrap();
    let mut fine = JsLayout2d::from_coords(&[
        0.0, 0.0, 2.5, 0.25, 5.0, 0.0, 7.5, -0.25, 10.0, 0.0, 0.0, 0.0,
    ])
    .unwrap();
    let n2c = [0u32, 0, 0, 0, 0, 1];
    js_align_clustered_centroids(
        &mut fine,
        &coarse,
        &n2c,
        &make_opts(&[("gammaTether", JsValue::from_f64(1.0))]),
    )
    .unwrap();
    assert!(fine.is_all_finite());
    let proj = js_project_clustered_layout_with_radii(
        &coarse,
        &n2c,
        &[2.0, 4.0],
        &make_opts(&[
            ("jitterScale", JsValue::from_f64(0.5)),
            ("seed", JsValue::from_f64(7.0)),
        ]),
    )
    .unwrap();
    assert_eq!(proj.num_nodes(), 6);

    // 5. SparseBarrierDog with degreeWeightBeta + PacmapGraphSolver.withBackbone
    let grid = make_grid(6, 6);
    let mut sgd_beta = JsSgdSolver::sparse_barrier_dog(
        &grid,
        &make_opts(&[
            ("degreeWeightBeta", JsValue::from_f64(0.5)),
            ("minDegree", JsValue::from_f64(1.0)),
            ("jlDim", JsValue::from_f64(8.0)),
            ("iterations", JsValue::from_f64(5.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    let mut l_beta =
        JsLayout2d::init(36.0, &make_opts(&[("seed", JsValue::from_f64(42.0))]), None).unwrap();
    sgd_beta.run_2d(&mut l_beta).unwrap();
    assert!(l_beta.is_all_finite());

    let phase_arr = js_sys::Array::of3(
        &JsValue::from_f64(10.0),
        &JsValue::from_f64(10.0),
        &JsValue::from_f64(10.0),
    );
    let pm_bb = JsPacmapGraphSolver::with_backbone(
        &grid,
        &make_opts(&[
            ("nNeighbors", JsValue::from_f64(8.0)),
            ("phaseIterations", phase_arr.into()),
            ("numLandmarks", JsValue::from_f64(12.0)),
            ("maxCoarseNodes", JsValue::from_f64(9.0)),
            ("coarseTMax", JsValue::from_f64(20.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    let mut pm_solver = pm_bb.solver();
    let mut pm_layout = pm_bb.fine_layout();
    pm_solver.step_2d(&mut pm_layout).unwrap();
    pm_solver.run_2d(&mut pm_layout).unwrap();
    assert!(pm_layout.is_all_finite());

    // 6. 20-node two-10-clique bridge with jitterScale=0.0, targetStd=1.0
    let mut two_clique_edges = Vec::new();
    for i in 0..10u32 {
        for j in (i + 1)..10u32 {
            two_clique_edges.push(i);
            two_clique_edges.push(j);
            two_clique_edges.push(10 + i);
            two_clique_edges.push(10 + j);
        }
    }
    two_clique_edges.push(9);
    two_clique_edges.push(10);
    let g_two = JsCsrGraph::from_edge_list(
        20.0,
        &two_clique_edges,
        None,
        &JsValue::from_bool(false),
        Some(true),
    )
    .unwrap();
    let bb_two = js_coarse_backbone_init(
        &g_two,
        &make_opts(&[
            ("maxCoarseNodes", JsValue::from_f64(2.0)),
            ("jitterScale", JsValue::from_f64(0.0)),
            ("targetStd", JsValue::from_f64(1.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    assert_eq!(bb_two.num_coarse_nodes(), 2);
    assert!((bb_two.scale_factor_applied() - 1.0).abs() > 1e-3);
    let fl = bb_two.fine_layout().copy_coords();
    let cl = bb_two.coarse_layout().copy_coords();
    let n2c = bb_two.node_to_coarse();
    for cid in 0..2u32 {
        let mut sx = 0.0_f64;
        let mut sy = 0.0_f64;
        let mut cnt = 0usize;
        for (idx, &c) in n2c.iter().enumerate() {
            if c == cid {
                sx += fl[2 * idx] as f64;
                sy += fl[2 * idx + 1] as f64;
                cnt += 1;
            }
        }
        let mx = sx / (cnt as f64);
        let my = sy / (cnt as f64);
        let diff =
            (mx - cl[2 * (cid as usize)] as f64).hypot(my - cl[2 * (cid as usize) + 1] as f64);
        assert!(
            diff < 1e-5,
            "WASM cluster {cid} centroid mismatch: diff={diff}"
        );
    }
}

#[wasm_bindgen_test]
fn hde_and_hde_nd_with_pivots_and_validation() {
    let (graph, _) = make_two_cluster_graph();

    let l2_default = js_hde(&graph, &JsValue::UNDEFINED).unwrap();
    assert_eq!(l2_default.len(), 8);
    assert!(l2_default.is_all_finite());

    let l2_repeat = js_hde(&graph, &JsValue::UNDEFINED).unwrap();
    assert_layouts_bit_exact(&l2_default, &l2_repeat);

    let piv_u32 = js_sys::Uint32Array::from(&[0u32, 3, 4, 7][..]);
    let opts_2d = make_opts(&[
        ("dimensions", JsValue::from_f64(4.0)),
        ("seed", JsValue::from_f64(7.0)),
        ("tol", JsValue::from_f64(1e-4)),
        ("maxSweeps", JsValue::from_f64(80.0)),
        ("normalize", JsValue::from_bool(false)),
        ("pivots", piv_u32.into()),
    ]);
    let l2_piv = js_hde(&graph, &opts_2d).unwrap();
    assert_eq!(l2_piv.len(), 8);
    assert!(l2_piv.is_all_finite());

    let opts_nd = make_opts(&[
        ("dimensions", JsValue::from_f64(6.0)),
        ("outputDims", JsValue::from_f64(3.0)),
        ("seed", JsValue::from_f64(42.0)),
        ("normalize", JsValue::from_bool(true)),
    ]);
    let lnd = js_hde_nd(&graph, &opts_nd).unwrap();
    assert_eq!(lnd.num_nodes(), 8);
    assert_eq!(lnd.dims(), 3);
    assert!(lnd.is_all_finite());

    let opts_nd_piv = make_opts(&[
        ("dimensions", JsValue::from_f64(4.0)),
        ("outputDims", JsValue::from_f64(3.0)),
        ("pivots", u32_array_js(&[0, 2, 5, 7])),
    ]);
    let lnd_piv = js_hde_nd(&graph, &opts_nd_piv).unwrap();
    assert_eq!(lnd_piv.num_nodes(), 8);
    assert_eq!(lnd_piv.dims(), 3);
    assert!(lnd_piv.is_all_finite());

    // N=0 and N=1 non-panic checks
    let g0 = JsCsrGraph::from_edge_list(0.0, &[], None, &JsValue::UNDEFINED, None).unwrap();
    assert_eq!(js_hde(&g0, &JsValue::UNDEFINED).unwrap().len(), 0);
    assert_eq!(js_hde_nd(&g0, &opts_nd).unwrap().num_nodes(), 0);

    let g1 = JsCsrGraph::from_edge_list(1.0, &[], None, &JsValue::UNDEFINED, None).unwrap();
    assert_eq!(js_hde(&g1, &JsValue::UNDEFINED).unwrap().len(), 1);
    assert_eq!(js_hde_nd(&g1, &opts_nd).unwrap().num_nodes(), 1);

    // Invalid option errors
    let bad_dim = make_opts(&[("dimensions", JsValue::from_f64(0.0))]);
    assert!(js_hde(&graph, &bad_dim).is_err());

    let bad_piv = make_opts(&[("pivots", u32_array_js(&[0, 99]))]);
    assert!(js_hde(&graph, &bad_piv).is_err());

    let bad_out = make_opts(&[("outputDims", JsValue::from_f64(0.0))]);
    assert!(js_hde_nd(&graph, &bad_out).is_err());
}
