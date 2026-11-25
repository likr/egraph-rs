#![cfg(target_arch = "wasm32")]

use egraph_wasm::v6::{
    JsCsrGraph, JsGemSolver, JsGripSolver, JsLayout2d, JsMaxentStressSolver,
    JsPfsPrimeOverlapRemoval, JsSelLayoutSolver, JsSfcLayoutSolver, JsSgdSolver, js_hde, js_hde_nd,
};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

fn make_two_cluster_graph() -> JsCsrGraph {
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
    JsCsrGraph::from_edge_list(8.0, &edges, None, &JsValue::UNDEFINED, None)
        .expect("two-cluster graph")
}

fn make_star_graph(leaves: u32) -> JsCsrGraph {
    let mut edges = Vec::new();
    for i in 1..=leaves {
        edges.push(0);
        edges.push(i);
    }
    JsCsrGraph::from_edge_list((leaves + 1) as f64, &edges, None, &JsValue::UNDEFINED, None)
        .expect("star graph")
}

fn make_disconnected_graph() -> JsCsrGraph {
    // Two disconnected triangles plus an isolated node 6 (N=7).
    let edges = [0, 1, 1, 2, 0, 2, 3, 4, 4, 5, 3, 5];
    JsCsrGraph::from_edge_list(7.0, &edges, None, &JsValue::UNDEFINED, None)
        .expect("disconnected graph")
}

fn make_circle_layout(n: usize, radius: f32) -> JsLayout2d {
    let mut coords = Vec::with_capacity(n * 2);
    for i in 0..n {
        let ang = std::f32::consts::TAU * (i as f32) / (n.max(1) as f32);
        let (s, c) = ang.sin_cos();
        coords.push(radius * c);
        coords.push(radius * s);
    }
    JsLayout2d::from_coords(&coords).expect("circle layout")
}

fn make_overlapping_layout() -> JsLayout2d {
    JsLayout2d::from_coords(&[
        0.0, 0.0, 0.1, 0.05, -0.08, 0.07, 0.05, -0.06, 0.12, 0.1, -0.1, -0.08, 0.02, 0.09, -0.04,
        0.02,
    ])
    .expect("overlapping layout")
}

fn make_opts(entries: &[(&str, JsValue)]) -> js_sys::Object {
    let obj = js_sys::Object::new();
    for (k, v) in entries {
        js_sys::Reflect::set(&obj, &JsValue::from_str(k), v).unwrap();
    }
    obj
}

fn u32_js(vals: &[u32]) -> JsValue {
    let arr = js_sys::Array::new();
    for &v in vals {
        arr.push(&JsValue::from_f64(v as f64));
    }
    arr.into()
}

fn assert_layouts_bit_exact(a: &JsLayout2d, b: &JsLayout2d) {
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        let idx = i as f64;
        assert_eq!(a.x(idx).unwrap().to_bits(), b.x(idx).unwrap().to_bits());
        assert_eq!(a.y(idx).unwrap().to_bits(), b.y(idx).unwrap().to_bits());
    }
}

fn count_rect_overlaps(layout: &JsLayout2d, hw: &[f32], hh: &[f32]) -> usize {
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

#[wasm_bindgen_test]
fn w2_1a_hde_and_hde_nd_with_pivots_and_validation() {
    let graph = make_two_cluster_graph();

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
    assert_eq!(lnd.dim(), 3);
    assert!(lnd.is_all_finite());

    let opts_nd_piv = make_opts(&[
        ("dimensions", JsValue::from_f64(4.0)),
        ("outputDims", JsValue::from_f64(3.0)),
        ("pivots", u32_js(&[0, 2, 5, 7])),
    ]);
    let lnd_piv = js_hde_nd(&graph, &opts_nd_piv).unwrap();
    assert_eq!(lnd_piv.num_nodes(), 8);
    assert_eq!(lnd_piv.dim(), 3);
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

    let bad_piv = make_opts(&[("pivots", u32_js(&[0, 99]))]);
    assert!(js_hde(&graph, &bad_piv).is_err());

    let bad_out = make_opts(&[("outputDims", JsValue::from_f64(0.0))]);
    assert!(js_hde_nd(&graph, &bad_out).is_err());
}

#[wasm_bindgen_test]
fn w2_1b_ortmann_sparse_stress_step_eq_run() {
    let graph = make_two_cluster_graph();
    let init = make_circle_layout(8, 1.5);

    for strategy in ["kmeans_sp", "maxmin_random_sp", "maxmin_sp"] {
        let opts = make_opts(&[
            ("numPivots", JsValue::from_f64(4.0)),
            ("pivotStrategy", JsValue::from_str(strategy)),
            ("iterations", JsValue::from_f64(12.0)),
            ("eps", JsValue::from_f64(0.1)),
            ("seed", JsValue::from_f64(42.0)),
        ]);
        let mut s_step = JsSgdSolver::ortmann_sparse_stress(&graph, &opts).unwrap();
        let mut s_run = JsSgdSolver::ortmann_sparse_stress(&graph, &opts).unwrap();
        assert_eq!(s_step.num_nodes(), 8);
        assert_eq!(s_step.total_iterations(), 12);
        assert!(s_step.num_springs() > 0);

        let mut l_step = init.clone();
        let mut l_run = init.clone();
        while !s_step.is_finished() {
            s_step.step(&mut l_step).unwrap();
        }
        let status = s_run.run(&mut l_run).unwrap();
        assert!(matches!(status.as_str(), "completed" | "converged"));
        assert_layouts_bit_exact(&l_step, &l_run);
        assert!(l_run.is_all_finite());
    }

    let opts_eta = make_opts(&[
        ("numPivots", JsValue::from_f64(4.0)),
        ("iterations", JsValue::from_f64(6.0)),
        ("etaMax", JsValue::from_f64(0.8)),
        ("etaMin", JsValue::from_f64(0.05)),
        ("seed", JsValue::from_f64(7.0)),
    ]);
    let mut s_steps = JsSgdSolver::ortmann_sparse_stress(&graph, &opts_eta).unwrap();
    let mut l_steps = init;
    s_steps.run_steps(&mut l_steps, 3.0).unwrap();
    assert_eq!(s_steps.iteration(), 3);
    s_steps.run(&mut l_steps).unwrap();
    assert!(s_steps.is_finished());

    let bad_pivots = make_opts(&[("numPivots", JsValue::from_f64(0.0))]);
    assert!(JsSgdSolver::ortmann_sparse_stress(&graph, &bad_pivots).is_err());

    let bad_strat = make_opts(&[("pivotStrategy", JsValue::from_str("unknown"))]);
    assert!(JsSgdSolver::ortmann_sparse_stress(&graph, &bad_strat).is_err());
}

#[wasm_bindgen_test]
fn w2_2_maxent_stress_step_eq_run_and_getters() {
    let graph = make_two_cluster_graph();
    let star = make_star_graph(7);
    let init = make_circle_layout(8, 1.5);

    // Auto-q rule: two-cluster has 0% deg-1 (q=0.0), star has >30% deg-1 (q=0.8)
    let auto_opts = make_opts(&[
        ("stages", JsValue::from_f64(2.0)),
        ("stepsPerStage", JsValue::from_f64(4.0)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let s_auto_clique = JsMaxentStressSolver::new(&graph, &auto_opts).unwrap();
    assert!((s_auto_clique.effective_q() - 0.0).abs() < 1e-12);
    let s_auto_star = JsMaxentStressSolver::new(&star, &auto_opts).unwrap();
    assert!((s_auto_star.effective_q() - 0.8).abs() < 1e-12);

    for use_bh in [true, false] {
        let opts = make_opts(&[
            ("q", JsValue::from_f64(0.4)),
            ("alphaInit", JsValue::from_f64(1.0)),
            ("alphaDecay", JsValue::from_f64(0.3)),
            ("alphaMin", JsValue::from_f64(0.008)),
            ("stages", JsValue::from_f64(2.0)),
            ("stepsPerStage", JsValue::from_f64(5.0)),
            ("tol", JsValue::from_f64(0.0)),
            ("cgTol", JsValue::from_f64(0.1)),
            ("cgMaxIters", JsValue::from_f64(10.0)),
            ("theta", JsValue::from_f64(1.2)),
            ("useBarnesHut", JsValue::from_bool(use_bh)),
            ("epsSq", JsValue::from_f64(1e-6)),
            ("seed", JsValue::from_f64(42.0)),
        ]);
        let mut s_step = JsMaxentStressSolver::new(&graph, &opts).unwrap();
        let mut s_run = JsMaxentStressSolver::new(&graph, &opts).unwrap();
        assert_eq!(s_step.num_nodes(), 8);
        assert_eq!(s_step.total_iterations(), 10);
        assert!((s_step.effective_q() - 0.4).abs() < 1e-12);
        let alpha_0 = s_step.current_alpha();
        assert!((alpha_0 - 1.0).abs() < 1e-12);

        let mut l_step = init.clone();
        let mut l_run = init.clone();
        while !s_step.is_finished() {
            s_step.step(&mut l_step).unwrap();
        }
        let status = s_run.run(&mut l_run).unwrap();
        assert!(matches!(status.as_str(), "completed" | "converged"));
        assert_layouts_bit_exact(&l_step, &l_run);
        assert!(s_run.current_alpha() <= alpha_0);
        assert!(l_run.is_all_finite());
    }

    // Disconnected, N=0, N=1, and invalid option checks
    let disc = make_disconnected_graph();
    let mut s_disc = JsMaxentStressSolver::new(&disc, &auto_opts).unwrap();
    let mut l_disc = make_circle_layout(7, 1.0);
    s_disc.run(&mut l_disc).unwrap();
    assert!(l_disc.is_all_finite());

    let g0 = JsCsrGraph::from_edge_list(0.0, &[], None, &JsValue::UNDEFINED, None).unwrap();
    let mut s0 = JsMaxentStressSolver::new(&g0, &JsValue::UNDEFINED).unwrap();
    let mut l0 = JsLayout2d::zeros(0.0).unwrap();
    s0.run(&mut l0).unwrap();

    let bad_q = make_opts(&[("q", JsValue::from_f64(-2.0))]);
    assert!(JsMaxentStressSolver::new(&graph, &bad_q).is_err());
}

#[wasm_bindgen_test]
fn w2_3_largevis_drgraph_graph_cne_step_eq_run() {
    let graph = make_two_cluster_graph();
    let init = make_circle_layout(8, 1.5);

    // LargeVis
    let lv_opts = make_opts(&[
        ("iterations", JsValue::from_f64(10.0)),
        ("negativeSamples", JsValue::from_f64(3.0)),
        ("gamma", JsValue::from_f64(7.0)),
        ("initialLr", JsValue::from_f64(1.0)),
        ("minLr", JsValue::from_f64(1e-4)),
        ("degreeExponent", JsValue::from_f64(0.75)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut lv_step = JsSgdSolver::largevis(&graph, &lv_opts).unwrap();
    let mut lv_run = JsSgdSolver::largevis(&graph, &lv_opts).unwrap();
    let mut l_lv_step = init.clone();
    let mut l_lv_run = init.clone();
    while !lv_step.is_finished() {
        lv_step.step(&mut l_lv_step).unwrap();
    }
    lv_run.run(&mut l_lv_run).unwrap();
    assert_layouts_bit_exact(&l_lv_step, &l_lv_run);
    assert!(l_lv_run.is_all_finite());

    // DRGraph
    let dr_opts = make_opts(&[
        ("kOrder", JsValue::from_f64(1.0)),
        ("negativeSamples", JsValue::from_f64(3.0)),
        ("b", JsValue::from_f64(2.0)),
        ("gammaCoarse", JsValue::from_f64(0.01)),
        ("gammaFine", JsValue::from_f64(0.1)),
        ("coarsenStopRatio", JsValue::from_f64(0.8)),
        ("iterations", JsValue::from_f64(12.0)),
        ("initialLr", JsValue::from_f64(1.0)),
        ("minLr", JsValue::from_f64(1e-4)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut dr_step = JsSgdSolver::drgraph(&graph, &dr_opts).unwrap();
    let mut dr_run = JsSgdSolver::drgraph(&graph, &dr_opts).unwrap();
    let mut l_dr_step = init.clone();
    let mut l_dr_run = init.clone();
    while !dr_step.is_finished() {
        dr_step.step(&mut l_dr_step).unwrap();
    }
    dr_run.run(&mut l_dr_run).unwrap();
    assert_layouts_bit_exact(&l_dr_step, &l_dr_run);
    assert!(l_dr_run.is_all_finite());

    // Graph-CNE ("infonce" and "neg_tsne")
    for obj in ["infonce", "neg_tsne"] {
        let cne_opts = make_opts(&[
            ("objective", JsValue::from_str(obj)),
            ("negativeSamples", JsValue::from_f64(3.0)),
            ("iterations", JsValue::from_f64(10.0)),
            ("initialLr", JsValue::from_f64(1.0)),
            ("minLr", JsValue::from_f64(1e-4)),
            ("repulsionWeight", JsValue::from_f64(1.0)),
            ("rowNormalize", JsValue::from_bool(true)),
            ("seed", JsValue::from_f64(42.0)),
        ]);
        let mut cne_step = JsSgdSolver::graph_cne(&graph, &cne_opts).unwrap();
        let mut cne_run = JsSgdSolver::graph_cne(&graph, &cne_opts).unwrap();
        let mut l_cne_step = init.clone();
        let mut l_cne_run = init.clone();
        while !cne_step.is_finished() {
            cne_step.step(&mut l_cne_step).unwrap();
        }
        cne_run.run(&mut l_cne_run).unwrap();
        assert_layouts_bit_exact(&l_cne_step, &l_cne_run);
        assert!(l_cne_run.is_all_finite());
    }

    // Invalid option errors
    assert!(
        JsSgdSolver::largevis(&graph, &make_opts(&[("gamma", JsValue::from_f64(-1.0))])).is_err()
    );
    assert!(
        JsSgdSolver::drgraph(&graph, &make_opts(&[("kOrder", JsValue::from_f64(0.0))])).is_err()
    );
    assert!(
        JsSgdSolver::graph_cne(
            &graph,
            &make_opts(&[("objective", JsValue::from_str("bad"))]),
        )
        .is_err()
    );
}

#[wasm_bindgen_test]
fn w2_4_grip_gem_sel_pfs_prime_sfc_step_eq_run() {
    let graph = make_two_cluster_graph();
    let disc = make_disconnected_graph();
    let init = make_circle_layout(8, 1.5);

    // GRIP
    let grip_opts = make_opts(&[
        ("roundsMin", JsValue::from_f64(3.0)),
        ("roundsMax", JsValue::from_f64(6.0)),
        ("localNeighborhoodSize", JsValue::from_f64(10.0)),
        ("repulsionScale", JsValue::from_f64(0.05)),
        ("edgeLength", JsValue::from_f64(1.0)),
        ("tempR", JsValue::from_f64(0.15)),
        ("tempS", JsValue::from_f64(3.0)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut grip_step = JsGripSolver::new(&graph, &grip_opts).unwrap();
    let mut grip_run = JsGripSolver::new(&graph, &grip_opts).unwrap();
    assert!(grip_step.num_levels() >= 1);
    assert!(grip_step.total_iterations() >= 1);
    let mut l_grip_step = init.clone();
    let mut l_grip_run = init.clone();
    while !grip_step.is_finished() {
        grip_step.step(&mut l_grip_step).unwrap();
    }
    grip_run.run(&mut l_grip_run).unwrap();
    assert_layouts_bit_exact(&l_grip_step, &l_grip_run);
    assert!(l_grip_run.is_all_finite());

    let mut grip_disc = JsGripSolver::new(&disc, &grip_opts).unwrap();
    let mut l_grip_disc = make_circle_layout(7, 1.0);
    grip_disc.run(&mut l_grip_disc).unwrap();
    assert!(l_grip_disc.is_all_finite());

    // GEM
    let gem_opts = make_opts(&[
        ("edgeLength", JsValue::from_f64(128.0)),
        ("gravity", JsValue::from_f64(1.0 / 16.0)),
        ("tInit", JsValue::from_f64(128.0)),
        ("tMax", JsValue::from_f64(256.0)),
        ("tMin", JsValue::from_f64(3.0)),
        ("maxRounds", JsValue::from_f64(10.0)),
        ("oscillationAngle", JsValue::from_f64(std::f64::consts::PI)),
        (
            "rotationAngle",
            JsValue::from_f64(std::f64::consts::PI / 3.0),
        ),
        ("oscillationSensitivity", JsValue::from_f64(1.0 / 3.0)),
        ("perturbation", JsValue::from_f64(32.0)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut gem_step = JsGemSolver::new(&graph, &gem_opts).unwrap();
    let mut gem_run = JsGemSolver::new(&graph, &gem_opts).unwrap();
    assert_eq!(gem_step.total_iterations(), 10);
    let mut l_gem_step = init.clone();
    let mut l_gem_run = init.clone();
    while !gem_step.is_finished() {
        gem_step.step(&mut l_gem_step).unwrap();
    }
    gem_run.run(&mut l_gem_run).unwrap();
    assert_layouts_bit_exact(&l_gem_step, &l_gem_run);
    assert!(l_gem_run.is_all_finite());

    // SEL (all 5 mappings + both backends)
    for mapping in ["sigmoid", "linear", "tanh", "custom_logistic", "binary"] {
        for backend in ["fr", "t_fdp"] {
            let sel_opts = make_opts(&[
                ("treeDepth", JsValue::from_f64(4.0)),
                ("pqMaxPops", JsValue::from_f64(8.0)),
                ("mapping", JsValue::from_str(mapping)),
                ("layoutBackend", JsValue::from_str(backend)),
                ("iterations", JsValue::from_f64(8.0)),
                ("seed", JsValue::from_f64(42.0)),
            ]);
            let mut sel_step = JsSelLayoutSolver::new(&graph, &sel_opts).unwrap();
            let mut sel_run = JsSelLayoutSolver::new(&graph, &sel_opts).unwrap();
            assert!(sel_step.se_target().is_finite());
            assert_eq!(sel_step.modulated_weights().len(), 13);
            let mut l_sel_step = init.clone();
            let mut l_sel_run = init.clone();
            while !sel_step.is_finished() {
                sel_step.step(&mut l_sel_step).unwrap();
            }
            sel_run.run(&mut l_sel_run).unwrap();
            assert_layouts_bit_exact(&l_sel_step, &l_sel_run);
            assert!(l_sel_run.is_all_finite());
        }
    }
    assert!(
        JsSelLayoutSolver::new(&graph, &make_opts(&[("mapping", JsValue::from_str("bad"))]))
            .is_err()
    );

    // PFS'
    let overlap_init = make_overlapping_layout();
    let hw = vec![0.4_f32; 8];
    let hh = vec![0.3_f32; 8];
    let pfs_opts = make_opts(&[
        ("padding", JsValue::from_f64(0.05)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut pfs_step = JsPfsPrimeOverlapRemoval::new(&hw, &hh, &pfs_opts).unwrap();
    let mut pfs_run = JsPfsPrimeOverlapRemoval::new(&hw, &hh, &pfs_opts).unwrap();
    assert_eq!(pfs_step.total_iterations(), 1);
    let mut l_pfs_step = overlap_init.clone();
    let mut l_pfs_run = overlap_init.clone();
    while !pfs_step.is_finished() {
        pfs_step.step(&mut l_pfs_step).unwrap();
    }
    pfs_run.run(&mut l_pfs_run).unwrap();
    assert_layouts_bit_exact(&l_pfs_step, &l_pfs_run);
    assert_eq!(pfs_run.last_overlap_count(), 0);
    assert_eq!(count_rect_overlaps(&l_pfs_run, &hw, &hh), 0);

    let mut pfs_radii = JsPfsPrimeOverlapRemoval::from_radii(&hw, &pfs_opts).unwrap();
    let mut l_pfs_rad = overlap_init;
    pfs_radii.run(&mut l_pfs_rad).unwrap();
    assert_eq!(pfs_radii.last_overlap_count(), 0);
    assert_eq!(count_rect_overlaps(&l_pfs_rad, &hw, &hw), 0);

    // SFC (Hilbert & Morton)
    for curve in ["hilbert", "morton"] {
        let sfc_opts = make_opts(&[
            ("curve", JsValue::from_str(curve)),
            ("scale", JsValue::from_f64(100.0)),
            ("resolutionOrder", JsValue::from_f64(3.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]);
        let mut sfc_step = JsSfcLayoutSolver::new(&graph, &sfc_opts).unwrap();
        let mut sfc_run = JsSfcLayoutSolver::new(&graph, &sfc_opts).unwrap();
        assert_eq!(sfc_step.total_iterations(), 1);
        let order = sfc_step.vertex_order();
        assert_eq!(order.len(), 8);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5, 6, 7]);

        let mut l_sfc_step = init.clone();
        let mut l_sfc_run = init.clone();
        while !sfc_step.is_finished() {
            sfc_step.step(&mut l_sfc_step).unwrap();
        }
        sfc_run.run(&mut l_sfc_run).unwrap();
        assert_layouts_bit_exact(&l_sfc_step, &l_sfc_run);

        // Zero vertex co-location check
        for i in 0..8 {
            for j in (i + 1)..8 {
                let dx = l_sfc_run.x(i as f64).unwrap() - l_sfc_run.x(j as f64).unwrap();
                let dy = l_sfc_run.y(i as f64).unwrap() - l_sfc_run.y(j as f64).unwrap();
                assert!(dx != 0.0 || dy != 0.0, "SFC co-located nodes {i} and {j}");
            }
        }
    }

    // Degenerate N=0 / N=1 non-panic checks across solvers
    let g0 = JsCsrGraph::from_edge_list(0.0, &[], None, &JsValue::UNDEFINED, None).unwrap();
    let mut l0 = JsLayout2d::zeros(0.0).unwrap();
    JsGripSolver::new(&g0, &JsValue::UNDEFINED)
        .unwrap()
        .run(&mut l0)
        .unwrap();
    JsGemSolver::new(&g0, &JsValue::UNDEFINED)
        .unwrap()
        .run(&mut l0)
        .unwrap();
    JsSelLayoutSolver::new(&g0, &JsValue::UNDEFINED)
        .unwrap()
        .run(&mut l0)
        .unwrap();
    JsPfsPrimeOverlapRemoval::new(&[], &[], &JsValue::UNDEFINED)
        .unwrap()
        .run(&mut l0)
        .unwrap();
    JsSfcLayoutSolver::new(&g0, &JsValue::UNDEFINED)
        .unwrap()
        .run(&mut l0)
        .unwrap();

    let g1 = JsCsrGraph::from_edge_list(1.0, &[], None, &JsValue::UNDEFINED, None).unwrap();
    let mut l1 = JsLayout2d::zeros(1.0).unwrap();
    JsGripSolver::new(&g1, &JsValue::UNDEFINED)
        .unwrap()
        .run(&mut l1)
        .unwrap();
    JsGemSolver::new(&g1, &JsValue::UNDEFINED)
        .unwrap()
        .run(&mut l1)
        .unwrap();
    JsSelLayoutSolver::new(&g1, &JsValue::UNDEFINED)
        .unwrap()
        .run(&mut l1)
        .unwrap();
    JsSfcLayoutSolver::new(&g1, &JsValue::UNDEFINED)
        .unwrap()
        .run(&mut l1)
        .unwrap();
}
