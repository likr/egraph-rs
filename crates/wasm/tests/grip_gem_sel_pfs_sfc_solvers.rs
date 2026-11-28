mod util;

use egraph_wasm::csr_graph::JsCsrGraph;
use egraph_wasm::force_baselines::{JsGemSolver, JsSelLayoutSolver};
use egraph_wasm::layout_buffers::JsLayout2d;
use egraph_wasm::overlap_pfs_sfc::{JsPfsPrimeOverlapRemoval, JsSfcLayout};
use egraph_wasm::stress_multilevel_baselines::JsGripSolver;
use util::fixtures::{
    assert_layouts_bit_exact, count_rect_overlaps, make_circle_layout, make_disconnected_graph,
    make_opts, make_overlapping_layout, make_two_cluster_graph,
};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn grip_gem_sel_pfs_prime_sfc_step_eq_run() {
    let (graph, _) = make_two_cluster_graph();
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
    assert!(grip_step.total_iterations().is_some_and(|v| v >= 1));
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
        ("maxIterations", JsValue::from_f64(10.0)),
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
    assert_eq!(gem_step.total_iterations(), Some(10));
    let mut l_gem_step = init.clone();
    let mut l_gem_run = init.clone();
    while !gem_step.is_finished() {
        gem_step.step(&mut l_gem_step).unwrap();
    }
    gem_run.run(&mut l_gem_run).unwrap();
    assert_layouts_bit_exact(&l_gem_step, &l_gem_run);
    assert!(l_gem_run.is_all_finite());

    // SEL (all 5 mappings + both backends)
    for mapping in ["sigmoid", "linear", "tanh", "customLogistic", "binary"] {
        for backend in ["fr", "tFdp"] {
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

    // PFS' (LayoutTransform.apply)
    let overlap_init = make_overlapping_layout();
    let hw = vec![0.4_f32; 8];
    let hh = vec![0.3_f32; 8];
    let pfs_opts = make_opts(&[
        ("padding", JsValue::from_f64(0.05)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut pfs = JsPfsPrimeOverlapRemoval::new(&hw, &hh, &pfs_opts).unwrap();
    let mut l_pfs = overlap_init.clone();
    pfs.apply(&mut l_pfs).unwrap();
    assert_eq!(pfs.last_overlap_count(), 0);
    assert_eq!(count_rect_overlaps(&l_pfs, &hw, &hh), 0);

    let mut pfs_radii = JsPfsPrimeOverlapRemoval::from_radii(&hw, &pfs_opts).unwrap();
    let mut l_pfs_rad = overlap_init;
    pfs_radii.apply(&mut l_pfs_rad).unwrap();
    assert_eq!(pfs_radii.last_overlap_count(), 0);
    assert_eq!(count_rect_overlaps(&l_pfs_rad, &hw, &hw), 0);

    // SFC (Hilbert & Morton LayoutGenerator)
    for curve in ["hilbert", "morton"] {
        let sfc_opts = make_opts(&[
            ("curve", JsValue::from_str(curve)),
            ("scale", JsValue::from_f64(100.0)),
            ("resolutionOrder", JsValue::from_f64(3.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]);
        let mut sfc_a = JsSfcLayout::new(&graph, &sfc_opts).unwrap();
        let mut sfc_b = JsSfcLayout::new(&graph, &sfc_opts).unwrap();
        let l_sfc_a = sfc_a.generate().unwrap();
        let l_sfc_b = sfc_b.generate().unwrap();
        assert_layouts_bit_exact(&l_sfc_a, &l_sfc_b);

        let order = sfc_a.vertex_order();
        assert_eq!(order.len(), 8);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5, 6, 7]);

        // Zero vertex co-location check
        for i in 0..8 {
            for j in (i + 1)..8 {
                let dx = l_sfc_a.x(i as f64).unwrap() - l_sfc_a.x(j as f64).unwrap();
                let dy = l_sfc_a.y(i as f64).unwrap() - l_sfc_a.y(j as f64).unwrap();
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
        .apply(&mut l0)
        .unwrap();
    JsSfcLayout::new(&g0, &JsValue::UNDEFINED)
        .unwrap()
        .generate()
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
    JsSfcLayout::new(&g1, &JsValue::UNDEFINED)
        .unwrap()
        .generate()
        .unwrap();
}
