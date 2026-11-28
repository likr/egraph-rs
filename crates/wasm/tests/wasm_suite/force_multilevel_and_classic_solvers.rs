use crate::util;

use egraph_wasm::force_classic::JsLinLogSolver;
use egraph_wasm::force_fr::JsFruchtermanReingoldSolver;
use egraph_wasm::force_multilevel::{JsForceAtlas2Solver, JsYifanHuSolver};
use egraph_wasm::layout_buffers::JsLayout2d;
use util::fixtures::{
    assert_layouts_bit_exact, f64_js, make_barbell, make_circle_layout, make_grid, make_opts,
    make_two_cluster_graph,
};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn force_atlas2_and_yifan_hu_solvers() {
    let barbell = make_barbell(6);
    let init_b = make_circle_layout(12, 1.0);

    // 1. ForceAtlas2Solver (Barnes-Hut + Exact, LinLog + StrongGravity)
    for use_bh in [true, false] {
        let fa2_opts = make_opts(&[
            ("iterations", JsValue::from_f64(15.0)),
            ("useBarnesHut", JsValue::from_bool(use_bh)),
            ("linlogMode", JsValue::from_bool(!use_bh)),
            ("strongGravity", JsValue::from_bool(!use_bh)),
            ("outboundAttractionDistribution", JsValue::from_bool(use_bh)),
            ("seed", JsValue::from_f64(42.0)),
        ]);
        let mut fa2_run = JsForceAtlas2Solver::new(&barbell, &fa2_opts).unwrap();
        let mut fa2_step = JsForceAtlas2Solver::new(&barbell, &fa2_opts).unwrap();
        let mut l_run = init_b.clone();
        let mut l_step = init_b.clone();
        let ptr_before = l_step.coords_ptr();

        assert_eq!(fa2_run.run_2d(&mut l_run).unwrap(), "completed");
        assert!(l_run.is_all_finite());
        assert_eq!(fa2_step.run_steps_2d(&mut l_step, 5.0).unwrap(), "running");
        while !fa2_step.is_finished() {
            fa2_step.step_2d(&mut l_step).unwrap();
        }
        assert_eq!(l_step.coords_ptr(), ptr_before);
        assert_layouts_bit_exact(&l_run, &l_step);
    }

    // 1b. ForceAtlas2Solver maxDisplacement overrides default, keeps step == run, and caps displacement
    let wide_coords: Vec<f32> = init_b
        .copy_coords()
        .into_iter()
        .map(|v| v * 150.0)
        .collect();
    let wide_init = JsLayout2d::from_coords(&wide_coords).unwrap();
    let fa2_def_opts = make_opts(&[
        ("iterations", JsValue::from_f64(4.0)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let fa2_cap_opts = make_opts(&[
        ("iterations", JsValue::from_f64(4.0)),
        ("maxDisplacement", JsValue::from_f64(0.2)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut fa2_def = JsForceAtlas2Solver::new(&barbell, &fa2_def_opts).unwrap();
    let mut fa2_cap_r = JsForceAtlas2Solver::new(&barbell, &fa2_cap_opts).unwrap();
    let mut fa2_cap_s = JsForceAtlas2Solver::new(&barbell, &fa2_cap_opts).unwrap();
    let mut l_fa_def = wide_init.clone();
    let mut l_fa_cap_r = wide_init.clone();
    let mut l_fa_cap_s = wide_init;
    assert_eq!(fa2_def.run_2d(&mut l_fa_def).unwrap(), "completed");
    assert_eq!(fa2_cap_r.run_2d(&mut l_fa_cap_r).unwrap(), "completed");
    while !fa2_cap_s.is_finished() {
        fa2_cap_s.step_2d(&mut l_fa_cap_s).unwrap();
    }
    assert_layouts_bit_exact(&l_fa_cap_r, &l_fa_cap_s);
    let max_fa_diff = l_fa_def
        .copy_coords()
        .iter()
        .zip(l_fa_cap_r.copy_coords().iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_fa_diff > 0.1,
        "expected maxDisplacement=0.2 to diverge from default, got {max_fa_diff}"
    );

    // 2. YifanHuSolver (multilevel EC + MIVS hierarchy, numLevels >= 2, currentLevel -> 0)
    let grid6 = make_grid(6, 6);
    let init36 = make_circle_layout(36, 1.0);
    let yh_opts = make_opts(&[
        ("maxIterationsPerLevel", JsValue::from_f64(12.0)),
        ("minCoarsenSize", JsValue::from_f64(8.0)),
        ("minCoarsenRatio", JsValue::from_f64(0.85)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut yh_run = JsYifanHuSolver::new(&grid6, &yh_opts).unwrap();
    let mut yh_step = JsYifanHuSolver::new(&grid6, &yh_opts).unwrap();
    assert!(yh_run.num_levels() >= 2);
    assert_eq!(yh_run.current_level(), yh_run.num_levels() - 1);

    let mut l_yh_run = init36.clone();
    let mut l_yh_step = init36;
    let ptr_before = l_yh_step.coords_ptr();

    let st = yh_run.run_2d(&mut l_yh_run).unwrap();
    assert!(st == "completed" || st == "converged");
    assert_eq!(yh_run.current_level(), 0);
    assert!(l_yh_run.is_all_finite());

    yh_step.run_steps_2d(&mut l_yh_step, 4.0).unwrap();
    while !yh_step.is_finished() {
        yh_step.step_2d(&mut l_yh_step).unwrap();
    }
    assert_eq!(l_yh_step.coords_ptr(), ptr_before);
    assert_eq!(yh_step.current_level(), 0);
    assert_layouts_bit_exact(&l_yh_run, &l_yh_step);

    // Adversarial checks
    assert!(
        JsForceAtlas2Solver::new(&barbell, &make_opts(&[("theta", JsValue::from_f64(-0.1))]))
            .is_err()
    );
    assert!(
        JsForceAtlas2Solver::new(
            &barbell,
            &make_opts(&[("maxDisplacement", JsValue::from_f64(0.0))])
        )
        .is_err()
    );
    assert!(
        JsForceAtlas2Solver::new(
            &barbell,
            &make_opts(&[("maxDisplacement", JsValue::from_str("bad"))])
        )
        .is_err()
    );
    assert!(
        JsYifanHuSolver::new(&grid6, &make_opts(&[("stepRatio", JsValue::from_f64(1.0))])).is_err()
    );
    assert!(
        JsYifanHuSolver::new(
            &grid6,
            &make_opts(&[("minCoarsenSize", JsValue::from_f64(0.0))])
        )
        .is_err()
    );
}

#[wasm_bindgen_test]
fn linlog_fr_and_fa2_overlap_step_eq_run() {
    let (graph, _) = make_two_cluster_graph();
    let init = make_circle_layout(8, 2.0);

    // LinLogSolver (edgeRepulsion & nodeRepulsion)
    for rep in ["edgeRepulsion", "nodeRepulsion"] {
        let opts = make_opts(&[
            ("iterations", JsValue::from_f64(10.0)),
            ("repulsionModel", JsValue::from_str(rep)),
            ("useBarnesHut", JsValue::from_bool(true)),
            ("seed", JsValue::from_f64(42.0)),
        ]);
        let mut s_step = JsLinLogSolver::new(&graph, &opts).unwrap();
        let mut s_run = JsLinLogSolver::new(&graph, &opts).unwrap();
        assert_eq!(s_step.num_nodes(), 8);
        assert_eq!(s_step.total_iterations(), Some(10));
        assert_eq!(s_step.masses().len(), 8);
        assert!(s_step.max_displacement() > 0.0);

        let mut l_step = init.clone();
        let mut l_run = init.clone();
        while !s_step.is_finished() {
            s_step.step_2d(&mut l_step).unwrap();
        }
        let status = s_run.run_2d(&mut l_run).unwrap();
        assert!(matches!(status.as_str(), "completed" | "converged"));
        assert_layouts_bit_exact(&l_step, &l_run);
        assert!(s_run.energy_2d(&l_run).unwrap().is_finite());
    }
    let bad_ll = make_opts(&[
        ("attractionExponent", JsValue::from_f64(-1.0)),
        ("repulsionExponent", JsValue::from_f64(0.0)),
    ]);
    assert!(JsLinLogSolver::new(&graph, &bad_ll).is_err());

    // FruchtermanReingoldSolver (exact, barnesHut, grid)
    for backend in ["exact", "barnesHut", "grid"] {
        let opts = make_opts(&[
            ("iterations", JsValue::from_f64(10.0)),
            ("backend", JsValue::from_str(backend)),
            ("gravity", JsValue::from_f64(0.1)),
            ("seed", JsValue::from_f64(42.0)),
        ]);
        let mut s_step = JsFruchtermanReingoldSolver::new(&graph, &opts).unwrap();
        let mut s_run = JsFruchtermanReingoldSolver::new(&graph, &opts).unwrap();
        assert!(s_step.optimal_distance() > 0.0);
        assert!(s_step.current_temperature() > 0.0);
        let mut l_step = init.clone();
        let mut l_run = init.clone();
        while !s_step.is_finished() {
            s_step.step_2d(&mut l_step).unwrap();
        }
        s_run.run_2d(&mut l_run).unwrap();
        assert_layouts_bit_exact(&l_step, &l_run);
        assert_eq!(s_run.current_temperature(), 0.0);
    }

    // ForceAtlas2Solver with preventOverlap + nodeRadii
    let radii = vec![0.4; 8];
    let fa2_opts = make_opts(&[
        ("iterations", JsValue::from_f64(10.0)),
        ("preventOverlap", JsValue::from_bool(true)),
        ("defaultNodeRadius", JsValue::from_f64(0.5)),
        ("nodeRadii", f64_js(&radii)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut fa_step = JsForceAtlas2Solver::new(&graph, &fa2_opts).unwrap();
    let mut fa_run = JsForceAtlas2Solver::new(&graph, &fa2_opts).unwrap();
    assert_eq!(fa_step.node_radii(), radii);
    assert!(fa_step.max_displacement() > 0.0);
    let mut l_fa_step = init.clone();
    let mut l_fa_run = init;
    while !fa_step.is_finished() {
        fa_step.step_2d(&mut l_fa_step).unwrap();
    }
    fa_run.run_2d(&mut l_fa_run).unwrap();
    assert_layouts_bit_exact(&l_fa_step, &l_fa_run);

    let bad_fa2 = make_opts(&[
        ("preventOverlap", JsValue::from_bool(true)),
        ("nodeRadii", f64_js(&[1.0, 2.0])),
    ]);
    assert!(JsForceAtlas2Solver::new(&graph, &bad_fa2).is_err());

    // Conflicting backend + useBarnesHut must fail fast with InvalidInput
    let conflict_exact = make_opts(&[
        ("backend", JsValue::from_str("exact")),
        ("useBarnesHut", JsValue::from_bool(true)),
    ]);
    let conflict_bh = make_opts(&[
        ("backend", JsValue::from_str("barnesHut")),
        ("useBarnesHut", JsValue::from_bool(false)),
    ]);
    for err in [
        JsForceAtlas2Solver::new(&graph, &conflict_exact).unwrap_err(),
        JsForceAtlas2Solver::new(&graph, &conflict_bh).unwrap_err(),
        JsLinLogSolver::new(&graph, &conflict_exact).unwrap_err(),
        JsYifanHuSolver::new(&graph, &conflict_exact).unwrap_err(),
    ] {
        assert_eq!(
            js_sys::Reflect::get(&err, &"code".into())
                .unwrap()
                .as_string(),
            Some("InvalidInput".to_owned())
        );
    }
}
