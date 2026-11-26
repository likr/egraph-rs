mod util;

use egraph_wasm::csr_graph::JsCsrGraph;
use egraph_wasm::dog_varadhan::js_compute_regularized_dog_varadhan_landmarks;
use egraph_wasm::linalg_spectral::js_pivot_mds_init;
use egraph_wasm::sgd::JsSgdSolver;
use util::fixtures::{assert_layouts_bit_exact, make_grid, make_opts};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn sgd_snap_t_fdp_pacmap_sgd2_and_squared_schedule() {
    let grid = make_grid(4, 4);
    let init = js_pivot_mds_init(&grid, None, None, None, &JsValue::UNDEFINED).unwrap();

    // 1. SgdSolver.snapTFdp
    let snap_opts = make_opts(&[
        ("iterations", JsValue::from_f64(15.0)),
        ("negativesPerEdge", JsValue::from_f64(3.0)),
        ("learningRate", JsValue::from_f64(0.5)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut snap_run = JsSgdSolver::snap_t_fdp(&grid, &snap_opts).unwrap();
    let mut snap_step = JsSgdSolver::snap_t_fdp(&grid, &snap_opts).unwrap();
    assert_eq!(snap_run.num_springs(), grid.num_edges());
    assert_eq!(snap_run.total_iterations(), 15);
    assert!(snap_run.current_eta() > 0.0);

    let mut l_snap_run = init.clone();
    let mut l_snap_step = init.clone();
    assert_eq!(snap_run.run(&mut l_snap_run).unwrap(), "completed");
    assert_eq!(snap_run.current_eta(), 0.0);
    assert_eq!(
        snap_step.run_steps(&mut l_snap_step, 5.0).unwrap(),
        "running"
    );
    while !snap_step.is_finished() {
        snap_step.step(&mut l_snap_step).unwrap();
    }
    assert_layouts_bit_exact(&l_snap_run, &l_snap_step);

    // 1b. SgdSolver.snapTFdp degreeBeta + boundedAttraction=false diverges from defaults + step == run
    let snap_mod_opts = make_opts(&[
        ("iterations", JsValue::from_f64(15.0)),
        ("negativesPerEdge", JsValue::from_f64(3.0)),
        ("learningRate", JsValue::from_f64(0.5)),
        ("degreeBeta", JsValue::from_f64(0.5)),
        ("boundedAttraction", JsValue::from_bool(false)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut snap_mod_run = JsSgdSolver::snap_t_fdp(&grid, &snap_mod_opts).unwrap();
    let mut snap_mod_step = JsSgdSolver::snap_t_fdp(&grid, &snap_mod_opts).unwrap();
    let mut l_snap_mod_run = init.clone();
    let mut l_snap_mod_step = init.clone();
    assert_eq!(snap_mod_run.run(&mut l_snap_mod_run).unwrap(), "completed");
    while !snap_mod_step.is_finished() {
        snap_mod_step.step(&mut l_snap_mod_step).unwrap();
    }
    assert_layouts_bit_exact(&l_snap_mod_run, &l_snap_mod_step);
    let max_snap_diff = l_snap_run
        .copy_coords()
        .iter()
        .zip(l_snap_mod_run.copy_coords().iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_snap_diff > 1e-3,
        "expected degreeBeta/boundedAttraction override to diverge, got {max_snap_diff}"
    );

    // 2. SgdSolver.pacmapGraph
    let pacmap_opts = make_opts(&[
        ("nNeighbors", JsValue::from_f64(6.0)),
        ("mnRatio", JsValue::from_f64(0.5)),
        ("fpRatio", JsValue::from_f64(1.5)),
        ("phase1Iters", JsValue::from_f64(5.0)),
        ("phase2Iters", JsValue::from_f64(5.0)),
        ("phase3Iters", JsValue::from_f64(5.0)),
        ("learningRate", JsValue::from_f64(0.2)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut pm_run = JsSgdSolver::pacmap_graph(&grid, &pacmap_opts).unwrap();
    let mut pm_step = JsSgdSolver::pacmap_graph(&grid, &pacmap_opts).unwrap();
    assert_eq!(pm_run.total_iterations(), 15);
    assert!(pm_run.num_springs() > 0);
    assert!(pm_run.num_hinges() > 0);

    let mut l_pm_run = init.clone();
    let mut l_pm_step = init.clone();
    assert_eq!(pm_run.run(&mut l_pm_run).unwrap(), "completed");
    assert_eq!(pm_step.run_steps(&mut l_pm_step, 6.0).unwrap(), "running");
    while !pm_step.is_finished() {
        pm_step.step(&mut l_pm_step).unwrap();
    }
    assert_layouts_bit_exact(&l_pm_run, &l_pm_step);

    // SgdSolver.pacmapGraph keepAll1hop=true retains all hub 1-hop edges beyond nNeighbors
    let star_edges: Vec<u32> = (1..=8u32).flat_map(|i| [0, i]).collect();
    let star = JsCsrGraph::from_edge_list(9.0, &star_edges, None, &JsValue::from_bool(false), None)
        .unwrap();
    let pm_keep = JsSgdSolver::pacmap_graph(
        &star,
        &make_opts(&[
            ("nNeighbors", JsValue::from_f64(2.0)),
            ("keepAll1hop", JsValue::from_bool(true)),
            ("phase1Iters", JsValue::from_f64(2.0)),
            ("phase2Iters", JsValue::from_f64(2.0)),
            ("phase3Iters", JsValue::from_f64(2.0)),
        ]),
    )
    .unwrap();
    assert!(pm_keep.num_springs() >= 8);

    // max1hopPerNode + degreeWeightBeta + pacmapGraphWithLandmarks + pacmapGraphWithLandmarkDistances
    let pm_cap = JsSgdSolver::pacmap_graph(
        &star,
        &make_opts(&[
            ("nNeighbors", JsValue::from_f64(2.0)),
            ("keepAll1hop", JsValue::from_bool(true)),
            ("max1hopPerNode", JsValue::from_f64(3.0)),
            ("degreeWeightBeta", JsValue::from_f64(0.5)),
            ("phase1Iters", JsValue::from_f64(2.0)),
            ("phase2Iters", JsValue::from_f64(2.0)),
            ("phase3Iters", JsValue::from_f64(2.0)),
        ]),
    )
    .unwrap();
    assert!(pm_cap.num_springs() <= 24);
    let dog_lm = js_compute_regularized_dog_varadhan_landmarks(
        &grid,
        Some(6.0),
        Some(6.0),
        Some(7.0),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let mut pm_lm = JsSgdSolver::pacmap_graph_with_landmarks(&grid, &dog_lm, &pacmap_opts).unwrap();
    let mut pm_dist = JsSgdSolver::pacmap_graph_with_landmark_distances(
        &grid,
        &dog_lm.table(),
        dog_lm.num_landmarks() as f64,
        &pacmap_opts,
    )
    .unwrap();
    let mut l_pm_lm = init.clone();
    let mut l_pm_dist = init.clone();
    assert_eq!(pm_lm.run(&mut l_pm_lm).unwrap(), "completed");
    assert_eq!(pm_dist.run(&mut l_pm_dist).unwrap(), "completed");
    assert_layouts_bit_exact(&l_pm_lm, &l_pm_dist);

    // 3. SgdSolver.sgd2 + "squared" schedule kind on SgdSolver.full
    let sgd2_opts = make_opts(&[
        ("tMax", JsValue::from_f64(12.0)),
        ("batchSize", JsValue::from_f64(16.0)),
        ("damping", JsValue::from_f64(1e-4)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut s2_run = JsSgdSolver::sgd2(&grid, &sgd2_opts).unwrap();
    let mut s2_step = JsSgdSolver::sgd2(&grid, &sgd2_opts).unwrap();
    let mut l_s2_run = init.clone();
    let mut l_s2_step = init.clone();
    assert_eq!(s2_run.run(&mut l_s2_run).unwrap(), "completed");
    assert_eq!(s2_step.run_steps(&mut l_s2_step, 4.0).unwrap(), "running");
    while !s2_step.is_finished() {
        s2_step.step(&mut l_s2_step).unwrap();
    }
    assert_layouts_bit_exact(&l_s2_run, &l_s2_step);

    let sq_opts = make_opts(&[
        ("tMax", JsValue::from_f64(10.0)),
        ("kind", JsValue::from_str("squared")),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut sq_run = JsSgdSolver::full(&grid, &sq_opts, None, None, None, None).unwrap();
    let mut sq_step = JsSgdSolver::full(&grid, &sq_opts, None, None, None, None).unwrap();
    let mut l_sq_run = init.clone();
    let mut l_sq_step = init;
    assert_eq!(sq_run.run(&mut l_sq_run).unwrap(), "completed");
    sq_step.run_steps(&mut l_sq_step, 3.0).unwrap();
    while !sq_step.is_finished() {
        sq_step.step(&mut l_sq_step).unwrap();
    }
    assert_layouts_bit_exact(&l_sq_run, &l_sq_step);

    // Adversarial checks
    assert!(
        JsSgdSolver::pacmap_graph(&grid, &make_opts(&[("nNeighbors", JsValue::from_f64(0.0))]))
            .is_err()
    );
    assert!(
        JsSgdSolver::sgd2(&grid, &make_opts(&[("batchSize", JsValue::from_f64(0.0))])).is_err()
    );
    assert!(JsSgdSolver::snap_t_fdp(&grid, &JsValue::from_f64(123.0)).is_err());
    assert!(
        JsSgdSolver::snap_t_fdp(
            &grid,
            &make_opts(&[("degreeBeta", JsValue::from_f64(-0.1))])
        )
        .is_err()
    );
    assert!(
        JsSgdSolver::snap_t_fdp(
            &grid,
            &make_opts(&[("boundedAttraction", JsValue::from_str("nope"))])
        )
        .is_err()
    );
}
