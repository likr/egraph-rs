use egraph_core::{
    CoreError, CsrGraph, IterativeSolver, Layout2d, LayoutGenerator, ManifoldKind,
    ManifoldLayout2d, SolverStatus,
};
use egraph_linalg::BfsWorkspace;
use egraph_linalg_mds::KorenSpectral;

use super::super::sgd::{
    Sgd2Config, Sgd2Solver, SgdScheduleConfig, SgdScheduleKind, SgdSolver, StreamingNegativeConfig,
};
use super::test_support::{
    circle_layout, compute_np2, euclidean_apsp_stress, grid_graph, manifold_pair_stress,
};

#[test]
fn test_step_vs_run_bit_exact_f64_and_f32_all_modes_and_schedules() {
    let g = grid_graph(3, 4);
    for kind in [
        SgdScheduleKind::Exponential,
        SgdScheduleKind::Linear,
        SgdScheduleKind::Quadratic,
        SgdScheduleKind::Reciprocal,
        SgdScheduleKind::Constant,
        SgdScheduleKind::Squared,
    ] {
        let sched = SgdScheduleConfig::new(15)
            .with_eps(0.01)
            .with_kind(kind)
            .with_step_scale(0.85);

        let mut s1_f64 = SgdSolver::<Layout2d<f64>>::full(&g, sched.clone(), 123).unwrap();
        let mut s2_f64 = SgdSolver::<Layout2d<f64>>::full(&g, sched.clone(), 123).unwrap();
        let mut l1_f64 = circle_layout::<f64>(g.num_nodes(), 1.0);
        let mut l2_f64 = l1_f64.clone();

        assert_eq!(s1_f64.status(), SolverStatus::Initialized);
        let initial_eta = s1_f64.current_eta();
        assert!(initial_eta > 0.0);

        assert_eq!(s1_f64.run(&mut l1_f64).unwrap(), SolverStatus::Completed);
        assert_eq!(s1_f64.current_eta(), 0.0);
        assert_eq!(
            s2_f64.run_steps(&mut l2_f64, 5).unwrap(),
            SolverStatus::Running
        );
        assert_eq!(s2_f64.iteration(), 5);
        for _ in 0..4 {
            s2_f64.step(&mut l2_f64).unwrap();
        }
        assert_eq!(s2_f64.run(&mut l2_f64).unwrap(), SolverStatus::Completed);
        assert_eq!(s2_f64.iteration(), 15);
        assert_eq!(s2_f64.total_iterations(), Some(15));
        assert_eq!(s2_f64.step(&mut l2_f64).unwrap(), SolverStatus::Completed);

        for (a, b) in l1_f64.x().iter().zip(l2_f64.x().iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        for (a, b) in l1_f64.y().iter().zip(l2_f64.y().iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }

        let mut s1_f32 = SgdSolver::<Layout2d<f32>>::full(&g, sched.clone(), 123).unwrap();
        let mut s2_f32 = SgdSolver::<Layout2d<f32>>::full(&g, sched, 123).unwrap();
        let mut l1_f32 = circle_layout::<f32>(g.num_nodes(), 1.0);
        let mut l2_f32 = l1_f32.clone();
        s1_f32.run(&mut l1_f32).unwrap();
        s2_f32.run_steps(&mut l2_f32, 7).unwrap();
        s2_f32.run(&mut l2_f32).unwrap();
        for (a, b) in l1_f32.x().iter().zip(l2_f32.x().iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        for (a, b) in l1_f32.y().iter().zip(l2_f32.y().iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    let sched = SgdScheduleConfig::new(12);
    let mut sn1 = SgdSolver::<Layout2d<f32>>::streaming_negatives(
        &g,
        StreamingNegativeConfig::default(),
        sched.clone(),
        77,
    )
    .unwrap();
    let mut sn2 = SgdSolver::<Layout2d<f32>>::streaming_negatives(
        &g,
        StreamingNegativeConfig::default(),
        sched,
        77,
    )
    .unwrap();
    let mut ln1 = circle_layout::<f32>(g.num_nodes(), 1.0);
    let mut ln2 = ln1.clone();
    sn1.run(&mut ln1).unwrap();
    sn2.run_steps(&mut ln2, 4).unwrap();
    sn2.step(&mut ln2).unwrap();
    sn2.run(&mut ln2).unwrap();
    for (a, b) in ln1.x().iter().zip(ln2.x().iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    for (a, b) in ln1.y().iter().zip(ln2.y().iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn test_hinge_zero_force_when_separated_and_repels_close() {
    let sched = SgdScheduleConfig::new(10)
        .with_eps(0.01)
        .with_kind(SgdScheduleKind::Constant)
        .with_step_scale(0.5);

    let mut far_solver = SgdSolver::<Layout2d<f64>>::from_pairs_with_hinges(
        2,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[0],
        &[1],
        &[2.0],
        &[2.0],
        &[1.0],
        &[1.0],
        sched.clone(),
        42,
    )
    .unwrap();
    assert_eq!(far_solver.num_nodes(), 2);
    assert_eq!(far_solver.num_springs(), 0);
    assert_eq!(far_solver.num_hinges(), 1);

    let mut far_layout = Layout2d::from_xy(vec![0.0, 3.0], vec![0.0, 0.0]).unwrap();
    far_solver.step(&mut far_layout).unwrap();
    assert_eq!(far_layout.x()[0].to_bits(), 0.0_f64.to_bits());
    assert_eq!(far_layout.y()[0].to_bits(), 0.0_f64.to_bits());
    assert_eq!(far_layout.x()[1].to_bits(), 3.0_f64.to_bits());
    assert_eq!(far_layout.y()[1].to_bits(), 0.0_f64.to_bits());

    let mut boundary_layout = Layout2d::from_xy(vec![0.0, 2.0], vec![0.0, 0.0]).unwrap();
    far_solver.step(&mut boundary_layout).unwrap();
    assert_eq!(boundary_layout.x()[0].to_bits(), 0.0_f64.to_bits());
    assert_eq!(boundary_layout.x()[1].to_bits(), 2.0_f64.to_bits());

    let mut close_layout = Layout2d::from_xy(vec![0.0, 0.5], vec![0.0, 0.0]).unwrap();
    far_solver.step(&mut close_layout).unwrap();
    assert!(close_layout.x()[0] < 0.0);
    assert!(close_layout.x()[1] > 0.5);
    assert_eq!(close_layout.y()[0].to_bits(), 0.0_f64.to_bits());
    assert_eq!(close_layout.y()[1].to_bits(), 0.0_f64.to_bits());

    let mut asym_solver = SgdSolver::<Layout2d<f64>>::from_pairs_with_hinges(
        2,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[0],
        &[1],
        &[2.0],
        &[2.0],
        &[0.0],
        &[1.0],
        sched,
        42,
    )
    .unwrap();
    let mut asym_layout = Layout2d::from_xy(vec![0.0, 0.5], vec![0.0, 0.0]).unwrap();
    asym_solver.step(&mut asym_layout).unwrap();
    assert_eq!(asym_layout.x()[0].to_bits(), 0.0_f64.to_bits());
    assert!(asym_layout.x()[1] > 0.5);
}

#[test]
fn test_manifolds_all_three_domain_invariants_and_stress_reduction() {
    let sched = SgdScheduleConfig::new(50)
        .with_eps(0.02)
        .with_kind(SgdScheduleKind::Exponential)
        .with_step_scale(1.0);

    // 1. Hyperbolic Poincare disk on a 7-node binary tree.
    let tree_edges = [(0u32, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)];
    let tree = CsrGraph::<f64>::from_edges(7, &tree_edges, None, false, false).unwrap();
    let mut tree_pairs = Vec::new();
    let mut bfs = BfsWorkspace::new(7);
    let mut row = vec![0.0_f64; 7];
    for i in 0..7 {
        bfs.bfs_from(&tree, i, &mut row).unwrap();
        for (j, &d_t) in row.iter().enumerate().skip(i + 1) {
            tree_pairs.push((i, j, d_t));
        }
    }
    let mut hyp_solver =
        SgdSolver::<ManifoldLayout2d<f64>>::full(&tree, sched.clone(), 11).unwrap();
    let mut hyp_solver_step =
        SgdSolver::<ManifoldLayout2d<f64>>::full(&tree, sched.clone(), 11).unwrap();
    let mut hyp_layout =
        ManifoldLayout2d::from_layout(ManifoldKind::HyperbolicPoincare, circle_layout(7, 0.35))
            .unwrap();
    let mut hyp_layout_step = hyp_layout.clone();
    let hyp_before = manifold_pair_stress(&hyp_layout, &tree_pairs);
    while !hyp_solver_step.status().is_finished() {
        hyp_solver_step.step(&mut hyp_layout_step).unwrap();
        for i in 0..7 {
            let r = hyp_layout_step.x()[i].hypot(hyp_layout_step.y()[i]);
            assert!(r.is_finite() && r < 1.0);
        }
    }
    hyp_solver.run(&mut hyp_layout).unwrap();
    assert_eq!(hyp_layout, hyp_layout_step);
    let hyp_after = manifold_pair_stress(&hyp_layout, &tree_pairs);
    assert!(
        hyp_after < hyp_before * 0.5,
        "hyperbolic stress did not drop: {hyp_before} -> {hyp_after}"
    );

    let mut hyp_rand =
        ManifoldLayout2d::<f64>::random_init(ManifoldKind::HyperbolicPoincare, 7, 0.25, 42)
            .unwrap();
    let rand_before = manifold_pair_stress(&hyp_rand, &tree_pairs);
    let mut hyp_rand_solver =
        SgdSolver::<ManifoldLayout2d<f64>>::full(&tree, sched.clone(), 11).unwrap();
    hyp_rand_solver.run(&mut hyp_rand).unwrap();
    let rand_after = manifold_pair_stress(&hyp_rand, &tree_pairs);
    assert!(
        rand_after < rand_before * 0.5,
        "random_init hyperbolic stress did not drop: {rand_before} -> {rand_after}"
    );
    hyp_rand.center_poincare_centroid().unwrap();
    let rand_centered = manifold_pair_stress(&hyp_rand, &tree_pairs);
    assert!((rand_centered - rand_after).abs() < 1e-10);

    // 2. Spherical on a 6-node cycle (diameter 3 < pi).
    let cyc = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
        None,
        false,
        false,
    )
    .unwrap();
    let mut cyc_pairs = Vec::new();
    let mut row6 = vec![0.0_f64; 6];
    for i in 0..6 {
        bfs.bfs_from(&cyc, i, &mut row6).unwrap();
        for (j, &d_c) in row6.iter().enumerate().skip(i + 1) {
            cyc_pairs.push((i, j, d_c));
        }
    }
    let mut sph_solver = SgdSolver::<ManifoldLayout2d<f32>>::full(&cyc, sched.clone(), 19).unwrap();
    let mut sph_layout = ManifoldLayout2d::<f32>::from_xy(
        ManifoldKind::Spherical,
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        vec![1.2, 1.3, 1.1, 1.4, 1.25, 1.35],
    )
    .unwrap();
    let sph_before = manifold_pair_stress(&sph_layout, &cyc_pairs);
    while !sph_solver.status().is_finished() {
        sph_solver.step(&mut sph_layout).unwrap();
        for i in 0..6 {
            let lon = sph_layout.x()[i];
            let colat = sph_layout.y()[i];
            assert!(lon.is_finite() && (0.0_f32..std::f32::consts::TAU).contains(&lon));
            assert!(colat.is_finite() && (0.0_f32..=std::f32::consts::PI).contains(&colat));
        }
    }
    let sph_after = manifold_pair_stress(&sph_layout, &cyc_pairs);
    assert!(
        sph_after < sph_before * 0.5,
        "spherical stress did not drop: {sph_before} -> {sph_after}"
    );

    // 3. Torus2d with scaled edge target lengths (0.15 per hop on 4-cycle).
    let t_src = [0u32, 1, 2, 3, 0, 1];
    let t_tgt = [1u32, 2, 3, 0, 2, 3];
    let t_dist = [0.15_f64, 0.15, 0.15, 0.15, 0.30, 0.30];
    let t_wt: Vec<f64> = t_dist.iter().map(|&d| 1.0 / (d * d)).collect();
    let torus_pairs: Vec<(usize, usize, f64)> = t_src
        .iter()
        .zip(t_tgt.iter())
        .zip(t_dist.iter())
        .map(|((&u, &v), &d)| (u as usize, v as usize, d))
        .collect();
    let mut tor_solver = SgdSolver::<ManifoldLayout2d<f64>>::from_pairs(
        4, &t_src, &t_tgt, &t_dist, &t_dist, &t_wt, &t_wt, sched, 23,
    )
    .unwrap();
    let mut tor_layout = ManifoldLayout2d::from_xy(
        ManifoldKind::Torus2d,
        vec![0.05, 0.85, 0.45, 0.12],
        vec![0.92, 0.10, 0.55, 0.48],
    )
    .unwrap();
    let tor_before = manifold_pair_stress(&tor_layout, &torus_pairs);
    tor_solver.run_steps(&mut tor_layout, 25).unwrap();
    tor_solver.run(&mut tor_layout).unwrap();
    for i in 0..4 {
        let x = tor_layout.x()[i];
        let y = tor_layout.y()[i];
        assert!(x.is_finite() && (0.0..1.0).contains(&x));
        assert!(y.is_finite() && (0.0..1.0).contains(&y));
    }
    let tor_after = manifold_pair_stress(&tor_layout, &torus_pairs);
    assert!(
        tor_after < tor_before * 0.25,
        "torus stress did not drop: {tor_before} -> {tor_after}"
    );

    // Extreme distance clamp test across all 3 manifolds.
    for kind in [
        ManifoldKind::HyperbolicPoincare,
        ManifoldKind::Spherical,
        ManifoldKind::Torus2d,
    ] {
        let mut ext_solver = SgdSolver::<ManifoldLayout2d<f64>>::from_pairs(
            2,
            &[0],
            &[1],
            &[1e6],
            &[1e6],
            &[1.0],
            &[1.0],
            SgdScheduleConfig::new(5).with_kind(SgdScheduleKind::Constant),
            5,
        )
        .unwrap();
        let mut m = ManifoldLayout2d::from_xy(kind, vec![0.1, 0.3], vec![0.2, 0.8]).unwrap();
        ext_solver.run(&mut m).unwrap();
        assert!(m.as_layout().is_all_finite());
    }
}

#[test]
fn test_full_weighted_directed_random_pair_and_omega_converge() {
    let g = grid_graph(3, 3);
    let sched = SgdScheduleConfig::default();

    let wg = CsrGraph::<f64>::from_edges(
        3,
        &[(0, 1), (1, 2), (0, 2)],
        Some(&[1.0, 2.0, 10.0]),
        false,
        false,
    )
    .unwrap();
    let mut w_solver = SgdSolver::<Layout2d<f64>>::full(&wg, sched.clone(), 42).unwrap();
    let mut w_layout = circle_layout::<f64>(3, 2.0);
    w_solver.run(&mut w_layout).unwrap();
    let d02 = (w_layout.x()[0] - w_layout.x()[2]).hypot(w_layout.y()[0] - w_layout.y()[2]);
    assert!(
        (d02 - 3.0).abs() < 0.2,
        "expected weighted shortest-path distance 3.0 for (0,2), got {d02}"
    );

    let dag = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, true, false).unwrap();
    let mut dag_solver = SgdSolver::<Layout2d<f64>>::full(&dag, sched.clone(), 7).unwrap();
    assert_eq!(dag_solver.num_springs(), 3);
    let mut dag_layout = circle_layout::<f64>(3, 0.5);
    dag_solver.run(&mut dag_layout).unwrap();
    assert!(dag_layout.is_all_finite());

    let mut full_solver = SgdSolver::<Layout2d<f64>>::full(&g, sched.clone(), 42).unwrap();
    let mut full_layout = circle_layout::<f64>(g.num_nodes(), 0.25);
    let s_before = euclidean_apsp_stress(&g, &full_layout);
    full_solver.run(&mut full_layout).unwrap();
    let s_after = euclidean_apsp_stress(&g, &full_layout);
    assert!(s_after < s_before * 0.2);

    let mut rp_solver = SgdSolver::<Layout2d<f64>>::random_pair(&g, 4, sched.clone(), 42).unwrap();
    let mut rp_layout = circle_layout::<f64>(g.num_nodes(), 0.25);
    let rp_before = euclidean_apsp_stress(&g, &rp_layout);
    rp_solver.run(&mut rp_layout).unwrap();
    let rp_after = euclidean_apsp_stress(&g, &rp_layout);
    assert!(rp_after < rp_before * 0.3);

    let spec: Layout2d<f64> = KorenSpectral::new(&g).generate().unwrap();
    let mut emb_flat = Vec::with_capacity(g.num_nodes() * 2);
    for i in 0..g.num_nodes() {
        emb_flat.push(spec.x()[i]);
        emb_flat.push(spec.y()[i]);
    }
    let mut om_solver =
        SgdSolver::<Layout2d<f64>>::omega(&g, &emb_flat, 2, 4, 1e-3, sched, 42).unwrap();
    let mut om_layout = circle_layout::<f64>(g.num_nodes(), 0.25);
    om_solver.run(&mut om_layout).unwrap();
    assert!(om_layout.is_all_finite());
    assert!(compute_np2(&g, &om_layout) >= 0.75);
}

#[test]
fn test_reset_schedule_and_run_schedule_window_euclidean_and_manifold() {
    let g = grid_graph(3, 4);
    let sched_a = SgdScheduleConfig::new(10)
        .with_eps(0.01)
        .with_kind(SgdScheduleKind::Exponential)
        .with_step_scale(1.0);

    // 1. Euclidean windowed execution matches full run bit-for-bit.
    let mut s_full = SgdSolver::<Layout2d<f64>>::full(&g, sched_a.clone(), 77).unwrap();
    let mut s_win = SgdSolver::<Layout2d<f64>>::full(&g, sched_a.clone(), 77).unwrap();
    let mut l_full = circle_layout::<f64>(g.num_nodes(), 1.0);
    let mut l_win = l_full.clone();

    s_full.run(&mut l_full).unwrap();
    assert_eq!(
        s_win.run_schedule_window(&mut l_win, 0, 4).unwrap(),
        SolverStatus::Running
    );
    assert_eq!(s_win.iteration(), 4);
    assert_eq!(
        s_win.run_schedule_window(&mut l_win, 4, 6).unwrap(),
        SolverStatus::Completed
    );
    assert_eq!(s_win.iteration(), 10);
    for (a, b) in l_full.x().iter().zip(l_win.x().iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    for (a, b) in l_full.y().iter().zip(l_win.y().iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    // 2. reset_schedule reuses pairs and updates schedule/RNG cleanly.
    let springs_before = s_win.num_springs();
    let sched_b = SgdScheduleConfig::new(6)
        .with_eps(0.05)
        .with_kind(SgdScheduleKind::Linear)
        .with_step_scale(0.5);
    s_win.reset_schedule(sched_b.clone(), 99).unwrap();
    assert_eq!(s_win.num_springs(), springs_before);
    assert_eq!(s_win.iteration(), 0);
    assert_eq!(s_win.total_iterations(), Some(6));
    assert_eq!(s_win.status(), SolverStatus::Initialized);
    assert_eq!(
        s_win.run_schedule_window(&mut l_win, 2, 3).unwrap(),
        SolverStatus::Running
    );
    assert_eq!(s_win.iteration(), 5);
    assert_eq!(
        s_win.run_schedule_window(&mut l_win, 5, 4).unwrap(),
        SolverStatus::Completed
    );
    assert_eq!(s_win.iteration(), 6);

    // 3. Manifold windowed execution matches run bit-for-bit.
    let mut sm_full = SgdSolver::<ManifoldLayout2d<f64>>::full(&g, sched_a.clone(), 55).unwrap();
    let mut sm_win = SgdSolver::<ManifoldLayout2d<f64>>::full(&g, sched_a, 55).unwrap();
    let mut m_full =
        ManifoldLayout2d::from_layout(ManifoldKind::Torus2d, circle_layout(g.num_nodes(), 0.2))
            .unwrap();
    let mut m_win = m_full.clone();
    sm_full.run(&mut m_full).unwrap();
    assert_eq!(
        sm_win.run_schedule_window(&mut m_win, 0, 3).unwrap(),
        SolverStatus::Running
    );
    assert_eq!(
        sm_win.run_schedule_window(&mut m_win, 3, 7).unwrap(),
        SolverStatus::Completed
    );
    assert_eq!(m_full, m_win);

    // 4. Guards: invalid layout len and reset_schedule on Sgd2Solver.
    let mut wrong = Layout2d::<f64>::zeros(2);
    assert!(matches!(
        s_win.run_schedule_window(&mut wrong, 0, 1),
        Err(CoreError::DimensionMismatch { .. })
    ));
    let mut sgd2_solver = Sgd2Solver::<Layout2d<f64>>::new(&g, Sgd2Config::default()).unwrap();
    sgd2_solver.reset_schedule(sched_b, 1).unwrap();
    assert_eq!(sgd2_solver.total_iterations(), Some(6));
    assert_eq!(sgd2_solver.iteration(), 0);
}
