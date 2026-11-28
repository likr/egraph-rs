use egraph_core::{
    CsrGraph, IterativeSolver, Layout2d, Layout3d, LayoutNd, ManifoldKind, ManifoldLayout2d,
    MetricSpace, SolverStatus, cartesian_to_spherical, spherical_to_cartesian,
};

use super::super::sgd::{
    OrtmannSparseStressConfig, Sgd2Config, Sgd2Solver, SgdScheduleConfig, SgdScheduleKind,
    SgdSolver,
};
use super::test_support::{circle_layout, euclidean_apsp_stress, grid_graph, manifold_pair_stress};

fn rotate_2d_points(xs: &[f64], ys: &[f64], theta: f64) -> (Vec<f64>, Vec<f64>) {
    let (s, c) = theta.sin_cos();
    let mut rx = Vec::with_capacity(xs.len());
    let mut ry = Vec::with_capacity(ys.len());
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        rx.push(c * x - s * y);
        ry.push(s * x + c * y);
    }
    (rx, ry)
}

fn rotate_sphere_points_x(lons: &[f64], colats: &[f64], alpha: f64) -> (Vec<f64>, Vec<f64>) {
    let (sa, ca) = alpha.sin_cos();
    let mut out_lon = Vec::with_capacity(lons.len());
    let mut out_colat = Vec::with_capacity(colats.len());
    for (&lon, &colat) in lons.iter().zip(colats.iter()) {
        let [x, y, z] = spherical_to_cartesian(lon, colat);
        let ry = ca * y - sa * z;
        let rz = sa * y + ca * z;
        let (nl, nc) = cartesian_to_spherical(x, ry, rz);
        out_lon.push(nl);
        out_colat.push(nc);
    }
    (out_lon, out_colat)
}

#[test]
fn test_sgd2_and_squared_schedule_convergence_and_bit_exact() {
    let g = grid_graph(3, 4);

    let sq_sched = SgdScheduleConfig::new(11)
        .with_eps(0.05)
        .with_kind(SgdScheduleKind::Squared)
        .with_step_scale(1.0);
    let mut sq_solver = SgdSolver::<Layout2d<f64>>::full(&g, sq_sched, 42).unwrap();
    let mut etas = Vec::new();
    let mut sq_layout = circle_layout::<f64>(g.num_nodes(), 1.0);
    for _ in 0..11 {
        etas.push(sq_solver.current_eta());
        sq_solver.step(&mut sq_layout).unwrap();
    }
    assert_eq!(sq_solver.current_eta(), 0.0);
    assert!((etas[0] - sq_solver.eta_max).abs() < 1e-12);
    assert!((etas[10] - sq_solver.eta_min).abs() < 1e-12);
    for win in etas.windows(2) {
        assert!(win[0] >= win[1]);
    }

    let cfg = Sgd2Config::new()
        .with_iterations(40)
        .with_eps(0.01)
        .with_step_scale(1.0)
        .with_batch_size(7)
        .with_damping(1e-4)
        .with_seed(42);

    // Bit-exact step() == run() for f64 and f32.
    let mut s64_run = Sgd2Solver::<Layout2d<f64>>::new(&g, cfg.clone()).unwrap();
    let mut s64_step = Sgd2Solver::<Layout2d<f64>>::new(&g, cfg.clone()).unwrap();
    let mut l64_run = circle_layout::<f64>(g.num_nodes(), 0.25);
    let mut l64_step = l64_run.clone();
    let stress_before = euclidean_apsp_stress(&g, &l64_run);
    s64_run.run(&mut l64_run).unwrap();
    s64_step.run_steps(&mut l64_step, 11).unwrap();
    while !s64_step.status().is_finished() {
        s64_step.step(&mut l64_step).unwrap();
    }
    for i in 0..g.num_nodes() {
        assert_eq!(l64_run.x()[i].to_bits(), l64_step.x()[i].to_bits());
        assert_eq!(l64_run.y()[i].to_bits(), l64_step.y()[i].to_bits());
    }
    let stress_after = euclidean_apsp_stress(&g, &l64_run);
    assert!(
        stress_after < stress_before * 0.2,
        "Sgd2 did not reduce stress: {stress_before} -> {stress_after}"
    );

    let mut s32_run = Sgd2Solver::<Layout2d<f32>>::new(&g, cfg.clone()).unwrap();
    let mut s32_step = Sgd2Solver::<Layout2d<f32>>::new(&g, cfg.clone()).unwrap();
    let mut l32_run = circle_layout::<f32>(g.num_nodes(), 0.25);
    let mut l32_step = l32_run.clone();
    s32_run.run(&mut l32_run).unwrap();
    s32_step.run_steps(&mut l32_step, 13).unwrap();
    s32_step.run(&mut l32_step).unwrap();
    for i in 0..g.num_nodes() {
        assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
        assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
    }

    // 3D and ND Sgd2Solver support.
    let mut s_3d = Sgd2Solver::<Layout3d<f64>>::new(&g, cfg.clone()).unwrap();
    let mut l_3d = Layout3d::<f64>::init(g.num_nodes(), 7, 1.0).unwrap();
    assert_eq!(s_3d.run(&mut l_3d).unwrap(), SolverStatus::Completed);
    assert!(l_3d.is_all_finite());

    let mut s_nd = Sgd2Solver::<LayoutNd<f64>>::new(&g, cfg.clone()).unwrap();
    let mut l_nd = LayoutNd::<f64>::init(g.num_nodes(), 4, 7, 1.0).unwrap();
    assert_eq!(s_nd.run(&mut l_nd).unwrap(), SolverStatus::Completed);
    assert!(l_nd.is_all_finite());

    // Large-n (> 256) sparse path uses bounded O(N) pairs instead of O(N^2) APSP.
    let big_grid = grid_graph(18, 18);
    let mut big_s2 = Sgd2Solver::<Layout2d<f64>>::new(&big_grid, Sgd2Config::default()).unwrap();
    assert!(big_s2.num_springs() < 324 * 20);
    let mut big_l = circle_layout::<f64>(big_grid.num_nodes(), 1.0);
    big_s2.run(&mut big_l).unwrap();
    assert!(big_l.is_all_finite());

    // Weighted triangle shortest-path check (0-1-2 = 3.0 vs 0-2 = 10.0).
    let wg = CsrGraph::<f64>::from_edges(
        3,
        &[(0, 1), (1, 2), (0, 2)],
        Some(&[1.0, 2.0, 10.0]),
        false,
        false,
    )
    .unwrap();
    let mut w_s2 = Sgd2Solver::<Layout2d<f64>>::new(&wg, Sgd2Config::default()).unwrap();
    let mut w_l = circle_layout::<f64>(3, 2.0);
    w_s2.run(&mut w_l).unwrap();
    let d02 = (w_l.x()[0] - w_l.x()[2]).hypot(w_l.y()[0] - w_l.y()[2]);
    assert!((d02 - 3.0).abs() < 0.25, "expected dist ~3.0, got {d02}");

    // Validation & edge-case guards.
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    assert_eq!(
        Sgd2Solver::<Layout2d<f64>>::new(&empty, cfg.clone())
            .unwrap()
            .run(&mut Layout2d::zeros(0))
            .unwrap(),
        SolverStatus::Completed
    );
    assert_eq!(
        Sgd2Solver::<Layout2d<f64>>::new(&single, cfg.clone())
            .unwrap()
            .run(&mut Layout2d::zeros(1))
            .unwrap(),
        SolverStatus::Completed
    );
    for bad in [
        cfg.clone().with_eps(0.0),
        cfg.clone().with_eps(f64::NAN),
        cfg.clone().with_step_scale(-1.0),
        cfg.clone().with_batch_size(0),
        cfg.with_damping(0.0),
    ] {
        assert!(Sgd2Solver::<Layout2d<f64>>::new(&g, bad).is_err());
    }
}

#[test]
fn test_sgd2_manifold_so2_tangent_frame_rotation_invariance() {
    // Asymmetric 5-node graph so each vertex sees anisotropic pair directions.
    let g = CsrGraph::<f64>::from_edges(
        5,
        &[(0, 1), (0, 2), (0, 3), (1, 4), (2, 4)],
        None,
        false,
        false,
    )
    .unwrap();
    let cfg = Sgd2Config::new()
        .with_iterations(12)
        .with_batch_size(64)
        .with_damping(1e-2)
        .with_seed(17);
    let theta = 0.61_f64; // non-axis-aligned rotation (~35 degrees)

    // 1. HyperbolicPoincare and HyperbolicLorentz SO(2) rotational invariance.
    let base_x = [0.05, 0.38, -0.14, 0.22, -0.31];
    let base_y = [0.02, 0.11, 0.35, -0.27, -0.09];
    let (rot_x, rot_y) = rotate_2d_points(&base_x, &base_y, theta);

    for kind in [
        ManifoldKind::HyperbolicPoincare,
        ManifoldKind::HyperbolicLorentz,
    ] {
        let mut s_base = Sgd2Solver::<ManifoldLayout2d<f64>>::new(&g, cfg.clone()).unwrap();
        let mut s_rot = Sgd2Solver::<ManifoldLayout2d<f64>>::new(&g, cfg.clone()).unwrap();
        let mut l_base = ManifoldLayout2d::from_xy(kind, base_x.to_vec(), base_y.to_vec()).unwrap();
        let mut l_rot = ManifoldLayout2d::from_xy(kind, rot_x.clone(), rot_y.clone()).unwrap();

        s_base.run(&mut l_base).unwrap();
        s_rot.run(&mut l_rot).unwrap();

        let (expected_rx, expected_ry) = rotate_2d_points(l_base.x(), l_base.y(), theta);
        for i in 0..5 {
            assert!(
                (l_rot.x()[i] - expected_rx[i]).abs() < 1e-11
                    && (l_rot.y()[i] - expected_ry[i]).abs() < 1e-11,
                "{kind:?} vertex {i} broke SO(2) invariance: got ({}, {}), expected ({}, {})",
                l_rot.x()[i],
                l_rot.y()[i],
                expected_rx[i],
                expected_ry[i]
            );
            for j in (i + 1)..5 {
                let d_base = l_base.distance(i, j);
                let d_rot = l_rot.distance(i, j);
                assert!(
                    (d_base - d_rot).abs() < 1e-11,
                    "{kind:?} pairwise distance ({i},{j}) differed: {d_base} vs {d_rot}"
                );
            }
        }
    }

    // 2. Spherical SO(3) isometry rotating tangent frames (e_lon, e_colat) non-trivially.
    let base_lon = [0.2, 1.1, 2.3, 3.8, 5.1];
    let base_colat = [0.7, 1.2, 0.9, 1.6, 1.4];
    let (rot_lon, rot_colat) = rotate_sphere_points_x(&base_lon, &base_colat, 0.73);

    let mut ss_base = Sgd2Solver::<ManifoldLayout2d<f64>>::new(&g, cfg.clone()).unwrap();
    let mut ss_rot = Sgd2Solver::<ManifoldLayout2d<f64>>::new(&g, cfg).unwrap();
    let mut ls_base = ManifoldLayout2d::from_xy(
        ManifoldKind::Spherical,
        base_lon.to_vec(),
        base_colat.to_vec(),
    )
    .unwrap();
    let mut ls_rot =
        ManifoldLayout2d::from_xy(ManifoldKind::Spherical, rot_lon, rot_colat).unwrap();

    ss_base.run(&mut ls_base).unwrap();
    ss_rot.run(&mut ls_rot).unwrap();

    let (exp_lon, exp_colat) = rotate_sphere_points_x(ls_base.x(), ls_base.y(), 0.73);
    for i in 0..5 {
        let p_rot = spherical_to_cartesian(ls_rot.x()[i], ls_rot.y()[i]);
        let p_exp = spherical_to_cartesian(exp_lon[i], exp_colat[i]);
        let chord = ((p_rot[0] - p_exp[0]).powi(2)
            + (p_rot[1] - p_exp[1]).powi(2)
            + (p_rot[2] - p_exp[2]).powi(2))
        .sqrt();
        assert!(
            chord < 1e-11,
            "Spherical vertex {i} broke tangent-frame SO(2) invariance: chord={chord}"
        );
        for j in (i + 1)..5 {
            let d_base = ls_base.distance(i, j);
            let d_rot = ls_rot.distance(i, j);
            assert!(
                (d_base - d_rot).abs() < 1e-11,
                "Spherical pairwise distance ({i},{j}) differed: {d_base} vs {d_rot}"
            );
        }
    }

    // Also verify Sgd2Solver reduces manifold geodesic stress on Poincare disk.
    let tree =
        CsrGraph::<f64>::from_edges(5, &[(0, 1), (0, 2), (1, 3), (1, 4)], None, false, false)
            .unwrap();
    let mut ml = ManifoldLayout2d::from_xy(
        ManifoldKind::HyperbolicPoincare,
        base_x.to_vec(),
        base_y.to_vec(),
    )
    .unwrap();
    let pairs = vec![
        (0, 1, 1.0),
        (0, 2, 1.0),
        (1, 3, 1.0),
        (1, 4, 1.0),
        (2, 3, 3.0),
    ];
    let before = manifold_pair_stress(&ml, &pairs);
    let mut s_stress = Sgd2Solver::<ManifoldLayout2d<f64>>::new(
        &tree,
        Sgd2Config::new().with_iterations(40).with_seed(42),
    )
    .unwrap();
    s_stress.run(&mut ml).unwrap();
    let after = manifold_pair_stress(&ml, &pairs);
    assert!(
        after < before * 0.5,
        "Sgd2 manifold stress {before} -> {after}"
    );
}

#[test]
fn test_ortmann_config_validations() {
    assert!(OrtmannSparseStressConfig::default().validate().is_ok());
    assert!(
        OrtmannSparseStressConfig::default()
            .with_num_pivots(0)
            .validate()
            .is_err()
    );
    for bad in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        assert!(
            OrtmannSparseStressConfig::default()
                .with_eps(bad)
                .validate()
                .is_err()
        );
        assert!(
            OrtmannSparseStressConfig::default()
                .with_eta_max(Some(bad))
                .validate()
                .is_err()
        );
        assert!(
            OrtmannSparseStressConfig::default()
                .with_eta_min(Some(bad))
                .validate()
                .is_err()
        );
    }
    assert!(
        OrtmannSparseStressConfig::default()
            .with_eta_min(Some(2.0))
            .with_eta_max(Some(1.0))
            .validate()
            .is_err()
    );
}
