use egraph_core::{
    CoreError, CsrGraph, IterativeSolver, ManifoldKind, ManifoldLayout2d, Scalar, SolverStatus,
};

use super::{KamadaKawaiConfig, KamadaKawaiSolver};

fn grid_graph(rows: usize, cols: usize) -> CsrGraph<f64> {
    let n = rows * cols;
    let mut edges = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let u = (r * cols + c) as u32;
            if c + 1 < cols {
                edges.push((u, u + 1));
            }
            if r + 1 < rows {
                edges.push((u, u + cols as u32));
            }
        }
    }
    CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
}

fn init_manifold<S: Scalar>(kind: ManifoldKind, n: usize, seed: u64) -> ManifoldLayout2d<S> {
    let init_scale = match kind {
        ManifoldKind::HyperbolicPoincare => 0.45,
        ManifoldKind::HyperbolicLorentz => 0.8,
        ManifoldKind::Spherical | ManifoldKind::Torus2d => 1.0,
        _ => 0.5,
    };
    ManifoldLayout2d::<S>::init(kind, n, seed, init_scale).unwrap()
}

fn compute_kk_energy_manifold<S: Scalar>(
    solver: &KamadaKawaiSolver<ManifoldLayout2d<S>>,
    layout: &ManifoldLayout2d<S>,
) -> f64 {
    let n = layout.len();
    let l = solver.target_distances();
    let k = solver.spring_weights();
    let mut total = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let kij = k[i * n + j];
            if kij <= 0.0 {
                continue;
            }
            let d = layout.geodesic_distance(i, j).unwrap().to_f64();
            let diff = d - l[i * n + j];
            total += 0.5 * kij * diff * diff;
        }
    }
    total
}

#[test]
fn test_kamada_kawai_manifold_all_kinds_bit_exact_and_energy_reduction() {
    let grid = grid_graph(3, 3);
    let n = grid.num_nodes();
    let cfg = KamadaKawaiConfig::default();

    for kind in [
        ManifoldKind::HyperbolicPoincare,
        ManifoldKind::HyperbolicLorentz,
        ManifoldKind::Spherical,
        ManifoldKind::Torus2d,
    ] {
        // f64 bit-exact step == run and energy reduction.
        let mut s_run =
            KamadaKawaiSolver::<ManifoldLayout2d<f64>>::from_graph(&grid, cfg.clone()).unwrap();
        let mut s_step =
            KamadaKawaiSolver::<ManifoldLayout2d<f64>>::from_graph(&grid, cfg.clone()).unwrap();
        let mut l_run = init_manifold::<f64>(kind, n, 31);
        if kind == ManifoldKind::Torus2d {
            l_run.set_torus_extents(3.0, 2.0).unwrap();
        } else {
            l_run.set_scale(2.0).unwrap();
        }
        let mut l_step = l_run.clone();

        let e_before = compute_kk_energy_manifold(&s_run, &l_run);
        assert_eq!(s_run.total_iterations(), Some(100));
        assert!(s_run.run(&mut l_run).unwrap().is_finished());
        assert!(l_run.validate_finite().is_ok());
        let e_after = compute_kk_energy_manifold(&s_run, &l_run);
        assert!(
            e_after < e_before * 0.75,
            "{kind:?}: before={e_before}, after={e_after}"
        );

        let _ = s_step.run_steps(&mut l_step, 5).unwrap();
        while !s_step.status().is_finished() {
            s_step.step(&mut l_step).unwrap();
        }
        assert_eq!(s_run.status(), s_step.status());
        assert_eq!(s_run.iteration(), s_step.iteration());
        for i in 0..n {
            assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
            assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
        }

        // f32 bit-exact step == run.
        let mut s32_run =
            KamadaKawaiSolver::<ManifoldLayout2d<f32>>::from_graph(&grid, cfg.clone()).unwrap();
        let mut s32_step =
            KamadaKawaiSolver::<ManifoldLayout2d<f32>>::from_graph(&grid, cfg.clone()).unwrap();
        let mut l32_run = init_manifold::<f32>(kind, n, 31);
        if kind == ManifoldKind::Torus2d {
            l32_run.set_torus_extents(3.0, 2.0).unwrap();
        } else {
            l32_run.set_scale(2.0).unwrap();
        }
        let mut l32_step = l32_run.clone();
        s32_run.run(&mut l32_run).unwrap();
        assert!(l32_run.validate_finite().is_ok());
        while !s32_step.status().is_finished() {
            s32_step.step(&mut l32_step).unwrap();
        }
        assert_eq!(s32_run.iteration(), s32_step.iteration());
        for i in 0..n {
            assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
            assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
        }
    }
}

#[test]
fn test_kamada_kawai_manifold_indefinite_saddle_uses_steepest_descent_not_newton() {
    // Construct a negative-determinant Hessian saddle point on Torus2d (scale = 1.0, aspect = 1.0):
    // Vertex 0 at (0.5, 0.5) and vertex 1 at (0.6, 0.51) with target distance l_{01} = 0.4.
    // Then dx = -0.1, dy = -0.01, d ~ 0.1004988 < 0.4, so:
    //   k = 1 / 0.16 = 6.25
    //   hxx ~ k * (1 - 0.4 * 0.0001 / d^3) > 0
    //   hyy ~ k * (1 - 0.4 * 0.01 / d^3) < 0  =>  det = hxx * hyy - hxy^2 < -HESSIAN_DET_TOL
    //   dedx < 0, dedy > 0.
    // An unguarded Newton step would invert the negative curvature in y (moving +y toward 0.51).
    // The positive-definite guard must fall back to steepest descent (moving -y, away from 0.51)
    // with exact step length min(0.1 / ||g||, 0.25) * ||g|| = 0.1 and exit after one fallback step.
    let dists = [0.0, 0.4, 0.4, 0.0];
    let cfg = KamadaKawaiConfig::new()
        .with_eps(1e-4)
        .with_max_inner_iters(20)
        .with_max_iterations(Some(5));
    let mut solver =
        KamadaKawaiSolver::<ManifoldLayout2d<f64>>::from_distance_matrix(2, &dists, cfg).unwrap();
    let mut layout =
        ManifoldLayout2d::from_xy(ManifoldKind::Torus2d, vec![0.5, 0.6], vec![0.5, 0.51]).unwrap();

    let x0_before = layout.x()[0];
    let y0_before = layout.y()[0];
    solver.step(&mut layout).unwrap();

    // Vertex 0 is moved (symmetric gradients, first vertex wins tie).
    let moved_x = layout.x()[0] - x0_before;
    let moved_y = layout.y()[0] - y0_before;
    assert_eq!(layout.x()[1], 0.6);
    assert_eq!(layout.y()[1], 0.51);

    // Steepest descent moves opposite the gradient: -dedx < 0 and -dedy < 0.
    assert!(
        moved_x < 0.0,
        "expected negative x displacement, got {moved_x}"
    );
    assert!(
        moved_y < 0.0,
        "saddle point must fall back to steepest descent (moved_y < 0), got {moved_y}"
    );
    // Exact fallback step magnitude is 0.1 (whereas Newton would move ~0.3 in x and exit at d=0.4).
    let step_mag = moved_x.hypot(moved_y);
    assert!(
        (step_mag - 0.1).abs() < 1e-12,
        "expected single steepest-descent step of magnitude 0.1, got {step_mag}"
    );
}

#[test]
fn test_kamada_kawai_manifold_singular_fallback_edge_cases_and_lock() {
    let cfg = KamadaKawaiConfig::default();

    // Collinear hyperbolic det == 0 singular fallback terminates cleanly.
    let phi_inv = (5.0_f64.sqrt() - 1.0) * 0.5;
    let dists = [
        0.0, 1.0, phi_inv, //
        1.0, 0.0, 1.0, //
        phi_inv, 1.0, 0.0,
    ];
    let mut s_col = KamadaKawaiSolver::<ManifoldLayout2d<f64>>::from_distance_matrix(
        3,
        &dists,
        KamadaKawaiConfig::new()
            .with_eps(1e-3)
            .with_max_inner_iters(50)
            .with_max_iterations(Some(60)),
    )
    .unwrap();
    let mut l_col = ManifoldLayout2d::from_xy(
        ManifoldKind::HyperbolicLorentz,
        vec![0.0, 0.5, 1.0],
        vec![0.0, 0.0, 0.0],
    )
    .unwrap();
    let e_before = compute_kk_energy_manifold(&s_col, &l_col);
    assert!(s_col.run(&mut l_col).unwrap().is_finished());
    assert!(l_col.validate_finite().is_ok());
    assert!(compute_kk_energy_manifold(&s_col, &l_col) < e_before);

    // n=0, n=1, edgeless, disconnected, coincident.
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let mut s0 =
        KamadaKawaiSolver::<ManifoldLayout2d<f64>>::from_graph(&empty, cfg.clone()).unwrap();
    let mut l0 = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Spherical, 0);
    assert_eq!(s0.run(&mut l0).unwrap(), SolverStatus::Converged);

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let mut s1 =
        KamadaKawaiSolver::<ManifoldLayout2d<f64>>::from_graph(&single, cfg.clone()).unwrap();
    let mut l1 = ManifoldLayout2d::from_xy(ManifoldKind::Torus2d, vec![0.2], vec![0.7]).unwrap();
    assert_eq!(s1.run(&mut l1).unwrap(), SolverStatus::Converged);

    let edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    let mut se =
        KamadaKawaiSolver::<ManifoldLayout2d<f64>>::from_graph(&edgeless, cfg.clone()).unwrap();
    let mut le = init_manifold::<f64>(ManifoldKind::HyperbolicPoincare, 4, 5);
    assert_eq!(se.run(&mut le).unwrap(), SolverStatus::Converged);

    let disc = CsrGraph::<f64>::from_edges(4, &[(0, 1), (2, 3)], None, false, false).unwrap();
    let mut s_disc =
        KamadaKawaiSolver::<ManifoldLayout2d<f64>>::from_graph(&disc, cfg.clone()).unwrap();
    let mut l_disc = init_manifold::<f64>(ManifoldKind::Spherical, 4, 13);
    assert!(s_disc.run(&mut l_disc).unwrap().is_finished());
    assert!(l_disc.validate_finite().is_ok());

    let g3 = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, false, false).unwrap();
    let mut s_coin =
        KamadaKawaiSolver::<ManifoldLayout2d<f64>>::from_graph(&g3, cfg.clone()).unwrap();
    let mut l_coin = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Torus2d, 3);
    assert!(s_coin.run(&mut l_coin).unwrap().is_finished());
    assert!(l_coin.validate_finite().is_ok());

    // Node mismatch and ManifoldKind switching rejections (including run_steps(0)).
    let grid = grid_graph(2, 2);
    let mut s_err =
        KamadaKawaiSolver::<ManifoldLayout2d<f64>>::from_graph(&grid, cfg.clone()).unwrap();
    let mut l_wrong = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Spherical, 3);
    assert!(matches!(
        s_err.step(&mut l_wrong),
        Err(CoreError::DimensionMismatch { .. })
    ));

    let mut l_sph = init_manifold::<f64>(ManifoldKind::Spherical, 4, 2);
    let mut l_tor = init_manifold::<f64>(ManifoldKind::Torus2d, 4, 2);
    let mut s_from_m = KamadaKawaiSolver::<ManifoldLayout2d<f64>>::from_graph(&grid, cfg).unwrap();
    s_from_m.step(&mut l_sph).unwrap();
    assert!(matches!(
        s_from_m.run_steps(&mut l_tor, 0),
        Err(CoreError::InvalidInput(_))
    ));
}
