use egraph_core::{
    CoreError, CsrGraph, IterativeSolver, Layout2d, ManifoldKind, ManifoldLayout2d, Scalar,
    SolverStatus,
};

use super::{StressMajorizationConfig, StressMajorizationSolver};

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

#[test]
fn test_stress_majorization_manifold_all_kinds_bit_exact_and_stress_reduction() {
    let grid = grid_graph(3, 3);
    let n = grid.num_nodes();
    let cfg = StressMajorizationConfig::new()
        .with_max_iterations(40)
        .with_epsilon(1e-6);

    for kind in [
        ManifoldKind::HyperbolicPoincare,
        ManifoldKind::HyperbolicLorentz,
        ManifoldKind::Spherical,
        ManifoldKind::Torus2d,
    ] {
        // f64 bit-exact step == run and monotonic stress decrease.
        let mut s_run = StressMajorizationSolver::<ManifoldLayout2d<f64>>::from_graph_riemannian(
            &grid,
            cfg.clone(),
        )
        .unwrap();
        let mut s_step = StressMajorizationSolver::<ManifoldLayout2d<f64>>::from_graph_riemannian(
            &grid,
            cfg.clone(),
        )
        .unwrap();
        let mut l_run = init_manifold::<f64>(kind, n, 101);
        if kind == ManifoldKind::Torus2d {
            l_run.set_torus_extents(3.0, 2.0).unwrap();
        } else {
            l_run.set_scale(2.0).unwrap();
        }
        let mut l_step = l_run.clone();

        let stress_before = s_run.compute_stress(&l_run).unwrap();
        let st_run = s_run.run(&mut l_run).unwrap();
        assert!(st_run.is_finished());
        assert!(l_run.validate_finite().is_ok());
        let stress_after = s_run.compute_stress(&l_run).unwrap();
        assert!(
            stress_after < stress_before * 0.75,
            "{kind:?}: before={stress_before}, after={stress_after}"
        );

        assert_eq!(
            s_step.run_steps(&mut l_step, 4).unwrap(),
            SolverStatus::Running
        );
        while !s_step.status().is_finished() {
            s_step.step(&mut l_step).unwrap();
        }
        assert_eq!(s_run.status(), s_step.status());
        assert_eq!(s_run.iteration(), s_step.iteration());
        assert_eq!(s_run.total_iterations(), Some(40));
        for i in 0..n {
            assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
            assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
        }

        // f32 bit-exact step == run.
        let mut s32_run = StressMajorizationSolver::<ManifoldLayout2d<f32>>::from_graph_riemannian(
            &grid,
            cfg.clone(),
        )
        .unwrap();
        let mut s32_step =
            StressMajorizationSolver::<ManifoldLayout2d<f32>>::from_graph_riemannian(
                &grid,
                cfg.clone(),
            )
            .unwrap();
        let mut l32_run = init_manifold::<f32>(kind, n, 101);
        if kind == ManifoldKind::Torus2d {
            l32_run.set_torus_extents(3.0, 2.0).unwrap();
        } else {
            l32_run.set_scale(2.0).unwrap();
        }
        let mut l32_step = l32_run.clone();

        let stress32_before = s32_run.compute_stress(&l32_run).unwrap();
        s32_run.run(&mut l32_run).unwrap();
        assert!(l32_run.validate_finite().is_ok());
        assert!(s32_run.compute_stress(&l32_run).unwrap() < stress32_before * 0.75);

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
fn test_stress_majorization_manifold_edge_cases_and_lock() {
    let cfg = StressMajorizationConfig::default();

    // n=0, n=1, edgeless, disconnected, coincident.
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let mut s0 = StressMajorizationSolver::<ManifoldLayout2d<f64>>::from_graph_riemannian(
        &empty,
        cfg.clone(),
    )
    .unwrap();
    let mut l0 = ManifoldLayout2d::<f64>::zeros(ManifoldKind::HyperbolicLorentz, 0);
    assert_eq!(s0.run(&mut l0).unwrap(), SolverStatus::Converged);
    assert_eq!(s0.compute_stress(&l0).unwrap(), 0.0);

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let mut s1 = StressMajorizationSolver::<ManifoldLayout2d<f64>>::from_graph_riemannian(
        &single,
        cfg.clone(),
    )
    .unwrap();
    let mut l1 = ManifoldLayout2d::from_xy(ManifoldKind::Spherical, vec![0.5], vec![1.0]).unwrap();
    assert_eq!(s1.run(&mut l1).unwrap(), SolverStatus::Converged);
    assert_eq!(l1.x(), &[0.5]);
    assert_eq!(l1.y(), &[1.0]);

    let edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    let mut se = StressMajorizationSolver::<ManifoldLayout2d<f64>>::from_graph_riemannian(
        &edgeless,
        cfg.clone(),
    )
    .unwrap();
    let mut le = init_manifold::<f64>(ManifoldKind::Torus2d, 4, 7);
    let le_orig = le.clone();
    assert_eq!(se.run(&mut le).unwrap(), SolverStatus::Converged);
    assert_eq!(le, le_orig);

    let disc = CsrGraph::<f64>::from_edges(5, &[(0, 1), (2, 3)], None, false, false).unwrap();
    let mut s_disc = StressMajorizationSolver::<ManifoldLayout2d<f64>>::from_graph_riemannian(
        &disc,
        cfg.clone(),
    )
    .unwrap();
    let mut l_disc = init_manifold::<f64>(ManifoldKind::HyperbolicPoincare, 5, 19);
    let iso_x = l_disc.x()[4];
    let iso_y = l_disc.y()[4];
    assert!(s_disc.run(&mut l_disc).unwrap().is_finished());
    assert!(l_disc.validate_finite().is_ok());
    assert_eq!(l_disc.x()[4], iso_x);
    assert_eq!(l_disc.y()[4], iso_y);

    // Distance-matrix Riemannian constructor + unweighted CsrGraph<()> Riemannian constructor.
    let dists = [0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0];
    let mut s_dm =
        StressMajorizationSolver::<ManifoldLayout2d<f64>>::from_distance_matrix_riemannian(
            3,
            &dists,
            cfg.clone(),
        )
        .unwrap();
    let mut l_coin = ManifoldLayout2d::<f64>::zeros(ManifoldKind::HyperbolicPoincare, 3);
    assert!(s_dm.run(&mut l_coin).unwrap().is_finished());
    assert!(l_coin.validate_finite().is_ok());

    let g3_unit =
        CsrGraph::<()>::from_unweighted_edges(3, &[(0, 1), (1, 2)], false, false).unwrap();
    let mut s_unit = StressMajorizationSolver::<Layout2d<f64>>::from_unweighted_graph_riemannian(
        &g3_unit,
        cfg.clone(),
    )
    .unwrap();
    let mut l_euc_riem = Layout2d::<f64>::init(3, 5, 2.0).unwrap();
    let e_before = s_unit.compute_stress(&l_euc_riem).unwrap();
    assert!(s_unit.run(&mut l_euc_riem).unwrap().is_finished());
    assert!(s_unit.compute_stress(&l_euc_riem).unwrap() <= e_before);

    // Node count mismatch and ManifoldKind lock (including run_steps(0)).
    let grid = grid_graph(2, 2);
    let mut s_err = StressMajorizationSolver::<ManifoldLayout2d<f64>>::from_graph_riemannian(
        &grid,
        cfg.clone(),
    )
    .unwrap();
    let mut l_wrong = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Torus2d, 5);
    assert!(matches!(
        s_err.run_steps(&mut l_wrong, 0),
        Err(CoreError::DimensionMismatch { .. })
    ));

    let mut l_poinc = init_manifold::<f64>(ManifoldKind::HyperbolicPoincare, 4, 2);
    let mut l_lor = init_manifold::<f64>(ManifoldKind::HyperbolicLorentz, 4, 2);
    let mut s_from_m =
        StressMajorizationSolver::<ManifoldLayout2d<f64>>::from_graph_riemannian(&grid, cfg)
            .unwrap();
    s_from_m.step(&mut l_poinc).unwrap();
    assert!(matches!(
        s_from_m.run_steps(&mut l_lor, 0),
        Err(CoreError::InvalidInput(_))
    ));
}
