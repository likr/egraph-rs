use egraph_core::{
    CoreError, CsrGraph, IterativeSolver, Layout2d, LayoutNd, ManifoldKind, ManifoldLayout2d,
    Scalar, SolverStatus,
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
    let cfg = StressMajorizationConfig {
        max_iterations: 40,
        epsilon: 1e-6,
        ..Default::default()
    };

    for kind in [
        ManifoldKind::HyperbolicPoincare,
        ManifoldKind::HyperbolicLorentz,
        ManifoldKind::Spherical,
        ManifoldKind::Torus2d,
    ] {
        // f64 bit-exact step == run and monotonic stress decrease.
        let mut s_run = StressMajorizationSolver::<f64>::from_graph(&grid, cfg).unwrap();
        let mut s_step = StressMajorizationSolver::<f64>::from_graph(&grid, cfg).unwrap();
        let mut l_run = init_manifold::<f64>(kind, n, 101);
        if kind == ManifoldKind::Torus2d {
            l_run.set_torus_extents(3.0, 2.0).unwrap();
        } else {
            l_run.set_scale(2.0).unwrap();
        }
        let mut l_step = l_run.clone();

        let stress_before = s_run.compute_stress_manifold(&l_run).unwrap();
        let st_run = s_run.run(&mut l_run).unwrap();
        assert!(st_run.is_finished());
        assert!(l_run.validate_finite().is_ok());
        let stress_after = s_run.compute_stress_manifold(&l_run).unwrap();
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
        assert_eq!(s_run.total_iterations(), 40);
        for i in 0..n {
            assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
            assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
        }

        // f32 bit-exact step == run.
        let mut s32_run = StressMajorizationSolver::<f32>::from_graph(&grid, cfg).unwrap();
        let mut s32_step = StressMajorizationSolver::<f32>::from_graph(&grid, cfg).unwrap();
        let mut l32_run = init_manifold::<f32>(kind, n, 101);
        if kind == ManifoldKind::Torus2d {
            l32_run.set_torus_extents(3.0, 2.0).unwrap();
        } else {
            l32_run.set_scale(2.0).unwrap();
        }
        let mut l32_step = l32_run.clone();

        let stress32_before = s32_run.compute_stress_manifold(&l32_run).unwrap();
        s32_run.run(&mut l32_run).unwrap();
        assert!(l32_run.validate_finite().is_ok());
        assert!(s32_run.compute_stress_manifold(&l32_run).unwrap() < stress32_before * 0.75);

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
fn test_stress_majorization_manifold_edge_cases_and_layout_kind_lock() {
    let cfg = StressMajorizationConfig::default();

    // n=0, n=1, edgeless, disconnected, coincident.
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let mut s0 = StressMajorizationSolver::<f64>::from_graph(&empty, cfg).unwrap();
    let mut l0 = ManifoldLayout2d::<f64>::zeros(ManifoldKind::HyperbolicLorentz, 0);
    assert_eq!(s0.run(&mut l0).unwrap(), SolverStatus::Converged);
    assert_eq!(s0.compute_stress_manifold(&l0).unwrap(), 0.0);

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let mut s1 = StressMajorizationSolver::<f64>::from_graph(&single, cfg).unwrap();
    let mut l1 = ManifoldLayout2d::from_xy(ManifoldKind::Spherical, vec![0.5], vec![1.0]).unwrap();
    assert_eq!(s1.run(&mut l1).unwrap(), SolverStatus::Converged);
    assert_eq!(l1.x(), &[0.5]);
    assert_eq!(l1.y(), &[1.0]);

    let edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    let mut se = StressMajorizationSolver::<f64>::from_graph(&edgeless, cfg).unwrap();
    let mut le = init_manifold::<f64>(ManifoldKind::Torus2d, 4, 7);
    let le_orig = le.clone();
    assert_eq!(se.run(&mut le).unwrap(), SolverStatus::Converged);
    assert_eq!(le, le_orig);

    let disc = CsrGraph::<f64>::from_edges(5, &[(0, 1), (2, 3)], None, false, false).unwrap();
    let mut s_disc = StressMajorizationSolver::<f64>::from_graph(&disc, cfg).unwrap();
    let mut l_disc = init_manifold::<f64>(ManifoldKind::HyperbolicPoincare, 5, 19);
    let iso_x = l_disc.x()[4];
    let iso_y = l_disc.y()[4];
    assert!(s_disc.run(&mut l_disc).unwrap().is_finished());
    assert!(l_disc.validate_finite().is_ok());
    assert_eq!(l_disc.x()[4], iso_x);
    assert_eq!(l_disc.y()[4], iso_y);

    let g3 = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, false, false).unwrap();
    let mut s_coin = StressMajorizationSolver::<f64>::from_graph(&g3, cfg).unwrap();
    let mut l_coin = ManifoldLayout2d::<f64>::zeros(ManifoldKind::HyperbolicPoincare, 3);
    assert!(s_coin.run(&mut l_coin).unwrap().is_finished());
    assert!(l_coin.validate_finite().is_ok());

    // Node count mismatch and non-finite rejection.
    let grid = grid_graph(2, 2);
    let mut s_err = StressMajorizationSolver::<f64>::from_graph(&grid, cfg).unwrap();
    let mut l_wrong = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Torus2d, 5);
    assert!(matches!(
        s_err.step(&mut l_wrong),
        Err(CoreError::DimensionMismatch { .. })
    ));

    let l_nan_raw = Layout2d::from_xy(vec![0.0, f64::NAN, 0.0, 0.0], vec![0.0; 4]).unwrap();
    assert!(l_nan_raw.validate_finite().is_err());

    // ActiveLayoutKind lock across 2d, nd, and manifold kinds.
    let mut l2 = Layout2d::<f64>::init(4, 1, 1.0).unwrap();
    let mut lnd = LayoutNd::<f64>::init(4, 3, 1, 1.0).unwrap();
    let mut l_poinc = init_manifold::<f64>(ManifoldKind::HyperbolicPoincare, 4, 2);
    let mut l_lor = init_manifold::<f64>(ManifoldKind::HyperbolicLorentz, 4, 2);

    let mut s_from_2d = StressMajorizationSolver::<f64>::from_graph(&grid, cfg).unwrap();
    s_from_2d.step(&mut l2).unwrap();
    assert!(matches!(
        s_from_2d.step(&mut lnd),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        s_from_2d.step(&mut l_poinc),
        Err(CoreError::InvalidInput(_))
    ));

    let mut s_from_nd = StressMajorizationSolver::<f64>::from_graph(&grid, cfg).unwrap();
    s_from_nd.step(&mut lnd).unwrap();
    assert!(matches!(
        s_from_nd.step(&mut l2),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        s_from_nd.step(&mut l_poinc),
        Err(CoreError::InvalidInput(_))
    ));

    let mut s_from_m = StressMajorizationSolver::<f64>::from_graph(&grid, cfg).unwrap();
    s_from_m.step(&mut l_poinc).unwrap();
    assert!(matches!(
        s_from_m.step(&mut l2),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        s_from_m.step(&mut lnd),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        s_from_m.step(&mut l_lor),
        Err(CoreError::InvalidInput(_))
    ));
}
