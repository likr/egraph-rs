use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Layout3d,
    LayoutStorage, MetricSpace, RiemannianLayout, Scalar, SolverStatus,
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

fn seeded_layout<S: Scalar>(n: usize, scale: f64, seed: u64) -> Layout2d<S> {
    let mut rng = DeterministicRng::from_seed_u64(seed);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        x.push(S::from_f64((rng.next_f64() * 2.0 - 1.0) * scale));
        y.push(S::from_f64((rng.next_f64() * 2.0 - 1.0) * scale));
    }
    Layout2d::from_xy(x, y).unwrap()
}

#[test]
fn test_kamada_kawai_step_vs_run_bit_exact_f64_f32_and_3d() {
    let grid = grid_graph(3, 3);
    let cfg = KamadaKawaiConfig::default();

    // f64 bit-exact parity on Layout2d
    let mut s_run = KamadaKawaiSolver::<Layout2d<f64>>::from_graph(&grid, cfg.clone()).unwrap();
    let mut s_step = KamadaKawaiSolver::<Layout2d<f64>>::from_graph(&grid, cfg.clone()).unwrap();
    let mut l_run = seeded_layout::<f64>(9, 1.5, 21);
    let mut l_step = l_run.clone();

    assert_eq!(s_run.total_iterations(), Some(100));
    let st_run = s_run.run(&mut l_run).unwrap();
    assert!(st_run.is_finished());
    assert!(l_run.is_all_finite());

    let _ = s_step.run_steps(&mut l_step, 5).unwrap();
    while !s_step.status().is_finished() {
        s_step.step(&mut l_step).unwrap();
    }
    assert_eq!(s_run.status(), s_step.status());
    assert_eq!(s_run.iteration(), s_step.iteration());
    for i in 0..9 {
        assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
        assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
    }

    // f32 bit-exact parity on Layout2d
    let mut s32_run = KamadaKawaiSolver::<Layout2d<f32>>::from_graph(&grid, cfg.clone()).unwrap();
    let mut s32_step = KamadaKawaiSolver::<Layout2d<f32>>::from_graph(&grid, cfg.clone()).unwrap();
    let mut l32_run = seeded_layout::<f32>(9, 1.5, 21);
    let mut l32_step = l32_run.clone();
    s32_run.run(&mut l32_run).unwrap();
    while !s32_step.status().is_finished() {
        s32_step.step(&mut l32_step).unwrap();
    }
    assert_eq!(s32_run.iteration(), s32_step.iteration());
    for i in 0..9 {
        assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
        assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
    }

    // Layout3d bit-exact parity and unweighted CsrGraph<()> support
    let unweighted =
        CsrGraph::<()>::from_unweighted_csr(grid.indptr(), grid.indices(), false, false).unwrap();
    let mut s3d_run =
        KamadaKawaiSolver::<Layout3d<f64>>::from_unweighted_graph(&unweighted, cfg.clone())
            .unwrap();
    let mut s3d_step =
        KamadaKawaiSolver::<Layout3d<f64>>::from_unweighted_graph(&unweighted, cfg).unwrap();
    let mut l3d_run = Layout3d::<f64>::init(9, 21, 3.0).unwrap();
    let mut l3d_step = l3d_run.clone();
    assert!(s3d_run.run(&mut l3d_run).unwrap().is_finished());
    while !s3d_step.status().is_finished() {
        s3d_step.step(&mut l3d_step).unwrap();
    }
    assert_eq!(s3d_run.iteration(), s3d_step.iteration());
    for i in 0..9 {
        assert_eq!(l3d_run.x()[i].to_bits(), l3d_step.x()[i].to_bits());
        assert_eq!(l3d_run.y()[i].to_bits(), l3d_step.y()[i].to_bits());
        assert_eq!(l3d_run.z()[i].to_bits(), l3d_step.z()[i].to_bits());
    }
}

#[test]
fn test_kamada_kawai_degenerate_collinear_det_zero_no_infinite_loop() {
    // Construct exact Hessian singularity (det == 0) at node 0 on a collinear x-axis layout:
    // x = [0.0, 0.5, 1.0], y = [0.0, 0.0, 0.0], with l_{01} = 1.0 and l_{02} = (sqrt(5) - 1) / 2.
    let phi_inv = (5.0_f64.sqrt() - 1.0) * 0.5;
    let dists = [
        0.0, 1.0, phi_inv, //
        1.0, 0.0, 1.0, //
        phi_inv, 1.0, 0.0,
    ];
    let cfg = KamadaKawaiConfig::new()
        .with_eps(1e-3)
        .with_max_inner_iters(50)
        .with_max_iterations(Some(60));
    let mut solver =
        KamadaKawaiSolver::<Layout2d<f64>>::from_distance_matrix(3, &dists, cfg).unwrap();
    let mut layout = Layout2d::from_xy(vec![0.0, 0.5, 1.0], vec![0.0, 0.0, 0.0]).unwrap();
    let status = solver.run(&mut layout).unwrap();
    assert!(status.is_finished());
    assert!(solver.iteration() <= 60);
    assert!(layout.is_all_finite());

    let g15 = grid_graph(3, 5);
    let s15 = KamadaKawaiSolver::<Layout2d<f64>>::new(&g15, KamadaKawaiConfig::default()).unwrap();
    assert_eq!(s15.total_iterations(), Some(150));
}

#[test]
fn test_kamada_kawai_edge_cases_disconnected_zero_steps_and_errors() {
    let cfg = KamadaKawaiConfig::default();

    // Empty and singleton graphs
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let mut s0 = KamadaKawaiSolver::<Layout2d<f64>>::from_graph(&empty, cfg.clone()).unwrap();
    let mut l0 = Layout2d::<f64>::zeros(0);
    assert_eq!(s0.run(&mut l0).unwrap(), SolverStatus::Converged);

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let mut s1 = KamadaKawaiSolver::<Layout2d<f64>>::from_graph(&single, cfg.clone()).unwrap();
    let mut l1 = Layout2d::from_xy(vec![1.0], vec![2.0]).unwrap();
    assert_eq!(s1.run(&mut l1).unwrap(), SolverStatus::Converged);

    // Disconnected graph remains finite and preserves target distances per component
    let disc = CsrGraph::<f64>::from_edges(4, &[(0, 1), (2, 3)], None, false, false).unwrap();
    let mut s_disc = KamadaKawaiSolver::<Layout2d<f64>>::from_graph(&disc, cfg.clone()).unwrap();
    let mut l_disc = Layout2d::from_xy(vec![0.0, 0.3, 4.0, 6.0], vec![0.0, 0.2, 1.0, 1.3]).unwrap();
    let status = s_disc.run(&mut l_disc).unwrap();
    assert!(status.is_finished());
    assert!(l_disc.is_all_finite());
    let d01 = (l_disc.x()[0] - l_disc.x()[1]).hypot(l_disc.y()[0] - l_disc.y()[1]);
    let d23 = (l_disc.x()[2] - l_disc.x()[3]).hypot(l_disc.y()[2] - l_disc.y()[3]);
    assert!((d01 - 1.0).abs() < 0.05, "d01={d01}");
    assert!((d23 - 1.0).abs() < 0.05, "d23={d23}");

    // Validation errors
    assert!(cfg.clone().with_eps(0.0).validate().is_err());
    assert!(cfg.clone().with_max_inner_iters(0).validate().is_err());
    assert!(matches!(
        KamadaKawaiSolver::<Layout2d<f64>>::from_distance_matrix(
            2,
            &[0.0, f64::NAN, 1.0, 0.0],
            cfg.clone()
        ),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        KamadaKawaiSolver::<Layout2d<f64>>::from_distance_matrix(
            2,
            &[0.0, -1.0, 1.0, 0.0],
            cfg.clone()
        ),
        Err(CoreError::InvalidInput(_))
    ));

    // max_iterations = Some(0) finishes immediately; run_steps(0) still validates target.
    let g2 = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
    let mut s_zero = KamadaKawaiSolver::<Layout2d<f64>>::from_graph(
        &g2,
        cfg.clone().with_max_iterations(Some(0)),
    )
    .unwrap();
    assert_eq!(s_zero.status(), SolverStatus::Completed);
    assert_eq!(s_zero.total_iterations(), Some(0));

    let mut l_wrong = Layout2d::<f64>::zeros(3);
    assert!(matches!(
        s_zero.run_steps(&mut l_wrong, 0),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert_eq!(s_zero.status(), SolverStatus::Completed);

    let mut s2 = KamadaKawaiSolver::<Layout2d<f64>>::from_graph(&g2, cfg).unwrap();
    assert!(matches!(
        s2.run_steps(&mut l_wrong, 0),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert_eq!(s2.status(), SolverStatus::Initialized);
    let mut l_nan = Layout2d::from_xy(vec![0.0, f64::NAN], vec![0.0, 1.0]).unwrap();
    assert!(matches!(
        s2.run_steps(&mut l_nan, 0),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert_eq!(s2.status(), SolverStatus::Initialized);
}

#[test]
fn test_kamada_kawai_fatal_breakdown_transitions_to_failed() {
    #[derive(Clone)]
    struct FailingRiemannian {
        coords: Vec<f64>,
    }

    impl LayoutStorage for FailingRiemannian {
        type Scalar = f64;
        fn len(&self) -> usize {
            self.coords.len() / 2
        }
        fn validate_finite(&self) -> CoreResult<()> {
            Ok(())
        }
    }

    impl MetricSpace for FailingRiemannian {
        fn distance(&self, i: usize, j: usize) -> f64 {
            let dx = self.coords[i * 2] - self.coords[j * 2];
            let dy = self.coords[i * 2 + 1] - self.coords[j * 2 + 1];
            dx.hypot(dy)
        }
    }

    impl RiemannianLayout for FailingRiemannian {
        fn tangent_dim(&self) -> usize {
            2
        }
        fn log_map_into(&self, i: usize, j: usize, out: &mut [f64]) -> CoreResult<()> {
            out[0] = self.coords[j * 2] - self.coords[i * 2];
            out[1] = self.coords[j * 2 + 1] - self.coords[i * 2 + 1];
            Ok(())
        }
        fn exp_map(&mut self, _i: usize, _delta: &[f64]) -> CoreResult<()> {
            Err(CoreError::numerical_breakdown("injected exp_map breakdown"))
        }
        fn parallel_transport_into(
            &self,
            _i: usize,
            _delta: &[f64],
            _vec: &mut [f64],
        ) -> CoreResult<()> {
            Ok(())
        }
    }

    let g = CsrGraph::<()>::from_unweighted_edges(2, &[(0, 1)], false, false).unwrap();
    let mut solver =
        KamadaKawaiSolver::<FailingRiemannian>::from_unweighted_graph(&g, KamadaKawaiConfig::new())
            .unwrap();
    let mut target = FailingRiemannian {
        coords: vec![0.0, 0.0, 0.2, 0.0],
    };

    assert!(matches!(
        solver.step(&mut target),
        Err(CoreError::NumericalBreakdown(_))
    ));
    assert_eq!(solver.status(), SolverStatus::Failed);
    assert!(solver.status().is_failed());
    assert!(solver.status().is_finished());
    assert_eq!(solver.step(&mut target).unwrap(), SolverStatus::Failed);
}
