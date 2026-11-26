use egraph_core::{
    CoreError, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Scalar, SolverStatus,
};

use super::{PcgWorkspace, StressMajorizationConfig, StressMajorizationSolver};

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
fn test_pcg_workspace_exact_zero_residual_and_singular_safety() {
    let mut pcg = PcgWorkspace::new(3);
    let a = [3.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0];
    let b = [5.0, 5.0, 0.0];
    let mut x = [0.0, 0.0, 7.0];
    let iters = pcg
        .solve_dense_symmetric(&a, &b, &mut x, 1e-10, 20)
        .unwrap();
    assert!(iters > 0);
    assert!((x[0] - 1.0).abs() < 1e-8);
    assert!((x[1] - 2.0).abs() < 1e-8);
    assert_eq!(x[2], 7.0);

    // Zero residual exits immediately with 0 iterations (no 0/0 NaN).
    let iters_zero = pcg
        .solve_dense_symmetric(&a, &b, &mut x, 1e-10, 20)
        .unwrap();
    assert_eq!(iters_zero, 0);

    // Zero matrix (all singular) exits safely in 0 iterations.
    let zero_a = [0.0; 4];
    let zero_b = [0.0; 2];
    let mut x2 = [1.0, -2.0];
    assert_eq!(
        pcg.solve_dense_symmetric(&zero_a, &zero_b, &mut x2, 1e-6, 10)
            .unwrap(),
        0
    );
    assert_eq!(x2, [1.0, -2.0]);
}

#[test]
fn test_stress_majorization_step_vs_run_bit_exact_f64_and_f32() {
    let grid = grid_graph(3, 3);
    let cfg = StressMajorizationConfig {
        max_iterations: 30,
        epsilon: 1e-6,
        ..Default::default()
    };

    // f64 bit-exact check
    let mut s_run = StressMajorizationSolver::<f64>::from_graph(&grid, cfg).unwrap();
    let mut s_step = StressMajorizationSolver::<f64>::from_graph(&grid, cfg).unwrap();
    let mut l_run = seeded_layout::<f64>(9, 1.0, 11);
    let mut l_step = l_run.clone();

    let stress_before = s_run.compute_stress(&l_run).unwrap();
    let st_run = s_run.run(&mut l_run).unwrap();
    assert!(st_run.is_finished());
    let stress_after = s_run.compute_stress(&l_run).unwrap();
    assert!(stress_after < stress_before * 0.2);

    let _ = s_step.run_steps(&mut l_step, 4).unwrap();
    while !s_step.status().is_finished() {
        s_step.step(&mut l_step).unwrap();
    }
    assert_eq!(s_run.status(), s_step.status());
    assert_eq!(s_run.iteration(), s_step.iteration());
    for i in 0..9 {
        assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
        assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
    }

    // f32 bit-exact check
    let mut s32_run = StressMajorizationSolver::<f32>::from_graph(&grid, cfg).unwrap();
    let mut s32_step = StressMajorizationSolver::<f32>::from_graph(&grid, cfg).unwrap();
    let mut l32_run = seeded_layout::<f32>(9, 1.0, 11);
    let mut l32_step = l32_run.clone();
    s32_run.run(&mut l32_run).unwrap();
    while !s32_step.status().is_finished() {
        s32_step.step(&mut l32_step).unwrap();
    }
    for i in 0..9 {
        assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
        assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
    }
}

#[test]
fn test_stress_majorization_disconnected_weighted_singleton_empty_and_errors() {
    let cfg = StressMajorizationConfig::default();

    // Empty and singleton graphs
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let mut s0 = StressMajorizationSolver::<f64>::from_graph(&empty, cfg).unwrap();
    let mut l0 = Layout2d::<f64>::zeros(0);
    assert_eq!(s0.run(&mut l0).unwrap(), SolverStatus::Converged);

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let mut s1 = StressMajorizationSolver::<f64>::from_graph(&single, cfg).unwrap();
    let mut l1 = Layout2d::from_xy(vec![3.0], vec![-4.0]).unwrap();
    assert_eq!(s1.run(&mut l1).unwrap(), SolverStatus::Converged);
    assert_eq!(l1.x(), &[3.0]);
    assert_eq!(l1.y(), &[-4.0]);

    // Disconnected components + isolated vertex remain finite and minimize component stress.
    let disc = CsrGraph::<f64>::from_edges(5, &[(0, 1), (2, 3)], None, false, false).unwrap();
    let mut s_disc = StressMajorizationSolver::<f64>::from_graph(&disc, cfg).unwrap();
    let mut l_disc = Layout2d::from_xy(
        vec![0.0, 0.2, 5.0, 8.0, -3.0],
        vec![0.0, 0.0, 1.0, 1.0, 2.0],
    )
    .unwrap();
    let status = s_disc.run(&mut l_disc).unwrap();
    assert!(status.is_finished());
    assert!(l_disc.is_all_finite());
    let d01 = (l_disc.x()[0] - l_disc.x()[1]).hypot(l_disc.y()[0] - l_disc.y()[1]);
    let d23 = (l_disc.x()[2] - l_disc.x()[3]).hypot(l_disc.y()[2] - l_disc.y()[3]);
    assert!((d01 - 1.0).abs() < 1e-2, "d01={d01}");
    assert!((d23 - 1.0).abs() < 1e-2, "d23={d23}");
    assert_eq!(l_disc.x()[4], -3.0);
    assert_eq!(l_disc.y()[4], 2.0);

    // Weighted graph via DijkstraWorkspace
    let wg = CsrGraph::<f64>::from_edges(
        3,
        &[(0, 1), (1, 2), (0, 2)],
        Some(&[1.0, 2.0, 10.0]),
        false,
        false,
    )
    .unwrap();
    let mut sw = StressMajorizationSolver::<f64>::from_graph(&wg, cfg).unwrap();
    let mut lw = Layout2d::from_xy(vec![0.0, 1.0, 5.0], vec![0.0, 0.2, 0.0]).unwrap();
    sw.run(&mut lw).unwrap();
    let d02 = (lw.x()[0] - lw.x()[2]).hypot(lw.y()[0] - lw.y()[2]);
    assert!((d02 - 3.0).abs() < 0.05, "d02={d02}");

    // Validation checks: NaN, negative distance, dimension mismatch
    assert!(matches!(
        StressMajorizationSolver::<f64>::from_distance_matrix(2, &[0.0, f64::NAN, 1.0, 0.0], cfg),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        StressMajorizationSolver::<f64>::from_distance_matrix(2, &[0.0, -1.0, -1.0, 0.0], cfg),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        StressMajorizationSolver::<f64>::from_distance_matrix(2, &[0.0, 1.0], cfg),
        Err(CoreError::DimensionMismatch { .. })
    ));
    let mut wrong_l = Layout2d::<f64>::zeros(4);
    assert!(matches!(
        sw.step(&mut wrong_l),
        Err(CoreError::DimensionMismatch { .. })
    ));
}
