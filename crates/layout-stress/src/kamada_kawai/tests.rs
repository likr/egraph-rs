use egraph_core::{
    CoreError, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Scalar, SolverStatus,
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
fn test_kamada_kawai_step_vs_run_bit_exact_f64_and_f32() {
    let grid = grid_graph(3, 3);
    let cfg = KamadaKawaiConfig::default();

    // f64 bit-exact parity
    let mut s_run = KamadaKawaiSolver::<f64>::from_graph(&grid, cfg).unwrap();
    let mut s_step = KamadaKawaiSolver::<f64>::from_graph(&grid, cfg).unwrap();
    let mut l_run = seeded_layout::<f64>(9, 1.5, 21);
    let mut l_step = l_run.clone();

    assert_eq!(s_run.total_iterations(), 100);
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

    // f32 bit-exact parity
    let mut s32_run = KamadaKawaiSolver::<f32>::from_graph(&grid, cfg).unwrap();
    let mut s32_step = KamadaKawaiSolver::<f32>::from_graph(&grid, cfg).unwrap();
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
}

#[test]
fn test_kamada_kawai_degenerate_collinear_det_zero_no_infinite_loop() {
    // Construct exact Hessian singularity (det == 0) at node 0 on a collinear x-axis layout:
    // x = [0.0, 0.5, 1.0], y = [0.0, 0.0, 0.0], with l_{01} = 1.0 and l_{02} = (sqrt(5) - 1) / 2.
    // For node 0: hyy = 1*(1 - 1/0.5) + k_{02}*(1 - l_{02}/1.0) = -1 + (1 - l_{02})/l_{02}^2 = 0,
    // and hxy = 0 => det = hxx * hyy - hxy^2 = 0 exactly while dedx != 0!
    let phi_inv = (5.0_f64.sqrt() - 1.0) * 0.5;
    let dists = [
        0.0, 1.0, phi_inv, //
        1.0, 0.0, 1.0, //
        phi_inv, 1.0, 0.0,
    ];
    let cfg = KamadaKawaiConfig {
        eps: 1e-3,
        max_inner_iters: 50,
        max_iterations: Some(60),
    };
    let mut solver = KamadaKawaiSolver::<f64>::from_distance_matrix(3, &dists, cfg).unwrap();
    let mut layout = Layout2d::from_xy(vec![0.0, 0.5, 1.0], vec![0.0, 0.0, 0.0]).unwrap();
    let status = solver.run(&mut layout).unwrap();
    assert!(status.is_finished());
    assert!(solver.iteration() <= 60);
    assert!(layout.is_all_finite());

    // Also verify default (10 * n).max(100) bound on n = 15 and `new()` constructor parity.
    let g15 = grid_graph(3, 5);
    let s15 = KamadaKawaiSolver::<f64>::new(&g15, KamadaKawaiConfig::default()).unwrap();
    assert_eq!(s15.total_iterations(), 150);
}

#[test]
fn test_kamada_kawai_edge_cases_disconnected_and_errors() {
    let cfg = KamadaKawaiConfig::default();

    // Empty and singleton graphs
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let mut s0 = KamadaKawaiSolver::<f64>::from_graph(&empty, cfg).unwrap();
    let mut l0 = Layout2d::<f64>::zeros(0);
    assert_eq!(s0.run(&mut l0).unwrap(), SolverStatus::Converged);

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let mut s1 = KamadaKawaiSolver::<f64>::from_graph(&single, cfg).unwrap();
    let mut l1 = Layout2d::from_xy(vec![1.0], vec![2.0]).unwrap();
    assert_eq!(s1.run(&mut l1).unwrap(), SolverStatus::Converged);

    // Disconnected graph remains finite and preserves target distances per component
    let disc = CsrGraph::<f64>::from_edges(4, &[(0, 1), (2, 3)], None, false, false).unwrap();
    let mut s_disc = KamadaKawaiSolver::<f64>::from_graph(&disc, cfg).unwrap();
    let mut l_disc = Layout2d::from_xy(vec![0.0, 0.3, 4.0, 6.0], vec![0.0, 0.2, 1.0, 1.3]).unwrap();
    let status = s_disc.run(&mut l_disc).unwrap();
    assert!(status.is_finished());
    assert!(l_disc.is_all_finite());
    let d01 = (l_disc.x()[0] - l_disc.x()[1]).hypot(l_disc.y()[0] - l_disc.y()[1]);
    let d23 = (l_disc.x()[2] - l_disc.x()[3]).hypot(l_disc.y()[2] - l_disc.y()[3]);
    assert!((d01 - 1.0).abs() < 0.05, "d01={d01}");
    assert!((d23 - 1.0).abs() < 0.05, "d23={d23}");

    // Validation errors
    assert!(KamadaKawaiConfig { eps: 0.0, ..cfg }.validate().is_err());
    assert!(
        KamadaKawaiConfig {
            max_inner_iters: 0,
            ..cfg
        }
        .validate()
        .is_err()
    );
    assert!(matches!(
        KamadaKawaiSolver::<f64>::from_distance_matrix(2, &[0.0, f64::NAN, 1.0, 0.0], cfg),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        KamadaKawaiSolver::<f64>::from_distance_matrix(2, &[0.0, -1.0, 1.0, 0.0], cfg),
        Err(CoreError::InvalidInput(_))
    ));

    // max_iterations = Some(0) finishes immediately; dimension mismatch & NaN rejected.
    let g2 = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
    let s_zero = KamadaKawaiSolver::<f64>::from_graph(
        &g2,
        KamadaKawaiConfig {
            max_iterations: Some(0),
            ..cfg
        },
    )
    .unwrap();
    assert_eq!(s_zero.status(), SolverStatus::Completed);
    assert_eq!(s_zero.total_iterations(), 0);

    let mut s2 = KamadaKawaiSolver::<f64>::from_graph(&g2, cfg).unwrap();
    let mut l_wrong = Layout2d::<f64>::zeros(3);
    assert!(matches!(
        s2.step(&mut l_wrong),
        Err(CoreError::DimensionMismatch { .. })
    ));
    let mut l_nan = Layout2d::from_xy(vec![0.0, f64::NAN], vec![0.0, 1.0]).unwrap();
    assert!(matches!(
        s2.step(&mut l_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));
}
