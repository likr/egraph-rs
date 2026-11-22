//! Integration tests for M4 Lane 3 (`ContinuousFdp`: `ForceAtlas2` and multilevel `Yifan Hu`).

use std::time::Instant;

use egraph_core::{CsrGraph, DeterministicRng, IterativeSolver, Layout2d, SolverStatus};
use egraph_layout::continuous_fdp::{
    ContinuousFdp, ForceAtlas2Config, ForceAtlas2Solver, YifanHuConfig, YifanHuSolver,
};

fn make_grid_csr(rows: usize, cols: usize) -> CsrGraph<f64> {
    let n = rows * cols;
    let mut edges = Vec::with_capacity(2 * n);
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

fn seeded_layout(n: usize, seed: u64) -> Layout2d<f64> {
    let mut rng = DeterministicRng::from_seed_u64(seed);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        x.push(20.0 * (rng.next_f64() - 0.5));
        y.push(20.0 * (rng.next_f64() - 0.5));
    }
    Layout2d::from_xy(x, y).unwrap()
}

#[test]
fn test_force_atlas2_n5000_step_budget_and_stability() {
    // 50 x 100 grid => N = 5000, E = 9850.
    let graph = make_grid_csr(50, 100);
    assert_eq!(graph.num_nodes(), 5000);

    let cfg = ForceAtlas2Config {
        iterations: 10,
        use_barnes_hut: true,
        theta: 0.5,
        ..Default::default()
    };
    let mut solver = ForceAtlas2Solver::<f64>::new(&graph, cfg).unwrap();
    let mut layout = seeded_layout(5000, 42);

    // Warm up one step then measure 9 steps.
    assert_eq!(solver.step(&mut layout).unwrap(), SolverStatus::Running);
    let start = Instant::now();
    assert_eq!(solver.run(&mut layout).unwrap(), SolverStatus::Completed);
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let ms_per_step = elapsed_ms / 9.0;

    assert!(layout.is_all_finite());
    assert_eq!(solver.iteration(), 10);

    // Enforce strict <= 10 ms/step budget in release builds; debug builds just verify finite sanity.
    if cfg!(not(debug_assertions)) {
        assert!(
            ms_per_step <= 10.0,
            "ForceAtlas2 N=5000 exceeded 10 ms/step release budget: {ms_per_step:.3} ms/step"
        );
    }
}

#[test]
fn test_yifan_hu_multilevel_coarsening_depth_and_convergence() {
    // 20 x 25 grid => N = 500.
    let graph = make_grid_csr(20, 25);
    let cfg = YifanHuConfig {
        max_iterations_per_level: 50,
        min_coarsen_size: 25,
        min_coarsen_ratio: 0.75,
        ..Default::default()
    };
    let mut solver = YifanHuSolver::<f64>::new(&graph, cfg).unwrap();
    assert!(
        solver.num_levels() >= 4,
        "expected >= 4 levels on N=500 grid, got {}",
        solver.num_levels()
    );
    assert_eq!(solver.current_level(), solver.num_levels() - 1);

    let mut layout = seeded_layout(500, 2026);
    let mut wrapped = ContinuousFdp::YifanHu(solver.clone());
    assert_eq!(wrapped.num_nodes(), 500);

    let status = solver.run(&mut layout).unwrap();
    assert!(status.is_finished());
    assert_eq!(solver.current_level(), 0);
    assert!(layout.is_all_finite());

    let mut layout2 = seeded_layout(500, 2026);
    while !wrapped.status().is_finished() {
        wrapped.step(&mut layout2).unwrap();
    }
    assert_eq!(wrapped.status(), status);
    assert_eq!(wrapped.iteration(), solver.iteration());
    for i in 0..500 {
        assert_eq!(layout.x()[i].to_bits(), layout2.x()[i].to_bits());
        assert_eq!(layout.y()[i].to_bits(), layout2.y()[i].to_bits());
    }
}
