//! Integration tests for `ForceAtlas2Solver` and multilevel `YifanHuSolver`.

use std::time::Instant;

use egraph_core::{
    CoreError, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Layout3d, SolverStatus,
};
use egraph_layout_force::{ForceAtlas2Config, ForceAtlas2Solver, YifanHuConfig, YifanHuSolver};

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
    let graph = make_grid_csr(50, 100);
    assert_eq!(graph.num_nodes(), 5000);

    let cfg = ForceAtlas2Config::new()
        .with_iterations(10)
        .with_use_barnes_hut(true)
        .with_theta(0.5);
    let mut solver = ForceAtlas2Solver::<Layout2d<f64>>::new(&graph, cfg).unwrap();
    let mut layout = seeded_layout(5000, 42);

    assert_eq!(solver.step(&mut layout).unwrap(), SolverStatus::Running);
    let start = Instant::now();
    assert_eq!(solver.run(&mut layout).unwrap(), SolverStatus::Completed);
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let ms_per_step = elapsed_ms / 9.0;

    assert!(layout.is_all_finite());
    assert_eq!(solver.iteration(), 10);

    if cfg!(not(debug_assertions)) {
        assert!(
            ms_per_step <= 25.0,
            "ForceAtlas2 N=5000 exceeded 25 ms/step release budget: {ms_per_step:.3} ms/step"
        );
    }
}

#[test]
fn test_yifan_hu_multilevel_coarsening_depth_and_convergence() {
    let graph = make_grid_csr(20, 25);
    let cfg = YifanHuConfig::new()
        .with_max_iterations_per_level(50)
        .with_min_coarsen_size(25)
        .with_min_coarsen_ratio(0.75);
    let mut solver = YifanHuSolver::<Layout2d<f64>>::new(&graph, cfg.clone()).unwrap();
    assert!(
        solver.num_levels() >= 4,
        "expected >= 4 levels on N=500 grid, got {}",
        solver.num_levels()
    );
    assert_eq!(solver.current_level(), solver.num_levels() - 1);

    let mut layout = seeded_layout(500, 2026);
    let mut solver2 = YifanHuSolver::<Layout2d<f64>>::new(&graph, cfg).unwrap();
    assert_eq!(solver2.num_nodes(), 500);

    let status = solver.run(&mut layout).unwrap();
    assert!(status.is_finished());
    assert_eq!(solver.current_level(), 0);
    assert!(layout.is_all_finite());

    let mut layout2 = seeded_layout(500, 2026);
    while !solver2.status().is_finished() {
        solver2.step(&mut layout2).unwrap();
    }
    assert_eq!(solver2.status(), status);
    assert_eq!(solver2.iteration(), solver.iteration());
    for i in 0..500 {
        assert_eq!(layout.x()[i].to_bits(), layout2.x()[i].to_bits());
        assert_eq!(layout.y()[i].to_bits(), layout2.y()[i].to_bits());
    }
}

#[test]
fn test_yifan_hu_weighted_undirected_coarse_symmetry_and_3d_bit_exact() {
    let cfg_n40 = YifanHuConfig::new()
        .with_max_iterations_per_level(10)
        .with_min_coarsen_size(4)
        .with_min_coarsen_ratio(0.75);

    for trial in 0..50u64 {
        let mut rng = DeterministicRng::from_seed_u64(10_000 + trial);
        let n = 40usize;
        let mut edges = Vec::new();
        let mut weights = Vec::new();
        for u in 0..n {
            for v in (u + 1)..n {
                if rng.next_f64() < 0.163 {
                    edges.push((u as u32, v as u32));
                    weights.push(0.1 + 10.0 * rng.next_f64());
                }
            }
        }
        let graph = CsrGraph::<f64>::from_edges(n, &edges, Some(&weights), false, false).unwrap();

        let mut s64_run = YifanHuSolver::<Layout2d<f64>>::new(&graph, cfg_n40.clone()).unwrap();
        let mut s64_step = YifanHuSolver::<Layout2d<f64>>::new(&graph, cfg_n40.clone()).unwrap();
        assert!(s64_run.num_levels() >= 2);

        let mut l64_run = seeded_layout(n, 2000 + trial);
        let mut l64_step = l64_run.clone();
        let st64 = s64_run.run(&mut l64_run).unwrap();
        assert!(st64.is_finished());
        assert!(l64_run.is_all_finite());

        while !s64_step.status().is_finished() {
            s64_step.step(&mut l64_step).unwrap();
        }
        assert_eq!(s64_run.status(), s64_step.status());
        for i in 0..n {
            assert_eq!(l64_run.x()[i].to_bits(), l64_step.x()[i].to_bits());
            assert_eq!(l64_run.y()[i].to_bits(), l64_step.y()[i].to_bits());
        }

        let mut s32_run = YifanHuSolver::<Layout3d<f32>>::new(&graph, cfg_n40.clone()).unwrap();
        let mut s32_step = YifanHuSolver::<Layout3d<f32>>::new(&graph, cfg_n40.clone()).unwrap();
        let mut l32_run = Layout3d::<f32>::init(n, 3000 + trial, 2.0).unwrap();
        let mut l32_step = l32_run.clone();
        assert!(s32_run.run(&mut l32_run).unwrap().is_finished());
        while !s32_step.status().is_finished() {
            s32_step.step(&mut l32_step).unwrap();
        }
        for i in 0..n {
            assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
            assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
            assert_eq!(l32_run.z()[i].to_bits(), l32_step.z()[i].to_bits());
        }
    }

    let neg_g = CsrGraph::from_edges(2, &[(0, 1)], Some(&[-1.0]), false, false).unwrap();
    assert!(matches!(
        ForceAtlas2Solver::<Layout2d<f64>>::new(&neg_g, ForceAtlas2Config::new()),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        YifanHuSolver::<Layout2d<f64>>::new(&neg_g, YifanHuConfig::new()),
        Err(CoreError::InvalidInput(_))
    ));
}
