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

    // Enforce release step-time ceiling (25 ms/step accounts for shared 2-vCPU CI VM jitter,
    // e.g. ~10.4 ms/step on GitHub ubuntu-latest; browser frame budget <= 10 ms is verified in Playwright).
    if cfg!(not(debug_assertions)) {
        assert!(
            ms_per_step <= 25.0,
            "ForceAtlas2 N=5000 exceeded 25 ms/step release budget: {ms_per_step:.3} ms/step"
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

#[test]
fn test_yifan_hu_weighted_undirected_coarse_symmetry_and_step_bit_exact() {
    // Regression test for Bug #1: 50 random weighted N=40 graphs (~127 edges, weights in [0.1, 10.1])
    // must construct symmetric coarse CSR graphs without 1-ULP InvalidCsr failures.
    let cfg_n40 = YifanHuConfig {
        max_iterations_per_level: 10,
        min_coarsen_size: 4,
        min_coarsen_ratio: 0.75,
        ..Default::default()
    };

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

        let mut s64_run = YifanHuSolver::<f64>::new(&graph, cfg_n40).unwrap();
        let mut s64_step = YifanHuSolver::<f64>::new(&graph, cfg_n40).unwrap();
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

        // Also test f32 solver construction and bit-exact step == run.
        let mut s32_run = YifanHuSolver::<f32>::new(&graph, cfg_n40).unwrap();
        let mut s32_step = YifanHuSolver::<f32>::new(&graph, cfg_n40).unwrap();
        let mut l32_run = Layout2d::from_xy(
            l64_run.x().iter().map(|&v| v as f32).collect(),
            l64_run.y().iter().map(|&v| v as f32).collect(),
        )
        .unwrap();
        let mut l32_step = l32_run.clone();
        let st32 = s32_run.run(&mut l32_run).unwrap();
        assert!(st32.is_finished());
        assert!(l32_run.is_all_finite());

        while !s32_step.status().is_finished() {
            s32_step.step(&mut l32_step).unwrap();
        }
        assert_eq!(s32_run.status(), s32_step.status());
        for i in 0..n {
            assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
            assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
        }
    }

    // N=300 disconnected weighted graph (3 components + 20 isolated nodes), f64 and f32.
    let mut rng = DeterministicRng::from_seed_u64(99_300);
    let n300 = 300usize;
    let mut edges300 = Vec::new();
    let mut weights300 = Vec::new();
    let components = [
        (0usize, 100usize),
        (100usize, 200usize),
        (200usize, 280usize),
    ];
    for &(lo, hi) in &components {
        for u in lo..(hi - 1) {
            edges300.push((u as u32, (u + 1) as u32));
            weights300.push(0.1 + 10.0 * rng.next_f64());
        }
        for u in lo..hi {
            for v in (u + 2)..hi {
                if rng.next_f64() < 0.08 {
                    edges300.push((u as u32, v as u32));
                    weights300.push(0.1 + 10.0 * rng.next_f64());
                }
            }
        }
    }
    let disc300 =
        CsrGraph::<f64>::from_edges(n300, &edges300, Some(&weights300), false, false).unwrap();
    let cfg_n300 = YifanHuConfig {
        max_iterations_per_level: 12,
        min_coarsen_size: 30,
        min_coarsen_ratio: 0.75,
        ..Default::default()
    };

    let mut s300_64_run = YifanHuSolver::<f64>::new(&disc300, cfg_n300).unwrap();
    let mut s300_64_step = YifanHuSolver::<f64>::new(&disc300, cfg_n300).unwrap();
    assert!(s300_64_run.num_levels() >= 3);
    let mut l300_64_run = seeded_layout(n300, 300);
    let mut l300_64_step = l300_64_run.clone();
    assert!(s300_64_run.run(&mut l300_64_run).unwrap().is_finished());
    while !s300_64_step.status().is_finished() {
        s300_64_step.step(&mut l300_64_step).unwrap();
    }
    assert!(l300_64_run.is_all_finite());
    for i in 0..n300 {
        assert_eq!(l300_64_run.x()[i].to_bits(), l300_64_step.x()[i].to_bits());
        assert_eq!(l300_64_run.y()[i].to_bits(), l300_64_step.y()[i].to_bits());
    }

    let mut s300_32_run = YifanHuSolver::<f32>::new(&disc300, cfg_n300).unwrap();
    let mut s300_32_step = YifanHuSolver::<f32>::new(&disc300, cfg_n300).unwrap();
    let mut l300_32_run = Layout2d::from_xy(
        l300_64_run.x().iter().map(|&v| v as f32).collect(),
        l300_64_run.y().iter().map(|&v| v as f32).collect(),
    )
    .unwrap();
    let mut l300_32_step = l300_32_run.clone();
    assert!(s300_32_run.run(&mut l300_32_run).unwrap().is_finished());
    while !s300_32_step.status().is_finished() {
        s300_32_step.step(&mut l300_32_step).unwrap();
    }
    assert!(l300_32_run.is_all_finite());
    for i in 0..n300 {
        assert_eq!(l300_32_run.x()[i].to_bits(), l300_32_step.x()[i].to_bits());
        assert_eq!(l300_32_run.y()[i].to_bits(), l300_32_step.y()[i].to_bits());
    }
}
