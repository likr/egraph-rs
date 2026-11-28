use egraph_core::{
    CoreError, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Scalar, SolverStatus,
};
use egraph_neighborhood::{
    DogVaradhanConfig, DogVaradhanLandmarks, KnnGraph, SparseProbabilities,
    compute_regularized_dog_varadhan_landmarks,
};

use super::{GuMapConfig, GuMapSolver};

fn barbell_graph(clique_size: usize) -> CsrGraph<f64> {
    let n = 2 * clique_size;
    let mut edges = Vec::new();
    for c in 0..2 {
        let off = (c * clique_size) as u32;
        for i in 0..clique_size as u32 {
            for j in (i + 1)..clique_size as u32 {
                edges.push((off + i, off + j));
            }
        }
    }
    edges.push((0, clique_size as u32));
    CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
}

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
fn test_gumap_config_defaults_ab_and_validation() {
    let cfg = GuMapConfig::default();
    assert_eq!(cfg.iterations, 200);
    assert_eq!(cfg.negative_sample_rate, 5);
    assert_eq!(cfg.learning_rate, 1.0);
    assert_eq!(cfg.min_dist, 0.1);
    assert_eq!(cfg.spread, 1.0);
    assert_eq!(cfg.a, None);
    assert_eq!(cfg.b, None);
    assert_eq!(cfg.sigma_iters, 64);
    assert_eq!(cfg.sigma_tolerance, 1e-5);
    assert!(cfg.symmetric_negative);
    assert_eq!(cfg.seed, 42);
    cfg.validate().unwrap();
    let (a, b) = cfg.resolve_ab().unwrap();
    assert!((a - 1.577).abs() < 0.02);
    assert!((b - 0.895).abs() < 0.02);

    let bad_cases = [
        GuMapConfig::new().with_learning_rate(0.0),
        GuMapConfig::new().with_learning_rate(f64::NAN),
        GuMapConfig::new().with_min_dist(-0.1),
        GuMapConfig::new().with_spread(0.0),
        GuMapConfig::new().with_a(Some(-1.0)),
        GuMapConfig::new().with_b(Some(0.0)),
        GuMapConfig::new().with_sigma_iters(0),
        GuMapConfig::new().with_sigma_tolerance(0.0),
    ];
    for bad in bad_cases {
        assert!(bad.validate().is_err());
    }
}

#[test]
fn test_gumap_step_vs_run_bit_exact_f64_and_f32() {
    let grid = grid_graph(4, 4);
    let cfg = GuMapConfig::new().with_iterations(25).with_seed(99);
    let dog_cfg = DogVaradhanConfig::default();

    // f64 bit-exact parity (and `GuMapSolver::new` parity)
    let mut s_run = GuMapSolver::<Layout2d<f64>>::new(&grid, cfg.clone()).unwrap();
    let mut s_step =
        GuMapSolver::<Layout2d<f64>>::from_graph_dog(&grid, 32, &dog_cfg, cfg.clone()).unwrap();
    let mut l_run = seeded_layout::<f64>(grid.num_nodes(), 1.0, 17);
    let mut l_step = l_run.clone();

    assert_eq!(s_run.status(), SolverStatus::Initialized);
    assert_eq!(s_run.total_iterations(), Some(25));
    assert_eq!(s_run.run(&mut l_run).unwrap(), SolverStatus::Completed);

    assert_eq!(
        s_step.run_steps(&mut l_step, 7).unwrap(),
        SolverStatus::Running
    );
    assert_eq!(s_step.iteration(), 7);
    while !s_step.status().is_finished() {
        s_step.step(&mut l_step).unwrap();
    }
    assert_eq!(s_step.status(), SolverStatus::Completed);
    assert_eq!(s_step.iteration(), 25);

    for i in 0..grid.num_nodes() {
        assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
        assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
    }

    // f32 bit-exact parity
    let mut s32_run =
        GuMapSolver::<Layout2d<f32>>::from_graph_dog(&grid, 32, &dog_cfg, cfg.clone()).unwrap();
    let mut s32_step =
        GuMapSolver::<Layout2d<f32>>::from_graph_dog(&grid, 32, &dog_cfg, cfg).unwrap();
    let mut l32_run = seeded_layout::<f32>(grid.num_nodes(), 1.0, 17);
    let mut l32_step = l32_run.clone();
    s32_run.run(&mut l32_run).unwrap();
    for _ in 0..25 {
        s32_step.step(&mut l32_step).unwrap();
    }
    assert_eq!(s32_step.status(), SolverStatus::Completed);
    for i in 0..grid.num_nodes() {
        assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
        assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
    }
}

#[test]
fn test_gumap_dist_sq_powf_b_and_dog_separates_barbell_clusters() {
    let barbell = barbell_graph(6);
    let cfg = GuMapConfig::new()
        .with_iterations(150)
        .with_negative_sample_rate(5)
        .with_learning_rate(1.0)
        .with_seed(42);
    let dog_cfg = DogVaradhanConfig::default();
    let mut solver =
        GuMapSolver::<Layout2d<f64>>::from_graph_dog(&barbell, 32, &dog_cfg, cfg).unwrap();
    assert!(solver.num_edges() > 0);
    let (a, b) = solver.ab();
    assert!(a > 0.0 && b > 0.0);
    assert_eq!(solver.a(), a);
    assert_eq!(solver.b(), b);

    // Also test starting from all-zeros layout (triggers deterministic spread init).
    let mut layout = Layout2d::<f64>::zeros(barbell.num_nodes());
    assert_eq!(solver.run(&mut layout).unwrap(), SolverStatus::Completed);
    assert!(layout.is_all_finite());

    let c0x = layout.x()[..6].iter().sum::<f64>() / 6.0;
    let c0y = layout.y()[..6].iter().sum::<f64>() / 6.0;
    let c1x = layout.x()[6..].iter().sum::<f64>() / 6.0;
    let c1y = layout.y()[6..].iter().sum::<f64>() / 6.0;
    let centroid_dist = (c0x - c1x).hypot(c0y - c1y);
    let mut intra = 0.0_f64;
    for i in 0..6 {
        intra += (layout.x()[i] - c0x).hypot(layout.y()[i] - c0y);
        intra += (layout.x()[6 + i] - c1x).hypot(layout.y()[6 + i] - c1y);
    }
    intra /= 12.0;
    assert!(
        centroid_dist > intra * 1.5,
        "expected barbell cluster separation, got centroid_dist={centroid_dist}, intra={intra}"
    );
}

#[test]
fn test_gumap_edge_cases_and_validation_errors() {
    let cfg = GuMapConfig::default();

    // Empty and singleton graphs via from_graph_dog
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let mut s_empty = GuMapSolver::<Layout2d<f64>>::from_graph_dog(
        &empty,
        16,
        &DogVaradhanConfig::default(),
        cfg.clone(),
    )
    .unwrap();
    let mut l_empty = Layout2d::<f64>::zeros(0);
    assert_eq!(s_empty.run(&mut l_empty).unwrap(), SolverStatus::Converged);

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let mut s_single = GuMapSolver::<Layout2d<f64>>::from_graph_dog(
        &single,
        16,
        &DogVaradhanConfig::default(),
        cfg.clone(),
    )
    .unwrap();
    let mut l_single = Layout2d::from_xy(vec![2.0], vec![-3.0]).unwrap();
    assert_eq!(
        s_single.run(&mut l_single).unwrap(),
        SolverStatus::Converged
    );
    assert_eq!(l_single.x(), &[2.0]);
    assert_eq!(l_single.y(), &[-3.0]);

    // Zero epochs completes immediately
    let grid = grid_graph(3, 3);
    let cfg_zero = cfg.clone().with_iterations(0);
    let s_zero = GuMapSolver::<Layout2d<f64>>::from_graph_dog(
        &grid,
        16,
        &DogVaradhanConfig::default(),
        cfg_zero,
    )
    .unwrap();
    assert_eq!(s_zero.status(), SolverStatus::Completed);
    assert_eq!(s_zero.total_iterations(), Some(0));

    // Dimension mismatch & non-finite layout rejected (including run_steps(_, 0))
    let mut s_grid = GuMapSolver::<Layout2d<f64>>::from_knn(
        &KnnGraph::new_validated(2, vec![0, 1, 2], vec![1, 0], vec![1.0, 1.0]).unwrap(),
        cfg.clone(),
    )
    .unwrap();
    let mut l_wrong = Layout2d::<f64>::zeros(3);
    assert!(matches!(
        s_grid.step(&mut l_wrong),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        s_grid.run_steps(&mut l_wrong, 0),
        Err(CoreError::DimensionMismatch { .. })
    ));
    let mut l_nan = Layout2d::from_xy(vec![0.0, f64::NAN], vec![0.0, 1.0]).unwrap();
    assert!(matches!(
        s_grid.step(&mut l_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        s_grid.run_steps(&mut l_nan, 0),
        Err(CoreError::NonFiniteValue { .. })
    ));

    // Valid target with run_steps(_, 0) keeps solver status unchanged.
    let mut l_ok = Layout2d::from_xy(vec![0.0, 1.0], vec![0.0, 0.0]).unwrap();
    assert_eq!(
        s_grid.run_steps(&mut l_ok, 0).unwrap(),
        SolverStatus::Initialized
    );
    assert_eq!(s_grid.iteration(), 0);

    // Invalid SparseProbabilities rejected by SparseProbabilities::from_csr.
    assert!(matches!(
        SparseProbabilities::from_csr(2, vec![0, 1, 1], vec![2], vec![0.5]),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    let _ = cfg;
}

#[test]
fn test_gumap_from_fuzzy_set_interleaves_symmetric_pairs() {
    let probs = SparseProbabilities::from_csr(
        4,
        vec![0, 2, 4, 6, 8],
        vec![1, 3, 0, 2, 1, 3, 0, 2],
        vec![0.8, 0.5, 0.8, 0.6, 0.6, 0.4, 0.5, 0.4],
    )
    .unwrap();
    let solver =
        GuMapSolver::<Layout2d<f64>>::from_fuzzy_set(probs, GuMapConfig::default()).unwrap();
    let got: Vec<(u32, u32)> = solver
        .core
        .edge_src
        .iter()
        .copied()
        .zip(solver.core.edge_tgt.iter().copied())
        .collect();
    assert_eq!(
        got,
        vec![
            (0, 1),
            (1, 0),
            (0, 3),
            (3, 0),
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 2)
        ]
    );
}

#[test]
fn test_gumap_from_landmarks_bit_exact_and_validation() {
    let grid = grid_graph(4, 4);
    let n = grid.num_nodes();
    let dog_cfg = DogVaradhanConfig::default();
    let cfg = GuMapConfig::new().with_iterations(20).with_seed(55);
    let (landmarks, _) = compute_regularized_dog_varadhan_landmarks(&grid, &dog_cfg).unwrap();

    let mut s_dog =
        GuMapSolver::<Layout2d<f64>>::from_graph_dog(&grid, 32, &dog_cfg, cfg.clone()).unwrap();
    let mut s_lm =
        GuMapSolver::<Layout2d<f64>>::from_landmarks(&grid, &landmarks, 32, cfg.clone()).unwrap();
    let mut l_dog = seeded_layout::<f64>(n, 1.0, 21);
    let mut l_lm = l_dog.clone();
    s_dog.run(&mut l_dog).unwrap();
    s_lm.run(&mut l_lm).unwrap();
    for i in 0..n {
        assert_eq!(l_dog.x()[i].to_bits(), l_lm.x()[i].to_bits());
        assert_eq!(l_dog.y()[i].to_bits(), l_lm.y()[i].to_bits());
    }

    // Unweighted CsrGraph<()> works without requiring W: Scalar.
    let mut edges = Vec::new();
    for r in 0..4 {
        for c in 0..4 {
            let u = (r * 4 + c) as u32;
            if c + 1 < 4 {
                edges.push((u, u + 1));
            }
            if r + 1 < 4 {
                edges.push((u, u + 4));
            }
        }
    }
    let unweighted = CsrGraph::<()>::from_unweighted_edges(n, &edges, false, false).unwrap();
    let mut s_unweighted =
        GuMapSolver::<Layout2d<f64>>::from_landmarks(&unweighted, &landmarks, 32, cfg.clone())
            .unwrap();
    let mut l_unweighted = seeded_layout::<f64>(n, 1.0, 21);
    s_unweighted.run(&mut l_unweighted).unwrap();
    for i in 0..n {
        assert_eq!(l_lm.x()[i].to_bits(), l_unweighted.x()[i].to_bits());
        assert_eq!(l_lm.y()[i].to_bits(), l_unweighted.y()[i].to_bits());
    }

    // Validation errors via public DogVaradhanLandmarks constructors
    let bad_nodes = DogVaradhanLandmarks::empty(n + 1).unwrap();
    assert!(matches!(
        GuMapSolver::<Layout2d<f64>>::from_landmarks(&grid, &bad_nodes, 32, cfg.clone()),
        Err(CoreError::DimensionMismatch { .. })
    ));

    let bad_empty_lm = DogVaradhanLandmarks::empty(n).unwrap();
    assert!(matches!(
        GuMapSolver::<Layout2d<f64>>::from_landmarks(&grid, &bad_empty_lm, 32, cfg),
        Err(CoreError::InvalidInput(_))
    ));

    assert!(matches!(
        DogVaradhanLandmarks::new_validated(n, vec![0], vec![f64::NAN; n], None),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        DogVaradhanLandmarks::new_validated(n, vec![n as u32], vec![1.0; n], None),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
}

#[test]
fn test_gumap_symmetric_negative_updates_both_endpoints_and_step_parity() {
    // 3 nodes: symmetric edge (0, 1) with weight 1.0, and isolated node 2.
    // With high negative_sample_rate, node 2 is sampled as a negative example.
    let probs =
        SparseProbabilities::from_csr(3, vec![0, 1, 2, 2], vec![1, 0], vec![1.0, 1.0]).unwrap();
    let init = Layout2d::from_xy(vec![0.0, 1.0, 0.25], vec![0.0, 0.0, 0.25]).unwrap();

    let cfg_sym = GuMapConfig::new()
        .with_iterations(1)
        .with_negative_sample_rate(20)
        .with_learning_rate(1.0)
        .with_symmetric_negative(true)
        .with_seed(42);
    let cfg_asym = cfg_sym.clone().with_symmetric_negative(false);

    let mut l_sym = init.clone();
    let mut s_sym = GuMapSolver::<Layout2d<f64>>::from_fuzzy_set(probs.clone(), cfg_sym).unwrap();
    s_sym.step(&mut l_sym).unwrap();
    let mut l_asym = init;
    let mut s_asym = GuMapSolver::<Layout2d<f64>>::from_fuzzy_set(probs, cfg_asym).unwrap();
    s_asym.step(&mut l_asym).unwrap();

    // When symmetric_negative is false, isolated node 2 never moves.
    assert_eq!(l_asym.x()[2], 0.25);
    assert_eq!(l_asym.y()[2], 0.25);
    // When symmetric_negative is true, isolated node 2 moves when sampled as a negative neighbor.
    assert!(
        (l_sym.x()[2] - 0.25).hypot(l_sym.y()[2] - 0.25) > 1e-6,
        "expected symmetric negative update to move isolated negative sample 2"
    );

    // step == run bit-exactness (f64 and f32) for symmetric_negative in {true, false}.
    let grid = grid_graph(4, 4);
    let dog_cfg = DogVaradhanConfig::default();
    for sym in [true, false] {
        let cfg = GuMapConfig::new()
            .with_iterations(15)
            .with_symmetric_negative(sym)
            .with_seed(77);
        let mut s64_r =
            GuMapSolver::<Layout2d<f64>>::from_graph_dog(&grid, 32, &dog_cfg, cfg.clone()).unwrap();
        let mut s64_s =
            GuMapSolver::<Layout2d<f64>>::from_graph_dog(&grid, 32, &dog_cfg, cfg.clone()).unwrap();
        let mut l64_r = seeded_layout::<f64>(grid.num_nodes(), 1.0, 17);
        let mut l64_s = l64_r.clone();
        s64_r.run(&mut l64_r).unwrap();
        s64_s.run_steps(&mut l64_s, 5).unwrap();
        while !s64_s.status().is_finished() {
            s64_s.step(&mut l64_s).unwrap();
        }
        for i in 0..grid.num_nodes() {
            assert_eq!(l64_r.x()[i].to_bits(), l64_s.x()[i].to_bits());
            assert_eq!(l64_r.y()[i].to_bits(), l64_s.y()[i].to_bits());
        }

        let mut s32_r =
            GuMapSolver::<Layout2d<f32>>::from_graph_dog(&grid, 32, &dog_cfg, cfg.clone()).unwrap();
        let mut s32_s =
            GuMapSolver::<Layout2d<f32>>::from_graph_dog(&grid, 32, &dog_cfg, cfg).unwrap();
        let mut l32_r = seeded_layout::<f32>(grid.num_nodes(), 1.0, 17);
        let mut l32_s = l32_r.clone();
        s32_r.run(&mut l32_r).unwrap();
        s32_s.run_steps(&mut l32_s, 5).unwrap();
        while !s32_s.status().is_finished() {
            s32_s.step(&mut l32_s).unwrap();
        }
        for i in 0..grid.num_nodes() {
            assert_eq!(l32_r.x()[i].to_bits(), l32_s.x()[i].to_bits());
            assert_eq!(l32_r.y()[i].to_bits(), l32_s.y()[i].to_bits());
        }
    }
}
