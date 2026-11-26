use egraph_core::{
    CoreError, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Scalar, SolverStatus,
};
use egraph_neighborhood::{
    DogVaradhanConfig, KnnGraph, SparseProbabilities, compute_regularized_dog_varadhan_landmarks,
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
    assert_eq!(cfg.n_epochs, 200);
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
        GuMapConfig {
            learning_rate: 0.0,
            ..cfg
        },
        GuMapConfig {
            learning_rate: f64::NAN,
            ..cfg
        },
        GuMapConfig {
            min_dist: -0.1,
            ..cfg
        },
        GuMapConfig { spread: 0.0, ..cfg },
        GuMapConfig {
            a: Some(-1.0),
            ..cfg
        },
        GuMapConfig {
            b: Some(0.0),
            ..cfg
        },
        GuMapConfig {
            sigma_iters: 0,
            ..cfg
        },
        GuMapConfig {
            sigma_tolerance: 0.0,
            ..cfg
        },
    ];
    for bad in bad_cases {
        assert!(bad.validate().is_err());
    }
}

#[test]
fn test_gumap_step_vs_run_bit_exact_f64_and_f32() {
    let grid = grid_graph(4, 4);
    let cfg = GuMapConfig {
        n_epochs: 25,
        seed: 99,
        ..Default::default()
    };
    let dog_cfg = DogVaradhanConfig::default();

    // f64 bit-exact parity
    let mut s_run = GuMapSolver::<f64>::from_graph_dog(&grid, 32, &dog_cfg, cfg).unwrap();
    let mut s_step = GuMapSolver::<f64>::from_graph_dog(&grid, 32, &dog_cfg, cfg).unwrap();
    let mut l_run = seeded_layout::<f64>(grid.num_nodes(), 1.0, 17);
    let mut l_step = l_run.clone();

    assert_eq!(s_run.status(), SolverStatus::Initialized);
    assert_eq!(s_run.total_iterations(), 25);
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
    let mut s32_run = GuMapSolver::<f32>::from_graph_dog(&grid, 32, &dog_cfg, cfg).unwrap();
    let mut s32_step = GuMapSolver::<f32>::from_graph_dog(&grid, 32, &dog_cfg, cfg).unwrap();
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
    let cfg = GuMapConfig {
        n_epochs: 150,
        negative_sample_rate: 5,
        learning_rate: 1.0,
        seed: 42,
        ..Default::default()
    };
    let dog_cfg = DogVaradhanConfig::default();
    let mut solver = GuMapSolver::<f64>::from_graph_dog(&barbell, 32, &dog_cfg, cfg).unwrap();
    assert!(solver.num_edges() > 0);
    let (a, b) = solver.ab();
    assert!(a > 0.0 && b > 0.0);

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
    let mut s_empty =
        GuMapSolver::<f64>::from_graph_dog(&empty, 16, &DogVaradhanConfig::default(), cfg).unwrap();
    let mut l_empty = Layout2d::<f64>::zeros(0);
    assert_eq!(s_empty.run(&mut l_empty).unwrap(), SolverStatus::Converged);

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let mut s_single =
        GuMapSolver::<f64>::from_graph_dog(&single, 16, &DogVaradhanConfig::default(), cfg)
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
    let cfg_zero = GuMapConfig { n_epochs: 0, ..cfg };
    let s_zero =
        GuMapSolver::<f64>::from_graph_dog(&grid, 16, &DogVaradhanConfig::default(), cfg_zero)
            .unwrap();
    assert_eq!(s_zero.status(), SolverStatus::Completed);
    assert_eq!(s_zero.total_iterations(), 0);

    // Dimension mismatch & non-finite layout rejected
    let mut s_grid = GuMapSolver::<f64>::from_knn(
        &KnnGraph::new_validated(2, vec![0, 1, 2], vec![1, 0], vec![1.0, 1.0]).unwrap(),
        cfg,
    )
    .unwrap();
    let mut l_wrong = Layout2d::<f64>::zeros(3);
    assert!(matches!(
        s_grid.step(&mut l_wrong),
        Err(CoreError::DimensionMismatch { .. })
    ));
    let mut l_nan = Layout2d::from_xy(vec![0.0, f64::NAN], vec![0.0, 1.0]).unwrap();
    assert!(matches!(
        s_grid.step(&mut l_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));

    // Invalid SparseProbabilities rejected
    let bad_probs = SparseProbabilities {
        n_nodes: 2,
        indptr: vec![0, 1, 1],
        indices: vec![2],
        values: vec![0.5],
    };
    assert!(matches!(
        GuMapSolver::<f64>::from_fuzzy_set(bad_probs, cfg),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
}

#[test]
fn test_gumap_from_fuzzy_set_interleaves_symmetric_pairs() {
    // Symmetric 4-node graph plus one asymmetric edge to confirm both cases work cleanly.
    let probs = SparseProbabilities {
        n_nodes: 4,
        indptr: vec![0, 2, 4, 6, 7],
        indices: vec![1, 3, 0, 2, 1, 3, 0],
        values: vec![0.8, 0.5, 0.8, 0.6, 0.6, 0.4, 0.5],
    };
    let solver = GuMapSolver::<f64>::from_fuzzy_set(probs, GuMapConfig::default()).unwrap();
    let got: Vec<(u32, u32)> = solver
        .edge_src
        .iter()
        .copied()
        .zip(solver.edge_tgt.iter().copied())
        .collect();
    assert_eq!(
        got,
        vec![(0, 1), (1, 0), (0, 3), (3, 0), (1, 2), (2, 1), (2, 3)]
    );
}

#[test]
fn test_gumap_from_landmarks_bit_exact_and_validation() {
    let grid = grid_graph(4, 4);
    let n = grid.num_nodes();
    let dog_cfg = DogVaradhanConfig::default();
    let cfg = GuMapConfig {
        n_epochs: 20,
        seed: 55,
        ..Default::default()
    };
    let (landmarks, _) = compute_regularized_dog_varadhan_landmarks(&grid, &dog_cfg).unwrap();

    let mut s_dog = GuMapSolver::<f64>::from_graph_dog(&grid, 32, &dog_cfg, cfg).unwrap();
    let mut s_lm = GuMapSolver::<f64>::from_landmarks(&grid, &landmarks, 32, cfg).unwrap();
    let mut l_dog = seeded_layout::<f64>(n, 1.0, 21);
    let mut l_lm = l_dog.clone();
    s_dog.run(&mut l_dog).unwrap();
    s_lm.run(&mut l_lm).unwrap();
    for i in 0..n {
        assert_eq!(l_dog.x()[i].to_bits(), l_lm.x()[i].to_bits());
        assert_eq!(l_dog.y()[i].to_bits(), l_lm.y()[i].to_bits());
    }

    // Validation errors
    let mut bad_nodes = landmarks.clone();
    bad_nodes.num_nodes = n + 1;
    assert!(matches!(
        GuMapSolver::<f64>::from_landmarks(&grid, &bad_nodes, 32, cfg),
        Err(CoreError::DimensionMismatch { .. })
    ));

    let mut bad_empty_lm = landmarks.clone();
    bad_empty_lm.landmarks.clear();
    bad_empty_lm.d_dog.clear();
    bad_empty_lm.d_coarse = None;
    assert!(matches!(
        GuMapSolver::<f64>::from_landmarks(&grid, &bad_empty_lm, 32, cfg),
        Err(CoreError::InvalidInput(_))
    ));

    let mut bad_nan = landmarks.clone();
    bad_nan.d_dog[0] = f64::NAN;
    assert!(matches!(
        GuMapSolver::<f64>::from_landmarks(&grid, &bad_nan, 32, cfg),
        Err(CoreError::NonFiniteValue { .. })
    ));

    let mut bad_idx = landmarks;
    bad_idx.landmarks[0] = n as u32;
    assert!(matches!(
        GuMapSolver::<f64>::from_landmarks(&grid, &bad_idx, 32, cfg),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
}

#[test]
fn test_gumap_symmetric_negative_updates_both_endpoints_and_step_parity() {
    // 3 nodes: edge (0, 1) with weight 1.0, and isolated node 2.
    // With high negative_sample_rate, node 2 is sampled as a negative example.
    let probs = SparseProbabilities {
        n_nodes: 3,
        indptr: vec![0, 1, 1, 1],
        indices: vec![1],
        values: vec![1.0],
    };
    let init = Layout2d::from_xy(vec![0.0, 1.0, 0.25], vec![0.0, 0.0, 0.25]).unwrap();

    let cfg_sym = GuMapConfig {
        n_epochs: 1,
        negative_sample_rate: 20,
        learning_rate: 1.0,
        symmetric_negative: true,
        seed: 42,
        ..Default::default()
    };
    let cfg_asym = GuMapConfig {
        symmetric_negative: false,
        ..cfg_sym
    };

    let mut l_sym = init.clone();
    let mut s_sym = GuMapSolver::<f64>::from_fuzzy_set(probs.clone(), cfg_sym).unwrap();
    s_sym.step(&mut l_sym).unwrap();
    let mut l_asym = init.clone();
    let mut s_asym = GuMapSolver::<f64>::from_fuzzy_set(probs, cfg_asym).unwrap();
    s_asym.step(&mut l_asym).unwrap();

    // When symmetric_negative is false, isolated node 2 never moves.
    assert_eq!(l_asym.x()[2], 0.25);
    assert_eq!(l_asym.y()[2], 0.25);
    // When symmetric_negative is true, isolated node 2 moves away from node 0.
    assert!(
        (l_sym.x()[2] - 0.25).hypot(l_sym.y()[2] - 0.25) > 1e-6,
        "expected symmetric negative update to move isolated negative sample 2"
    );

    // step == run bit-exactness (f64 and f32) for symmetric_negative in {true, false}.
    let grid = grid_graph(4, 4);
    let dog_cfg = DogVaradhanConfig::default();
    for sym in [true, false] {
        let cfg = GuMapConfig {
            n_epochs: 15,
            symmetric_negative: sym,
            seed: 77,
            ..Default::default()
        };
        let mut s64_r = GuMapSolver::<f64>::from_graph_dog(&grid, 32, &dog_cfg, cfg).unwrap();
        let mut s64_s = GuMapSolver::<f64>::from_graph_dog(&grid, 32, &dog_cfg, cfg).unwrap();
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

        let mut s32_r = GuMapSolver::<f32>::from_graph_dog(&grid, 32, &dog_cfg, cfg).unwrap();
        let mut s32_s = GuMapSolver::<f32>::from_graph_dog(&grid, 32, &dog_cfg, cfg).unwrap();
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
