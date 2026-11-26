use egraph_core::{CoreError, CsrGraph, IterativeSolver, Layout2d, SolverStatus};
use egraph_neighborhood::{
    SparseProbabilities, SymmetrizationMode, TwoHopSamplingMode, build_pure_2hop_knn,
    compute_knn_probabilities, compute_nodewise_2hop_perplexities,
    graph_12hop_row_normalized_probabilities_with_options,
};
use egraph_spatial_fft::{IbFftConfig, InterpolationOrder};

use super::super::{TsNetBackend, TsNetConfig, TsNetSolver};
use super::{make_grid_graph, seeded_layout};

#[test]
fn test_tsnet_config_pure_2hop_defaults_and_validation() {
    let cfg = TsNetConfig::pure_2hop();
    assert_eq!(cfg, TsNetConfig::default());
    assert_eq!(cfg.backend, TsNetBackend::FftInterpolation);
    assert_eq!(cfg.perplexity_c, 0.9);
    assert_eq!(cfg.min_perplexity, 5.0);
    assert_eq!(cfg.max_perplexity, 1200.0);
    assert_eq!(cfg.power, 2.0);
    assert_eq!(cfg.sigma_iters, 50);
    assert_eq!(cfg.sigma_tolerance, 1e-5);
    assert_eq!(cfg.symmetrization, SymmetrizationMode::StandardTsne);
    assert_eq!(cfg.stage_iterations, [250, 250, 500]);
    assert_eq!(cfg.exaggeration, 4.0);
    assert_eq!(cfg.lambda_c_stage1, 0.1);
    assert_eq!(cfg.lambda_c_stage2, 0.01);
    assert_eq!(cfg.lambda_c_stage3, 0.0);
    assert_eq!(cfg.lambda_r_stage3, 0.0);
    assert_eq!(cfg.learning_rate, None);
    assert_eq!(cfg.learning_rate_divisor, 3.0);
    assert!(cfg.jacobi_precondition);
    assert_eq!(cfg.resolve_learning_rate(16), 10.0);
    assert_eq!(cfg.resolve_learning_rate(600), 200.0);
    assert!((cfg.resolve_learning_rate(3000) - 1000.0).abs() < 1e-12);
    assert_eq!(
        TsNetConfig {
            learning_rate: Some(42.0),
            ..cfg
        }
        .resolve_learning_rate(3000),
        42.0
    );
    assert_eq!(cfg.momentum, 0.8);
    assert_eq!(cfg.epsilon_r, 0.05);
    assert_eq!(cfg.theta, 0.5);
    assert_eq!(cfg.ibfft.order, InterpolationOrder::Cubic);
    assert_eq!(cfg.interpolation_order(), InterpolationOrder::Cubic);
    assert_eq!(cfg.intervals(), 25);
    assert_eq!(cfg.min_grid_dim(), 64);
    assert_eq!(cfg.max_grid_dim(), 512);
    assert_eq!(cfg.seed, 42);
    cfg.validate().unwrap();

    let invalid_cases = [
        TsNetConfig {
            perplexity_c: 0.0,
            ..cfg
        },
        TsNetConfig {
            min_perplexity: -1.0,
            ..cfg
        },
        TsNetConfig {
            min_perplexity: 10.0,
            max_perplexity: 5.0,
            ..cfg
        },
        TsNetConfig { power: 0.0, ..cfg },
        TsNetConfig {
            sigma_iters: 0,
            ..cfg
        },
        TsNetConfig {
            sigma_tolerance: f64::NAN,
            ..cfg
        },
        TsNetConfig {
            exaggeration: 0.5,
            ..cfg
        },
        TsNetConfig {
            lambda_c_stage1: -0.1,
            ..cfg
        },
        TsNetConfig {
            lambda_c_stage2: -0.1,
            ..cfg
        },
        TsNetConfig {
            lambda_c_stage3: -0.1,
            ..cfg
        },
        TsNetConfig {
            lambda_r_stage3: f64::INFINITY,
            ..cfg
        },
        TsNetConfig {
            learning_rate: Some(0.0),
            ..cfg
        },
        TsNetConfig {
            learning_rate: Some(f64::NAN),
            ..cfg
        },
        TsNetConfig {
            learning_rate_divisor: 0.0,
            ..cfg
        },
        TsNetConfig {
            learning_rate_divisor: -2.0,
            ..cfg
        },
        TsNetConfig {
            learning_rate_divisor: f64::NAN,
            ..cfg
        },
        TsNetConfig {
            momentum: 1.0,
            ..cfg
        },
        TsNetConfig {
            momentum: -0.1,
            ..cfg
        },
        TsNetConfig {
            epsilon_r: 0.0,
            ..cfg
        },
        TsNetConfig { theta: -0.1, ..cfg },
        TsNetConfig {
            ibfft: IbFftConfig {
                intervals: 0,
                ..cfg.ibfft
            },
            ..cfg
        },
    ];
    for bad in invalid_cases {
        assert!(bad.validate().is_err(), "expected invalid config: {bad:?}");
    }
}

#[test]
fn test_constructors_from_knn_probabilities_and_partial_bfs() {
    let grid = make_grid_graph(3, 3);
    let (knn, _k2_deg, _stats) = build_pure_2hop_knn(&grid).unwrap();
    let perps = compute_nodewise_2hop_perplexities(&grid, 0.9, 5.0, 1200.0).unwrap();
    let cfg = TsNetConfig {
        stage_iterations: [5, 5, 5],
        ..TsNetConfig::pure_2hop()
    };

    let probs = compute_knn_probabilities(&knn, &perps, &cfg.perplexity_config()).unwrap();
    let mut s_probs = TsNetSolver::<f64>::from_probabilities(probs, cfg).unwrap();
    let mut s_knn = TsNetSolver::<f64>::from_knn(&knn, Some(&perps), 1.0, cfg).unwrap();
    let mut s_graph = TsNetSolver::<f64>::new(&grid, cfg).unwrap();

    let mut l1 = seeded_layout::<f64>(9, 0.5, 77);
    let mut l2 = l1.clone();
    let mut l3 = l1.clone();
    s_probs.run(&mut l1).unwrap();
    s_knn.run(&mut l2).unwrap();
    s_graph.run(&mut l3).unwrap();
    for i in 0..9 {
        assert_eq!(l1.x()[i].to_bits(), l2.x()[i].to_bits());
        assert_eq!(l1.y()[i].to_bits(), l2.y()[i].to_bits());
        assert_eq!(l1.x()[i].to_bits(), l3.x()[i].to_bits());
        assert_eq!(l1.y()[i].to_bits(), l3.y()[i].to_bits());
    }

    // Uniform perplexity via from_knn(..., None, ...) and from_graph_partial_bfs.
    let mut s_uni = TsNetSolver::<f64>::from_knn(&knn, None, 4.0, cfg).unwrap();
    let mut s_bfs = TsNetSolver::<f64>::from_graph_partial_bfs(&grid, 6, 4.0, cfg).unwrap();
    let mut l_uni = seeded_layout::<f64>(9, 0.5, 77);
    let mut l_bfs = l_uni.clone();
    s_uni.run(&mut l_uni).unwrap();
    s_bfs.run(&mut l_bfs).unwrap();
    assert!(l_uni.is_all_finite());
    assert!(l_bfs.is_all_finite());
}

#[test]
fn test_edge_cases_empty_singleton_two_node_coincident_and_errors() {
    for backend in [
        TsNetBackend::Exact,
        TsNetBackend::BarnesHut,
        TsNetBackend::FftInterpolation,
        TsNetBackend::LinearFft,
    ] {
        let cfg = TsNetConfig {
            backend,
            stage_iterations: [5, 5, 5],
            lambda_r_stage3: 0.6,
            ..TsNetConfig::pure_2hop()
        };

        // 0-node graph
        let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let mut s0 = TsNetSolver::<f64>::from_graph_pure_2hop(&g0, cfg).unwrap();
        let mut l0 = Layout2d::<f64>::zeros(0);
        assert_eq!(s0.run(&mut l0).unwrap(), SolverStatus::Converged);

        // 1-node graph
        let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
        let mut s1 = TsNetSolver::<f64>::from_graph_pure_2hop(&g1, cfg).unwrap();
        let mut l1 = Layout2d::from_xy(vec![3.5], vec![-2.0]).unwrap();
        assert_eq!(s1.run(&mut l1).unwrap(), SolverStatus::Converged);
        assert_eq!(l1.x(), &[3.5]);
        assert_eq!(l1.y(), &[-2.0]);

        // 2-node graph + coincident coordinates
        let g2 = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
        let mut s2 = TsNetSolver::<f64>::from_graph_pure_2hop(&g2, cfg).unwrap();
        let mut l2 = Layout2d::from_xy(vec![0.0, 0.0], vec![0.0, 0.0]).unwrap();
        assert_eq!(s2.run(&mut l2).unwrap(), SolverStatus::Completed);
        assert!(l2.is_all_finite());
    }

    // Zero total iterations finishes immediately.
    let g2 = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
    let cfg_zero = TsNetConfig {
        stage_iterations: [0, 0, 0],
        ..TsNetConfig::pure_2hop()
    };
    let mut s_zero = TsNetSolver::<f64>::from_graph_pure_2hop(&g2, cfg_zero).unwrap();
    assert_eq!(s_zero.status(), SolverStatus::Completed);
    assert_eq!(s_zero.total_iterations(), 0);
    let mut l2 = Layout2d::from_xy(vec![1.0, -1.0], vec![0.0, 0.0]).unwrap();
    assert_eq!(s_zero.step(&mut l2).unwrap(), SolverStatus::Completed);

    // Non-finite layout and mismatched dimensions rejected on step().
    let mut s_err =
        TsNetSolver::<f64>::from_graph_pure_2hop(&g2, TsNetConfig::pure_2hop()).unwrap();
    let mut l_nan = Layout2d::from_xy(vec![f64::NAN, 0.0], vec![0.0, 1.0]).unwrap();
    assert!(matches!(
        s_err.step(&mut l_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));
    let mut l_inf = Layout2d::from_xy(vec![0.0, f64::INFINITY], vec![0.0, 1.0]).unwrap();
    assert!(matches!(
        s_err.step(&mut l_inf),
        Err(CoreError::NonFiniteValue { .. })
    ));
    let mut l_mismatch = Layout2d::<f64>::zeros(3);
    assert!(matches!(
        s_err.step(&mut l_mismatch),
        Err(CoreError::DimensionMismatch { .. })
    ));

    // Invalid uniform_perplexity on from_knn.
    let (knn, _, _) = build_pure_2hop_knn(&g2).unwrap();
    assert!(matches!(
        TsNetSolver::<f64>::from_knn(&knn, None, 0.0, TsNetConfig::pure_2hop()),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        TsNetSolver::<f64>::from_knn(&knn, None, f64::NAN, TsNetConfig::pure_2hop()),
        Err(CoreError::NonFiniteValue { .. })
    ));

    // Invalid SparseProbabilities rejected.
    let bad_probs = SparseProbabilities {
        n_nodes: 2,
        indptr: vec![0, 1, 1],
        indices: vec![2],
        values: vec![0.5],
    };
    assert!(matches!(
        TsNetSolver::<f64>::from_probabilities(bad_probs, TsNetConfig::pure_2hop()),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    let bad_val_probs = SparseProbabilities {
        n_nodes: 2,
        indptr: vec![0, 1, 2],
        indices: vec![1, 0],
        values: vec![0.5, -0.1],
    };
    assert!(matches!(
        TsNetSolver::<f64>::from_probabilities(bad_val_probs, TsNetConfig::pure_2hop()),
        Err(CoreError::InvalidInput(_))
    ));
}

#[test]
fn test_from_graph_row_normalized_uncapped_capped_and_bit_exact() {
    let grid = make_grid_graph(4, 4);
    let n = grid.num_nodes();
    let cfg = TsNetConfig {
        stage_iterations: [8, 8, 8],
        seed: 91,
        ..TsNetConfig::pure_2hop()
    };

    for max_2hop in [None, Some(4usize)] {
        let probs = graph_12hop_row_normalized_probabilities_with_options(
            &grid,
            0.5,
            max_2hop,
            0.0,
            TwoHopSamplingMode::TwoStepRandomWalk,
            None,
            cfg.seed,
        )
        .unwrap();
        let mut s_ref = TsNetSolver::<f64>::from_probabilities(probs, cfg).unwrap();
        let mut s_run =
            TsNetSolver::<f64>::from_graph_row_normalized(&grid, 0.5, max_2hop, cfg).unwrap();
        let mut s_step =
            TsNetSolver::<f64>::from_graph_row_normalized(&grid, 0.5, max_2hop, cfg).unwrap();

        let mut l_ref = seeded_layout::<f64>(n, 0.5, 29);
        let mut l_run = l_ref.clone();
        let mut l_step = l_ref.clone();

        s_ref.run(&mut l_ref).unwrap();
        s_run.run(&mut l_run).unwrap();
        s_step.run_steps(&mut l_step, 10).unwrap();
        while !s_step.status().is_finished() {
            s_step.step(&mut l_step).unwrap();
        }
        for i in 0..n {
            assert_eq!(l_ref.x()[i].to_bits(), l_run.x()[i].to_bits());
            assert_eq!(l_ref.y()[i].to_bits(), l_run.y()[i].to_bits());
            assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
            assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
        }

        // f32 bit-exact step == run
        let mut s32_run =
            TsNetSolver::<f32>::from_graph_row_normalized(&grid, 0.5, max_2hop, cfg).unwrap();
        let mut s32_step =
            TsNetSolver::<f32>::from_graph_row_normalized(&grid, 0.5, max_2hop, cfg).unwrap();
        let mut l32_run = seeded_layout::<f32>(n, 0.5, 29);
        let mut l32_step = l32_run.clone();
        s32_run.run(&mut l32_run).unwrap();
        for _ in 0..24 {
            s32_step.step(&mut l32_step).unwrap();
        }
        for i in 0..n {
            assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
            assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
        }
    }

    // Validation & edge cases
    assert!(matches!(
        TsNetSolver::<f64>::from_graph_row_normalized(&grid, -0.5, None, cfg),
        Err(CoreError::InvalidInput(_))
    ));
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let mut s_empty =
        TsNetSolver::<f64>::from_graph_row_normalized(&empty, 0.5, None, cfg).unwrap();
    let mut l_empty = Layout2d::<f64>::zeros(0);
    assert_eq!(s_empty.run(&mut l_empty).unwrap(), SolverStatus::Converged);

    // Degree-decayed row-normalized solver matches standalone probabilities (TwoStepRandomWalk default)
    let probs_beta = graph_12hop_row_normalized_probabilities_with_options(
        &grid,
        0.5,
        Some(4),
        0.35,
        TwoHopSamplingMode::TwoStepRandomWalk,
        None,
        cfg.seed,
    )
    .unwrap();
    let mut s_prob_beta = TsNetSolver::<f64>::from_probabilities(probs_beta, cfg).unwrap();
    let mut s_decay = TsNetSolver::<f64>::from_graph_row_normalized_with_degree_decay(
        &grid,
        0.5,
        Some(4),
        0.35,
        cfg,
    )
    .unwrap();
    let mut l_p = seeded_layout::<f64>(n, 0.5, 29);
    let mut l_d = l_p.clone();
    s_prob_beta.run(&mut l_p).unwrap();
    s_decay.run(&mut l_d).unwrap();
    for i in 0..n {
        assert_eq!(l_p.x()[i].to_bits(), l_d.x()[i].to_bits());
        assert_eq!(l_p.y()[i].to_bits(), l_d.y()[i].to_bits());
    }
    // Explicit DegreeOrderedHubSuppressed with max_bridge_degree cap via from_graph_row_normalized_with_options.
    let probs_hub_sup = graph_12hop_row_normalized_probabilities_with_options(
        &grid,
        0.5,
        Some(4),
        0.0,
        TwoHopSamplingMode::DegreeOrderedHubSuppressed,
        Some(3),
        cfg.seed,
    )
    .unwrap();
    let mut s_opt_ref = TsNetSolver::<f64>::from_probabilities(probs_hub_sup, cfg).unwrap();
    let mut s_opt = TsNetSolver::<f64>::from_graph_row_normalized_with_options(
        &grid,
        0.5,
        Some(4),
        0.0,
        TwoHopSamplingMode::DegreeOrderedHubSuppressed,
        Some(3),
        cfg,
    )
    .unwrap();
    let mut l_or = seeded_layout::<f64>(n, 0.5, 29);
    let mut l_o = l_or.clone();
    s_opt_ref.run(&mut l_or).unwrap();
    s_opt.run(&mut l_o).unwrap();
    for i in 0..n {
        assert_eq!(l_or.x()[i].to_bits(), l_o.x()[i].to_bits());
        assert_eq!(l_or.y()[i].to_bits(), l_o.y()[i].to_bits());
    }
    assert!(matches!(
        TsNetSolver::<f64>::from_graph_row_normalized_with_degree_decay(
            &grid, 0.5, None, -0.1, cfg
        ),
        Err(CoreError::InvalidInput(_))
    ));
}

#[test]
fn test_tsnet_backend_auto_resolution_at_256_threshold() {
    let g_small = make_grid_graph(16, 16); // N = 256 -> Exact
    let g_large = make_grid_graph(17, 16); // N = 272 -> LinearFft (Bilinear p=1)
    let cfg = TsNetConfig {
        backend: TsNetBackend::Auto,
        stage_iterations: [1, 0, 0],
        ..TsNetConfig::pure_2hop()
    };

    let mut s_small = TsNetSolver::<f64>::new(&g_small, cfg).unwrap();
    assert_eq!(s_small.config().backend, TsNetBackend::Auto);
    assert_eq!(s_small.resolved_backend(), TsNetBackend::Exact);
    let mut l_small = seeded_layout::<f64>(256, 0.5, 11);
    s_small.run(&mut l_small).unwrap();
    assert_eq!(s_small.total_fft_2d_calls(), 0);

    let mut s_large = TsNetSolver::<f64>::new(&g_large, cfg).unwrap();
    assert_eq!(s_large.config().backend, TsNetBackend::Auto);
    assert_eq!(s_large.resolved_backend(), TsNetBackend::LinearFft);
    assert_eq!(
        s_large.fit_field.as_ref().unwrap().config().ibfft.order,
        InterpolationOrder::Bilinear
    );
    let mut l_large = seeded_layout::<f64>(272, 0.5, 11);
    s_large.run(&mut l_large).unwrap();
    assert!(s_large.total_fft_2d_calls() > 0);
}
