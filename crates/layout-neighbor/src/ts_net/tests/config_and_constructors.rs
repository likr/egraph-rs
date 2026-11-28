use egraph_core::{CoreError, CsrGraph, IterativeSolver, Layout2d, SolverStatus};
use egraph_neighborhood::{
    SparseProbabilities, SymmetrizationMode, TwoHopSamplingMode, build_pure_2hop_knn,
    compute_knn_probabilities, compute_nodewise_2hop_perplexities,
    graph_12hop_row_normalized_probabilities_with_options,
};
use egraph_spatial_fft::{IbFftConfig, InterpolationOrder};

use super::super::{TsNetBackend, TsNetConfig, TsNetSolver, TsNetStage};
use super::{make_grid_graph, make_stages, seeded_layout};

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
    assert_eq!(cfg.stages().len(), 3);
    assert_eq!(cfg.stages()[0].iterations, 250);
    assert_eq!(cfg.stages()[0].exaggeration, 4.0);
    assert_eq!(cfg.stages()[0].lambda_c, 0.1);
    assert_eq!(cfg.stages()[1].iterations, 250);
    assert_eq!(cfg.stages()[1].lambda_c, 0.01);
    assert_eq!(cfg.stages()[2].iterations, 500);
    assert_eq!(cfg.stages()[2].lambda_c, 0.0);
    assert_eq!(cfg.stages()[2].lambda_r, 0.0);
    assert_eq!(cfg.learning_rate, None);
    assert_eq!(cfg.learning_rate_divisor, 3.0);
    assert!(cfg.jacobi_precondition);
    assert_eq!(cfg.resolve_learning_rate(16), 10.0);
    assert_eq!(cfg.resolve_learning_rate(600), 200.0);
    assert!((cfg.resolve_learning_rate(3000) - 1000.0).abs() < 1e-12);
    assert_eq!(
        cfg.clone()
            .with_learning_rate(Some(42.0))
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
        cfg.clone().with_perplexity_c(0.0),
        cfg.clone().with_min_perplexity(-1.0),
        cfg.clone()
            .with_min_perplexity(10.0)
            .with_max_perplexity(5.0),
        cfg.clone().with_power(0.0),
        cfg.clone().with_sigma_iters(0),
        cfg.clone().with_sigma_tolerance(f64::NAN),
        cfg.clone()
            .with_stages(vec![TsNetStage::new().with_exaggeration(-0.5)]),
        cfg.clone()
            .with_stages(vec![TsNetStage::new().with_lambda_c(-0.1)]),
        cfg.clone()
            .with_stages(vec![TsNetStage::new().with_lambda_r(f64::INFINITY)]),
        cfg.clone().with_learning_rate(Some(0.0)),
        cfg.clone().with_learning_rate(Some(f64::NAN)),
        cfg.clone().with_learning_rate_divisor(0.0),
        cfg.clone().with_learning_rate_divisor(-2.0),
        cfg.clone().with_learning_rate_divisor(f64::NAN),
        cfg.clone().with_momentum(1.0),
        cfg.clone().with_momentum(-0.1),
        cfg.clone().with_epsilon_r(0.0),
        cfg.clone().with_theta(-0.1),
        cfg.with_ibfft(IbFftConfig::default().with_intervals(0)),
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
    let cfg = TsNetConfig::pure_2hop().with_stages(make_stages(5, 5, 5, 0.0));

    let probs = compute_knn_probabilities(&knn, &perps, &cfg.perplexity_config()).unwrap();
    let mut s_probs = TsNetSolver::<Layout2d<f64>>::from_probabilities(probs, cfg.clone()).unwrap();
    let mut s_knn =
        TsNetSolver::<Layout2d<f64>>::from_knn(&knn, Some(&perps), 1.0, cfg.clone()).unwrap();
    let mut s_graph = TsNetSolver::<Layout2d<f64>>::new(&grid, cfg.clone()).unwrap();

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

    // Unweighted CsrGraph<()> works directly without W: Scalar bounds.
    let grid_unit =
        CsrGraph::<()>::from_unweighted_csr(grid.indptr(), grid.indices(), false, false).unwrap();
    let mut s_unit = TsNetSolver::<Layout2d<f64>>::new(&grid_unit, cfg.clone()).unwrap();
    let mut l_unit = seeded_layout::<f64>(9, 0.5, 77);
    s_unit.run(&mut l_unit).unwrap();
    for i in 0..9 {
        assert_eq!(l_unit.x()[i].to_bits(), l3.x()[i].to_bits());
        assert_eq!(l_unit.y()[i].to_bits(), l3.y()[i].to_bits());
    }

    // Uniform perplexity via from_knn(..., None, ...) and from_graph_partial_bfs.
    let mut s_uni = TsNetSolver::<Layout2d<f64>>::from_knn(&knn, None, 4.0, cfg.clone()).unwrap();
    let mut s_bfs =
        TsNetSolver::<Layout2d<f64>>::from_graph_partial_bfs(&grid_unit, 6, 4.0, cfg).unwrap();
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
        let cfg = TsNetConfig::pure_2hop()
            .with_backend(backend)
            .with_stages(make_stages(5, 5, 5, 0.6));

        let g0 = CsrGraph::<()>::from_unweighted_edges(0, &[], false, false).unwrap();
        let mut s0 = TsNetSolver::<Layout2d<f64>>::from_graph_pure_2hop(&g0, cfg.clone()).unwrap();
        let mut l0 = Layout2d::<f64>::zeros(0);
        assert_eq!(s0.run(&mut l0).unwrap(), SolverStatus::Converged);

        let g1 = CsrGraph::<()>::from_unweighted_edges(1, &[], false, false).unwrap();
        let mut s1 = TsNetSolver::<Layout2d<f64>>::from_graph_pure_2hop(&g1, cfg.clone()).unwrap();
        let mut l1 = Layout2d::from_xy(vec![3.5], vec![-2.0]).unwrap();
        assert_eq!(s1.run(&mut l1).unwrap(), SolverStatus::Converged);
        assert_eq!(l1.x(), &[3.5]);
        assert_eq!(l1.y(), &[-2.0]);

        let g2 = CsrGraph::<()>::from_unweighted_edges(2, &[(0, 1)], false, false).unwrap();
        let mut s2 = TsNetSolver::<Layout2d<f64>>::from_graph_pure_2hop(&g2, cfg).unwrap();
        let mut l2 = Layout2d::from_xy(vec![0.0, 0.0], vec![0.0, 0.0]).unwrap();
        assert_eq!(s2.run(&mut l2).unwrap(), SolverStatus::Completed);
        assert!(l2.is_all_finite());
    }

    // Zero total iterations finishes immediately and validates run_steps(target, 0).
    let g2 = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
    let cfg_zero = TsNetConfig::pure_2hop().with_stages(make_stages(0, 0, 0, 0.0));
    let mut s_zero = TsNetSolver::<Layout2d<f64>>::from_graph_pure_2hop(&g2, cfg_zero).unwrap();
    assert_eq!(s_zero.status(), SolverStatus::Completed);
    assert_eq!(s_zero.total_iterations(), Some(0));
    let mut l2 = Layout2d::from_xy(vec![1.0, -1.0], vec![0.0, 0.0]).unwrap();
    assert_eq!(s_zero.step(&mut l2).unwrap(), SolverStatus::Completed);
    let mut l_bad = Layout2d::<f64>::zeros(3);
    assert!(matches!(
        s_zero.run_steps(&mut l_bad, 0),
        Err(CoreError::DimensionMismatch { .. })
    ));

    // Non-finite layout and mismatched dimensions rejected on step().
    let mut s_err =
        TsNetSolver::<Layout2d<f64>>::from_graph_pure_2hop(&g2, TsNetConfig::pure_2hop()).unwrap();
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
        TsNetSolver::<Layout2d<f64>>::from_knn(&knn, None, 0.0, TsNetConfig::pure_2hop()),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        TsNetSolver::<Layout2d<f64>>::from_knn(&knn, None, f64::NAN, TsNetConfig::pure_2hop()),
        Err(CoreError::NonFiniteValue { .. })
    ));

    // Invalid SparseProbabilities rejected by SparseProbabilities::from_csr.
    assert!(matches!(
        SparseProbabilities::from_csr(2, vec![0, 1, 1], vec![2], vec![0.5]),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        SparseProbabilities::from_csr(2, vec![0, 1, 2], vec![1, 0], vec![0.5, -0.1]),
        Err(CoreError::InvalidInput(_))
    ));
}

#[test]
fn test_from_graph_row_normalized_uncapped_capped_and_bit_exact() {
    let grid = make_grid_graph(4, 4);
    let n = grid.num_nodes();
    let cfg = TsNetConfig::pure_2hop()
        .with_stages(make_stages(8, 8, 8, 0.0))
        .with_seed(91);

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
        let mut s_ref =
            TsNetSolver::<Layout2d<f64>>::from_probabilities(probs, cfg.clone()).unwrap();
        let mut s_run = TsNetSolver::<Layout2d<f64>>::from_graph_row_normalized(
            &grid,
            0.5,
            max_2hop,
            cfg.clone(),
        )
        .unwrap();
        let mut s_step = TsNetSolver::<Layout2d<f64>>::from_graph_row_normalized(
            &grid,
            0.5,
            max_2hop,
            cfg.clone(),
        )
        .unwrap();

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

        let mut s32_run = TsNetSolver::<Layout2d<f32>>::from_graph_row_normalized(
            &grid,
            0.5,
            max_2hop,
            cfg.clone(),
        )
        .unwrap();
        let mut s32_step = TsNetSolver::<Layout2d<f32>>::from_graph_row_normalized(
            &grid,
            0.5,
            max_2hop,
            cfg.clone(),
        )
        .unwrap();
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

    assert!(matches!(
        TsNetSolver::<Layout2d<f64>>::from_graph_row_normalized(&grid, -0.5, None, cfg.clone()),
        Err(CoreError::InvalidInput(_))
    ));
    let empty = CsrGraph::<()>::from_unweighted_edges(0, &[], false, false).unwrap();
    let mut s_empty =
        TsNetSolver::<Layout2d<f64>>::from_graph_row_normalized(&empty, 0.5, None, cfg.clone())
            .unwrap();
    let mut l_empty = Layout2d::<f64>::zeros(0);
    assert_eq!(s_empty.run(&mut l_empty).unwrap(), SolverStatus::Converged);

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
    let mut s_prob_beta =
        TsNetSolver::<Layout2d<f64>>::from_probabilities(probs_beta, cfg.clone()).unwrap();
    let mut s_decay = TsNetSolver::<Layout2d<f64>>::from_graph_row_normalized_with_degree_decay(
        &grid,
        0.5,
        Some(4),
        0.35,
        cfg.clone(),
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
    let mut s_opt_ref =
        TsNetSolver::<Layout2d<f64>>::from_probabilities(probs_hub_sup, cfg.clone()).unwrap();
    let mut s_opt = TsNetSolver::<Layout2d<f64>>::from_graph_row_normalized_with_options(
        &grid,
        0.5,
        Some(4),
        0.0,
        TwoHopSamplingMode::DegreeOrderedHubSuppressed,
        Some(3),
        cfg.clone(),
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
        TsNetSolver::<Layout2d<f64>>::from_graph_row_normalized_with_degree_decay(
            &grid, 0.5, None, -0.1, cfg
        ),
        Err(CoreError::InvalidInput(_))
    ));
}

#[test]
fn test_tsnet_backend_auto_resolution_at_256_threshold() {
    let g_small = make_grid_graph(16, 16); // N = 256 -> Exact
    let g_large = make_grid_graph(17, 16); // N = 272 -> LinearFft (Bilinear p=1)
    let cfg = TsNetConfig::pure_2hop()
        .with_backend(TsNetBackend::Auto)
        .with_stages(make_stages(1, 0, 0, 0.0));

    let mut s_small = TsNetSolver::<Layout2d<f64>>::new(&g_small, cfg.clone()).unwrap();
    assert_eq!(s_small.config().backend, TsNetBackend::Auto);
    assert_eq!(s_small.resolved_backend(), TsNetBackend::Exact);
    let mut l_small = seeded_layout::<f64>(256, 0.5, 11);
    s_small.run(&mut l_small).unwrap();
    assert_eq!(s_small.total_fft_2d_calls(), 0);

    let mut s_large = TsNetSolver::<Layout2d<f64>>::new(&g_large, cfg).unwrap();
    assert_eq!(s_large.config().backend, TsNetBackend::Auto);
    assert_eq!(s_large.resolved_backend(), TsNetBackend::LinearFft);
    assert_eq!(
        s_large
            .state
            .fit_field
            .as_ref()
            .unwrap()
            .config()
            .ibfft
            .order,
        InterpolationOrder::Bilinear
    );
    let mut l_large = seeded_layout::<f64>(272, 0.5, 11);
    s_large.run(&mut l_large).unwrap();
    assert!(s_large.total_fft_2d_calls() > 0);
}
