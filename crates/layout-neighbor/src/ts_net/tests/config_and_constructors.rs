use egraph_core::{CoreError, CsrGraph, IterativeSolver, Layout2d, Layout3d, SolverStatus};
use egraph_neighborhood::{
    SparseProbabilities, SymmetrizationMode, TwoHopSamplingMode, build_pure_2hop_knn,
    compute_knn_probabilities, compute_nodewise_2hop_perplexities,
    graph_12hop_row_normalized_probabilities_with_options,
};
use egraph_spatial_fft::{IbFftConfig, InterpolationOrder};

use crate::backends::{
    Auto2d, Auto3d, BarnesHut, Exact, FitFft, LinearFft, TsNetBackend2d, TsNetBackend3d,
};

use super::super::{TsNetConfig, TsNetSolver, TsNetStage};
use super::{make_grid_graph, make_stages, seeded_layout};

#[test]
fn test_tsnet_config_pure_2hop_defaults_and_backend_validation() {
    let cfg = TsNetConfig::pure_2hop();
    assert_eq!(cfg, TsNetConfig::default());
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
        cfg.with_epsilon_r(0.0),
    ];
    for bad in invalid_cases {
        assert!(bad.validate().is_err(), "expected invalid config: {bad:?}");
    }

    // Backend-specific parameter builders and validation.
    let bh = BarnesHut::default();
    assert_eq!(bh.theta(), 0.5);
    assert_eq!(BarnesHut::new(0.25).theta(), 0.25);
    assert_eq!(bh.clone().with_theta(0.75).theta(), 0.75);
    assert!(BarnesHut::new(-0.1).validate().is_err());
    assert!(BarnesHut::new(f64::NAN).validate().is_err());

    let fit = FitFft::new();
    assert_eq!(fit.interpolation_order(), InterpolationOrder::Cubic);
    assert_eq!(fit.intervals(), 25);
    assert_eq!(fit.min_grid_dim(), 64);
    assert_eq!(fit.max_grid_dim(), 512);
    assert!(
        fit.with_ibfft(IbFftConfig::default().with_intervals(0))
            .validate()
            .is_err()
    );

    let lin =
        LinearFft::new().with_ibfft(IbFftConfig::default().with_order(InterpolationOrder::Cubic));
    assert_eq!(lin.ibfft().order, InterpolationOrder::Bilinear);
    lin.validate().unwrap();

    let a2 = Auto2d::new().with_exact_threshold(128);
    assert_eq!(a2.exact_threshold(), 128);
    a2.validate().unwrap();

    let a3 = Auto3d::new().with_exact_threshold(64).with_theta(0.4);
    assert_eq!(a3.exact_threshold(), 64);
    assert_eq!(a3.theta(), 0.4);
    a3.validate().unwrap();
    assert!(Auto3d::new().with_theta(-0.1).validate().is_err());
}

#[test]
fn test_constructors_from_knn_probabilities_and_partial_bfs() {
    let grid = make_grid_graph(3, 3);
    let (knn, _k2_deg, _stats) = build_pure_2hop_knn(&grid).unwrap();
    let perps = compute_nodewise_2hop_perplexities(&grid, 0.9, 5.0, 1200.0).unwrap();
    let cfg = TsNetConfig::pure_2hop().with_stages(make_stages(5, 5, 5, 0.0));

    let probs = compute_knn_probabilities(&knn, &perps, &cfg.perplexity_config()).unwrap();
    let mut s_probs =
        TsNetSolver::<Layout2d<f64>, FitFft>::from_probabilities(probs, cfg.clone()).unwrap();
    let mut s_knn =
        TsNetSolver::<Layout2d<f64>, FitFft>::from_knn(&knn, Some(&perps), 1.0, cfg.clone())
            .unwrap();
    let mut s_graph = TsNetSolver::<Layout2d<f64>, FitFft>::new(&grid, cfg.clone()).unwrap();
    let mut s_p2h = TsNetSolver::<Layout2d<f64>, FitFft>::pure_2hop(&grid, cfg.clone()).unwrap();

    let mut l1 = seeded_layout::<f64>(9, 0.5, 77);
    let mut l2 = l1.clone();
    let mut l3 = l1.clone();
    let mut l4 = l1.clone();
    s_probs.run(&mut l1).unwrap();
    s_knn.run(&mut l2).unwrap();
    s_graph.run(&mut l3).unwrap();
    s_p2h.run(&mut l4).unwrap();
    for i in 0..9 {
        assert_eq!(l1.x()[i].to_bits(), l2.x()[i].to_bits());
        assert_eq!(l1.y()[i].to_bits(), l2.y()[i].to_bits());
        assert_eq!(l1.x()[i].to_bits(), l3.x()[i].to_bits());
        assert_eq!(l1.y()[i].to_bits(), l3.y()[i].to_bits());
        assert_eq!(l1.x()[i].to_bits(), l4.x()[i].to_bits());
        assert_eq!(l1.y()[i].to_bits(), l4.y()[i].to_bits());
    }

    // Unweighted CsrGraph<()> works directly without W: Scalar bounds.
    let grid_unit =
        CsrGraph::<()>::from_unweighted_csr(grid.indptr(), grid.indices(), false, false).unwrap();
    let mut s_unit = TsNetSolver::<Layout2d<f64>, FitFft>::new(&grid_unit, cfg.clone()).unwrap();
    let mut l_unit = seeded_layout::<f64>(9, 0.5, 77);
    s_unit.run(&mut l_unit).unwrap();
    for i in 0..9 {
        assert_eq!(l_unit.x()[i].to_bits(), l3.x()[i].to_bits());
        assert_eq!(l_unit.y()[i].to_bits(), l3.y()[i].to_bits());
    }

    // Uniform perplexity via from_knn(..., None, ...) and from_graph_partial_bfs.
    let mut s_uni = TsNetSolver::<Layout2d<f64>>::from_knn(&knn, None, 4.0, cfg.clone()).unwrap();
    let mut s_bfs =
        TsNetSolver::<Layout2d<f64>>::from_graph_partial_bfs(&grid_unit, 6, 4.0, cfg.clone())
            .unwrap();
    let mut l_uni = seeded_layout::<f64>(9, 0.5, 77);
    let mut l_bfs = l_uni.clone();
    s_uni.run(&mut l_uni).unwrap();
    s_bfs.run(&mut l_bfs).unwrap();
    assert!(l_uni.is_all_finite());
    assert!(l_bfs.is_all_finite());

    // `*_with_backend` constructors allow explicit configured backends like `BarnesHut::new(0.3)`.
    let mut s_bh_custom = TsNetSolver::<Layout2d<f64>, BarnesHut>::new_with_backend(
        &grid_unit,
        cfg,
        BarnesHut::new(0.3),
    )
    .unwrap();
    assert_eq!(s_bh_custom.backend().theta(), 0.3);
    let mut l_bh = seeded_layout::<f64>(9, 0.5, 77);
    s_bh_custom.run(&mut l_bh).unwrap();
    assert!(l_bh.is_all_finite());
}

#[test]
fn test_edge_cases_empty_singleton_two_node_coincident_and_errors() {
    for backend in [
        TsNetBackend2d::Exact(Exact::new()),
        TsNetBackend2d::BarnesHut(BarnesHut::default()),
        TsNetBackend2d::FitFft(FitFft::new()),
        TsNetBackend2d::LinearFft(LinearFft::new()),
    ] {
        let cfg = TsNetConfig::pure_2hop().with_stages(make_stages(5, 5, 5, 0.6));

        let g0 = CsrGraph::<()>::from_unweighted_edges(0, &[], false, false).unwrap();
        let mut s0 =
            TsNetSolver::<Layout2d<f64>, TsNetBackend2d>::from_graph_pure_2hop_with_backend(
                &g0,
                cfg.clone(),
                backend.clone(),
            )
            .unwrap();
        let mut l0 = Layout2d::<f64>::zeros(0);
        assert_eq!(s0.run(&mut l0).unwrap(), SolverStatus::Converged);

        let g1 = CsrGraph::<()>::from_unweighted_edges(1, &[], false, false).unwrap();
        let mut s1 =
            TsNetSolver::<Layout2d<f64>, TsNetBackend2d>::from_graph_pure_2hop_with_backend(
                &g1,
                cfg.clone(),
                backend.clone(),
            )
            .unwrap();
        let mut l1 = Layout2d::from_xy(vec![3.5], vec![-2.0]).unwrap();
        assert_eq!(s1.run(&mut l1).unwrap(), SolverStatus::Converged);
        assert_eq!(l1.x(), &[3.5]);
        assert_eq!(l1.y(), &[-2.0]);

        let g2 = CsrGraph::<()>::from_unweighted_edges(2, &[(0, 1)], false, false).unwrap();
        let mut s2 =
            TsNetSolver::<Layout2d<f64>, TsNetBackend2d>::from_graph_pure_2hop_with_backend(
                &g2, cfg, backend,
            )
            .unwrap();
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

    // Invalid backend rejected at construction.
    assert!(matches!(
        TsNetSolver::<Layout2d<f64>, BarnesHut>::new_with_backend(
            &g2,
            TsNetConfig::pure_2hop(),
            BarnesHut::new(-0.5),
        ),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        TsNetSolver::<Layout3d<f64>, TsNetBackend3d>::new_with_backend(
            &g2,
            TsNetConfig::pure_2hop(),
            TsNetBackend3d::Auto(Auto3d::new().with_theta(f64::NAN)),
        ),
        Err(CoreError::InvalidInput(_))
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
            TsNetSolver::<Layout2d<f64>, FitFft>::from_probabilities(probs, cfg.clone()).unwrap();
        let mut s_run = TsNetSolver::<Layout2d<f64>, FitFft>::from_graph_row_normalized(
            &grid,
            0.5,
            max_2hop,
            cfg.clone(),
        )
        .unwrap();
        let mut s_step = TsNetSolver::<Layout2d<f64>, FitFft>::from_graph_row_normalized(
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

        let mut s32_run = TsNetSolver::<Layout2d<f32>, FitFft>::from_graph_row_normalized(
            &grid,
            0.5,
            max_2hop,
            cfg.clone(),
        )
        .unwrap();
        let mut s32_step = TsNetSolver::<Layout2d<f32>, FitFft>::from_graph_row_normalized(
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

    let (e_ptr, e_idx, e_val) = s_empty.probabilities_csr();
    assert_eq!(e_ptr, &[0]);
    assert!(e_idx.is_empty());
    assert!(e_val.is_empty());

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
            &grid,
            0.5,
            None,
            -0.1,
            cfg.clone()
        ),
        Err(CoreError::InvalidInput(_))
    ));

    let mut s_ir = TsNetSolver::<Layout2d<f64>>::new(&grid, cfg).unwrap();
    let (indptr, indices, p_vals) = s_ir.probabilities_csr();
    assert_eq!(indptr.len(), n + 1);
    assert_eq!(indices.len(), indptr[n] as usize);
    assert_eq!(p_vals.len(), indices.len());
    let ptr = indptr.as_ptr();
    let mut l_ir = seeded_layout::<f64>(n, 0.5, 29);
    s_ir.step(&mut l_ir).unwrap();
    assert_eq!(s_ir.probabilities_csr().0.as_ptr(), ptr);
}
