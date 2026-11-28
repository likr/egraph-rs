use egraph_core::{CoreError, CsrGraph, IterativeSolver, Layout2d, SolverStatus};
use egraph_layout_sgd::{
    OrtmannPivotStrategy, OrtmannSparseProblem, OrtmannSparseStressConfig, SgdScheduleConfig,
    SgdSolver, SparseBarrierConfig, StreamingNegativeConfig, build_ortmann_sparse_problem,
};
use egraph_linalg::BfsWorkspace;
use egraph_neighborhood::DogVaradhanConfig;

fn make_grid_unit(rows: usize, cols: usize) -> CsrGraph<()> {
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
    CsrGraph::<()>::from_unweighted_edges(n, &edges, false, false).unwrap()
}

fn make_grid_f64(rows: usize, cols: usize) -> CsrGraph<f64> {
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
    CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap()
}

#[test]
fn sgd_solver_ir_getters_num_pairs_and_streaming_negatives() {
    let g = make_grid_unit(2, 3);
    let sched = SgdScheduleConfig::new(8);

    let full = SgdSolver::<Layout2d<f64>>::full(&g, sched.clone(), 42).unwrap();
    assert_eq!(full.num_pairs(), full.num_springs() + full.num_hinges());
    assert_eq!(full.num_pairs(), 15);
    assert!(full.streaming_negative_adjacency().is_none());
    assert!(full.streaming_negative_config().is_none());

    let neg_cfg = StreamingNegativeConfig::new()
        .with_negatives_per_node(4)
        .with_target_distance(2.5)
        .with_weight(0.125)
        .with_hinge_only(false);
    let stream =
        SgdSolver::<Layout2d<f64>>::streaming_negatives(&g, neg_cfg.clone(), sched.clone(), 42)
            .unwrap();
    assert_eq!(stream.num_pairs(), g.num_edges());
    let (indptr, indices) = stream.streaming_negative_adjacency().unwrap();
    assert_eq!(indptr.len(), g.num_nodes() + 1);
    assert_eq!(indices.len(), 2 * g.num_edges());
    assert_eq!(stream.streaming_negative_config(), Some(neg_cfg));

    // When negatives_per_node == 0, streaming state is inactive.
    let zero_stream = SgdSolver::<Layout2d<f64>>::streaming_negatives(
        &g,
        StreamingNegativeConfig::new().with_negatives_per_node(0),
        sched,
        42,
    )
    .unwrap();
    assert!(zero_stream.streaming_negative_adjacency().is_none());
    assert!(zero_stream.streaming_negative_config().is_none());
}

#[test]
fn csr_graph_unit_supported_across_all_sgd_constructors() {
    let g_unit = make_grid_unit(3, 4);
    let g_f64 = make_grid_f64(3, 4);
    let sched = SgdScheduleConfig::new(10);
    let n = g_unit.num_nodes();

    let mut s_full_u = SgdSolver::<Layout2d<f64>>::full(&g_unit, sched.clone(), 7).unwrap();
    let mut s_full_f = SgdSolver::<Layout2d<f64>>::full(&g_f64, sched.clone(), 7).unwrap();
    assert_eq!(s_full_u.num_pairs(), s_full_f.num_pairs());

    let mut s_rand_u =
        SgdSolver::<Layout2d<f64>>::random_pair(&g_unit, 3, sched.clone(), 7).unwrap();
    let mut s_rand_f =
        SgdSolver::<Layout2d<f64>>::random_pair(&g_f64, 3, sched.clone(), 7).unwrap();
    assert_eq!(s_rand_u.num_pairs(), s_rand_f.num_pairs());

    let neg_cfg = StreamingNegativeConfig::new().with_negatives_per_node(3);
    let mut s_stream_u =
        SgdSolver::<Layout2d<f64>>::streaming_negatives(&g_unit, neg_cfg.clone(), sched.clone(), 7)
            .unwrap();
    let mut s_stream_f =
        SgdSolver::<Layout2d<f64>>::streaming_negatives(&g_f64, neg_cfg, sched, 7).unwrap();
    assert_eq!(
        s_stream_u.streaming_negative_adjacency(),
        s_stream_f.streaming_negative_adjacency()
    );

    let dog_cfg = DogVaradhanConfig::default()
        .with_n_landmarks(6)
        .with_seed(7);
    let bar_cfg = SparseBarrierConfig::default()
        .with_num_pivots(4)
        .with_seed(7);
    let (mut s_bar_u, stats_u, _) =
        SgdSolver::<Layout2d<f64>>::sparse_barrier_dog(&g_unit, &dog_cfg, &bar_cfg).unwrap();
    let (mut s_bar_f, stats_f, _) =
        SgdSolver::<Layout2d<f64>>::sparse_barrier_dog(&g_f64, &dog_cfg, &bar_cfg).unwrap();
    assert_eq!(stats_u, stats_f);

    let ort_cfg = OrtmannSparseStressConfig::new()
        .with_num_pivots(4)
        .with_iterations(10)
        .with_seed(7);
    let prob_u = build_ortmann_sparse_problem::<(), f64>(&g_unit, ort_cfg.clone()).unwrap();
    let prob_f = build_ortmann_sparse_problem::<f64, f64>(&g_f64, ort_cfg.clone()).unwrap();
    assert_eq!(prob_u, prob_f);
    let mut s_ort_u =
        SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress(&g_unit, ort_cfg.clone()).unwrap();
    let mut s_ort_f = SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress(&g_f64, ort_cfg).unwrap();
    assert_eq!(s_ort_u.num_pairs(), s_ort_f.num_pairs());

    for (su, sf) in [
        (&mut s_full_u, &mut s_full_f),
        (&mut s_rand_u, &mut s_rand_f),
        (&mut s_stream_u, &mut s_stream_f),
        (&mut s_bar_u, &mut s_bar_f),
        (&mut s_ort_u, &mut s_ort_f),
    ] {
        let mut lu = Layout2d::<f64>::init(n, 13, 1.0).unwrap();
        let mut lf = lu.clone();
        assert_eq!(su.run(&mut lu).unwrap(), SolverStatus::Completed);
        assert_eq!(sf.run(&mut lf).unwrap(), SolverStatus::Completed);
        for i in 0..n {
            assert_eq!(lu.x()[i].to_bits(), lf.x()[i].to_bits());
            assert_eq!(lu.y()[i].to_bits(), lf.y()[i].to_bits());
        }
    }
}

#[test]
fn ortmann_from_pivots_and_from_problem_match_single_call_bit_for_bit() {
    let g = make_grid_unit(3, 4);
    let n = g.num_nodes();
    let cfg = OrtmannSparseStressConfig::new()
        .with_num_pivots(4)
        .with_pivot_strategy(OrtmannPivotStrategy::MaxMinSp)
        .with_iterations(25)
        .with_eta_max(Some(3.0))
        .with_eta_min(Some(0.05))
        .with_seed(99);

    let prob_ref = build_ortmann_sparse_problem::<(), f64>(&g, cfg.clone()).unwrap();
    let pivots = prob_ref.pivots.clone();
    let h = pivots.len();
    let mut bfs = BfsWorkspace::new(n);
    let mut piv_dists = vec![f64::INFINITY; h * n];
    for (k, &p) in pivots.iter().enumerate() {
        bfs.bfs_from(&g, p as usize, &mut piv_dists[k * n..(k + 1) * n])
            .unwrap();
    }

    let prob_piv =
        OrtmannSparseProblem::<f64>::from_pivots(&g, &pivots, &piv_dists, cfg.clone()).unwrap();
    assert_eq!(prob_piv, prob_ref);

    let mut s_ref = SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress(&g, cfg.clone()).unwrap();
    let mut s_from_prob =
        SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress_from_problem(&prob_piv, cfg.clone())
            .unwrap();
    let mut s_from_piv =
        SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress_from_pivots(&g, &pivots, &piv_dists, cfg)
            .unwrap();

    assert_eq!(s_from_prob.eta_bounds(), s_ref.eta_bounds());
    assert_eq!(s_from_piv.eta_bounds(), s_ref.eta_bounds());

    let mut l_ref = Layout2d::<f64>::init(n, 5, 1.0).unwrap();
    let mut l_prob = l_ref.clone();
    let mut l_piv = l_ref.clone();
    s_ref.run(&mut l_ref).unwrap();
    s_from_prob.run(&mut l_prob).unwrap();
    s_from_piv.run(&mut l_piv).unwrap();
    for i in 0..n {
        assert_eq!(l_ref.x()[i].to_bits(), l_prob.x()[i].to_bits());
        assert_eq!(l_ref.y()[i].to_bits(), l_prob.y()[i].to_bits());
        assert_eq!(l_ref.x()[i].to_bits(), l_piv.x()[i].to_bits());
        assert_eq!(l_ref.y()[i].to_bits(), l_piv.y()[i].to_bits());
    }
}

#[test]
fn ortmann_from_pivots_handles_disconnected_infinity_and_validation_errors() {
    // Disconnected graph: two triangles (0,1,2) and (3,4,5) with +inf across components.
    let disc = CsrGraph::<()>::from_unweighted_edges(
        6,
        &[(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)],
        false,
        false,
    )
    .unwrap();
    let pivots = [0u32, 3u32];
    let dists_kn = [
        0.0,
        1.0,
        1.0,
        f64::INFINITY,
        f64::INFINITY,
        f64::INFINITY,
        f64::INFINITY,
        f64::INFINITY,
        f64::INFINITY,
        0.0,
        1.0,
        1.0,
    ];
    let cfg = OrtmannSparseStressConfig::new().with_iterations(10);
    let prob =
        OrtmannSparseProblem::<f64>::from_pivots(&disc, &pivots, &dists_kn, cfg.clone()).unwrap();
    for rec in prob.one_sided.iter().chain(prob.two_sided.iter()) {
        assert_eq!(rec.u < 3, rec.v < 3);
    }

    // Empty / singleton / edgeless graphs.
    let g0 = CsrGraph::<()>::from_unweighted_edges(0, &[], false, false).unwrap();
    let p0 = OrtmannSparseProblem::<f64>::from_pivots(&g0, &[], &[], cfg.clone()).unwrap();
    assert_eq!(p0.num_nodes, 0);
    assert!(p0.one_sided.is_empty() && p0.two_sided.is_empty());

    let g1 = CsrGraph::<()>::from_unweighted_edges(1, &[], false, false).unwrap();
    let p1 = OrtmannSparseProblem::<f64>::from_pivots(&g1, &[0], &[0.0], cfg.clone()).unwrap();
    assert_eq!(p1.num_nodes, 1);
    assert_eq!(p1.pivots, vec![0]);

    // Validation errors on `from_pivots`.
    assert!(matches!(
        OrtmannSparseProblem::<f64>::from_pivots(&disc, &[0, 6], &dists_kn, cfg.clone()),
        Err(CoreError::IndexOutOfBounds {
            index: 6,
            len: 6,
            ..
        })
    ));
    assert!(matches!(
        OrtmannSparseProblem::<f64>::from_pivots(&disc, &[2, 2], &dists_kn, cfg.clone()),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        OrtmannSparseProblem::<f64>::from_pivots(&disc, &[0, 3], &[0.0; 11], cfg.clone()),
        Err(CoreError::DimensionMismatch {
            expected: 12,
            actual: 11,
            ..
        })
    ));
    let mut nan_dists = dists_kn;
    nan_dists[4] = f64::NAN;
    assert!(matches!(
        OrtmannSparseProblem::<f64>::from_pivots(&disc, &[0, 3], &nan_dists, cfg.clone()),
        Err(CoreError::NonFiniteValue { .. })
    ));
    let mut neg_dists = dists_kn;
    neg_dists[4] = -0.5;
    assert!(matches!(
        OrtmannSparseProblem::<f64>::from_pivots(&disc, &[0, 3], &neg_dists, cfg.clone()),
        Err(CoreError::InvalidInput(_))
    ));
    let mut neg_inf_dists = dists_kn;
    neg_inf_dists[4] = f64::NEG_INFINITY;
    assert!(matches!(
        OrtmannSparseProblem::<f64>::from_pivots(&disc, &[0, 3], &neg_inf_dists, cfg.clone()),
        Err(CoreError::InvalidInput(_))
    ));

    // Out-of-bounds or self-loop pair inside `ortmann_sparse_stress_from_problem` must fail fast.
    let mut bad_prob = prob.clone();
    bad_prob.two_sided[0].v = bad_prob.two_sided[0].u;
    assert!(matches!(
        SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress_from_problem(&bad_prob, cfg.clone()),
        Err(CoreError::InvalidInput(_))
    ));
    let mut oob_prob = prob;
    oob_prob.two_sided[0].v = 9;
    assert!(matches!(
        SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress_from_problem(&oob_prob, cfg),
        Err(CoreError::IndexOutOfBounds {
            index: 9,
            len: 6,
            ..
        })
    ));
}
