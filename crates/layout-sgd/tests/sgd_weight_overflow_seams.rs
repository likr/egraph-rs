use egraph_core::{CoreError, CsrGraph, Layout2d};
use egraph_layout_sgd::{
    OrtmannSparseStressConfig, SgdScheduleConfig, SgdSolver, SparseBarrierConfig,
    StreamingNegativeConfig, build_ortmann_sparse_problem,
};
use egraph_neighborhood::DogVaradhanLandmarks;

#[test]
fn test_f32_tiny_distance_overflow_fails_fast_across_sgd_constructors() {
    let g_path3 = CsrGraph::<f64>::from_edges(
        3,
        &[(0, 1), (1, 2)],
        Some(&[1e-25_f64, 1.0_f64]),
        false,
        false,
    )
    .unwrap();
    let g_edge2 =
        CsrGraph::<f64>::from_edges(2, &[(0, 1)], Some(&[1e-25_f64]), false, false).unwrap();
    let g_dir2 =
        CsrGraph::<f64>::from_edges(2, &[(0, 1)], Some(&[1e-25_f64]), true, false).unwrap();

    for g in [&g_path3, &g_edge2] {
        assert!(matches!(
            build_ortmann_sparse_problem::<f64, f32>(g, OrtmannSparseStressConfig::default()),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            SgdSolver::<Layout2d<f32>>::ortmann_sparse_stress(
                g,
                OrtmannSparseStressConfig::default()
            ),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            SgdSolver::<Layout2d<f32>>::full(g, SgdScheduleConfig::default(), 42),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            SgdSolver::<Layout2d<f32>>::random_pair(g, 2, SgdScheduleConfig::default(), 42),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            SgdSolver::<Layout2d<f32>>::streaming_negatives(
                g,
                StreamingNegativeConfig::default(),
                SgdScheduleConfig::default(),
                42
            ),
            Err(CoreError::InvalidInput(_))
        ));
    }

    assert!(matches!(
        SgdSolver::<Layout2d<f32>>::full(&g_dir2, SgdScheduleConfig::default(), 42),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        SgdSolver::<Layout2d<f32>>::random_pair(&g_dir2, 2, SgdScheduleConfig::default(), 42),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        SgdSolver::<Layout2d<f32>>::omega(
            &g_edge2,
            &[0.0, 0.0, 0.0, 0.0],
            2,
            2,
            1e-25_f64,
            SgdScheduleConfig::default(),
            42
        ),
        Err(CoreError::InvalidInput(_))
    ));

    let lm = DogVaradhanLandmarks::new_validated(2, vec![0], vec![0.0, 1e-25_f64], None).unwrap();
    let sb_cfg = SparseBarrierConfig::default()
        .with_eps_dist(1e-25_f64)
        .with_neg_per_node(Some(0));
    assert!(matches!(
        SgdSolver::<Layout2d<f32>>::sparse_barrier_from_landmarks(
            &g_edge2,
            &lm,
            Some(&[0]),
            &sb_cfg
        ),
        Err(CoreError::InvalidInput(_))
    ));
}

#[test]
fn test_f64_tiny_distance_overflow_fails_fast_across_sgd_constructors() {
    let g_edge2 =
        CsrGraph::<f64>::from_edges(2, &[(0, 1)], Some(&[1e-200_f64]), false, false).unwrap();
    let g_dir2 =
        CsrGraph::<f64>::from_edges(2, &[(0, 1)], Some(&[1e-200_f64]), true, false).unwrap();

    assert!(matches!(
        build_ortmann_sparse_problem::<f64, f64>(&g_edge2, OrtmannSparseStressConfig::default()),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress(
            &g_edge2,
            OrtmannSparseStressConfig::default()
        ),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        SgdSolver::<Layout2d<f64>>::full(&g_edge2, SgdScheduleConfig::default(), 42),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        SgdSolver::<Layout2d<f64>>::full(&g_dir2, SgdScheduleConfig::default(), 42),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        SgdSolver::<Layout2d<f64>>::random_pair(&g_edge2, 2, SgdScheduleConfig::default(), 42),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        SgdSolver::<Layout2d<f64>>::random_pair(&g_dir2, 2, SgdScheduleConfig::default(), 42),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        SgdSolver::<Layout2d<f64>>::streaming_negatives(
            &g_edge2,
            StreamingNegativeConfig::default(),
            SgdScheduleConfig::default(),
            42
        ),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        SgdSolver::<Layout2d<f64>>::omega(
            &g_edge2,
            &[0.0, 0.0, 0.0, 0.0],
            2,
            2,
            1e-200_f64,
            SgdScheduleConfig::default(),
            42
        ),
        Err(CoreError::InvalidInput(_))
    ));

    let lm = DogVaradhanLandmarks::new_validated(2, vec![0], vec![0.0, 1e-200_f64], None).unwrap();
    let sb_cfg = SparseBarrierConfig::default()
        .with_eps_dist(1e-200_f64)
        .with_neg_per_node(Some(0));
    assert!(matches!(
        SgdSolver::<Layout2d<f64>>::sparse_barrier_from_landmarks(
            &g_edge2,
            &lm,
            Some(&[0]),
            &sb_cfg
        ),
        Err(CoreError::InvalidInput(_))
    ));
}
