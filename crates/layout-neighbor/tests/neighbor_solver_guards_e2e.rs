//! Opaque-box E2E tests for `IbUmapSolver` boundary conditions, configuration validation,
//! degree mismatches, and non-mutating target guards.

use egraph_core::{CoreError, CsrGraph, IterativeSolver, Layout2d, SolverStatus};
use egraph_layout_neighbor::{
    BarnesHut, Exact, IbFft, IbUmapBackend2d, IbUmapConfig, IbUmapSolver,
};
use egraph_neighborhood::{build_pure_2hop_knn, compute_fuzzy_simplicial_set};

fn make_grid(rows: usize, cols: usize) -> CsrGraph<f64> {
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
fn ib_umap_empty_and_singleton_graphs() {
    let cfg = IbUmapConfig::new().with_iterations(5);
    for n in [0usize, 1] {
        let g = CsrGraph::<f64>::from_edges(n, &[], None, false, false).unwrap();
        for wrapper in [
            IbUmapBackend2d::IbFft(IbFft::new()),
            IbUmapBackend2d::BarnesHut(BarnesHut::new(0.5)),
            IbUmapBackend2d::Exact(Exact::new()),
        ] {
            let mut solver = IbUmapSolver::<Layout2d<f64>, IbUmapBackend2d>::new_with_backend(
                &g,
                cfg.clone(),
                wrapper,
            )
            .unwrap();
            let mut layout = Layout2d::<f64>::from_xy(vec![1.5; n], vec![-2.5; n]).unwrap();
            let snap = layout.clone();
            assert_eq!(solver.run(&mut layout).unwrap(), SolverStatus::Converged);
            assert_eq!(layout, snap);
        }
    }
}

#[test]
fn ib_umap_zero_iterations_is_completed_and_noop() {
    let g = make_grid(2, 3);
    let cfg = IbUmapConfig::new().with_iterations(0);
    let mut solver = IbUmapSolver::<Layout2d<f64>>::new(&g, cfg).unwrap();
    assert_eq!(solver.status(), SolverStatus::Completed);
    assert_eq!(solver.total_iterations(), Some(0));

    let mut layout = Layout2d::<f64>::init(6, 4, 1.0).unwrap();
    let snap = layout.clone();
    assert_eq!(solver.step(&mut layout).unwrap(), SolverStatus::Completed);
    assert_eq!(layout, snap);
}

#[test]
fn ib_umap_all_zero_layout_applies_deterministic_jitter() {
    let g = make_grid(3, 3);
    let cfg = IbUmapConfig::new().with_iterations(4).with_seed(77);
    let mut s1 = IbUmapSolver::<Layout2d<f64>>::new(&g, cfg.clone()).unwrap();
    let mut s2 = IbUmapSolver::<Layout2d<f64>>::new(&g, cfg).unwrap();
    let mut l1 = Layout2d::<f64>::zeros(9);
    let mut l2 = Layout2d::<f64>::zeros(9);

    s1.run(&mut l1).unwrap();
    s2.run(&mut l2).unwrap();
    assert_eq!(l1, l2);
    assert!(l1.is_all_finite());
    assert!(l1.x().iter().any(|&v| v.abs() > 1e-6));
}

#[test]
fn ib_umap_invalid_config_and_backend_params_fail_fast() {
    let g = make_grid(2, 3);

    for bad_cfg in [
        IbUmapConfig::new().with_learning_rate(0.0),
        IbUmapConfig::new().with_learning_rate(-1.0),
        IbUmapConfig::new().with_c_k(-0.1),
        IbUmapConfig::new().with_tau_scale(0.0),
        IbUmapConfig::new().with_min_dist(-0.05),
        IbUmapConfig::new().with_spread(0.0),
        IbUmapConfig::new().with_a(Some(0.0)),
        IbUmapConfig::new().with_b(Some(0.0)),
    ] {
        assert!(matches!(
            bad_cfg.validate(),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            IbUmapSolver::<Layout2d<f64>>::new(&g, bad_cfg),
            Err(CoreError::InvalidInput(_))
        ));
    }

    for nf_cfg in [
        IbUmapConfig::new().with_learning_rate(f64::NAN),
        IbUmapConfig::new().with_c_k(f64::INFINITY),
        IbUmapConfig::new().with_tau_scale(f64::NAN),
        IbUmapConfig::new().with_min_dist(f64::NAN),
        IbUmapConfig::new().with_spread(f64::INFINITY),
    ] {
        assert!(matches!(
            nf_cfg.validate(),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            IbUmapSolver::<Layout2d<f64>>::new(&g, nf_cfg),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }

    let cfg = IbUmapConfig::new().with_iterations(3);
    assert!(matches!(
        IbUmapSolver::<Layout2d<f64>, BarnesHut>::new_with_backend(
            &g,
            cfg.clone(),
            BarnesHut::new(-0.1),
        ),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        IbUmapSolver::<Layout2d<f64>, BarnesHut>::new_with_backend(
            &g,
            cfg.clone(),
            BarnesHut::new(f64::NAN),
        ),
        Err(CoreError::InvalidInput(_)) | Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        IbUmapSolver::<Layout2d<f64>, IbFft>::new_with_backend(
            &g,
            cfg,
            IbFft::new().with_intervals(0),
        ),
        Err(CoreError::InvalidInput(_))
    ));
}

#[test]
fn ib_umap_fuzzy_set_mismatched_degrees_returns_dimension_mismatch() {
    let g = make_grid(3, 3);
    let (knn, _, _) = build_pure_2hop_knn(&g).unwrap();
    let probs = compute_fuzzy_simplicial_set(&knn, 64, 1e-5).unwrap();
    let bad_degs = [4_u32; 5];

    assert!(matches!(
        IbUmapSolver::<Layout2d<f64>, Exact>::from_fuzzy_set_with_backend(
            probs.clone(),
            Some(&bad_degs),
            IbUmapConfig::new(),
            Exact::new(),
        ),
        Err(CoreError::DimensionMismatch {
            expected: 9,
            actual: 5,
            ..
        })
    ));
    assert!(matches!(
        IbUmapSolver::<Layout2d<f64>, Exact>::from_fuzzy_set_ref_with_backend(
            &probs,
            Some(&bad_degs),
            IbUmapConfig::new(),
            Exact::new(),
        ),
        Err(CoreError::DimensionMismatch {
            expected: 9,
            actual: 5,
            ..
        })
    ));
}

#[test]
fn ib_umap_target_mismatch_and_non_finite_fail_fast_without_mutation() {
    let g = make_grid(2, 3);
    let mut solver =
        IbUmapSolver::<Layout2d<f64>>::new(&g, IbUmapConfig::new().with_iterations(4)).unwrap();

    let mut wrong_len = Layout2d::<f64>::init(4, 11, 2.0).unwrap();
    let wrong_snap = wrong_len.clone();
    assert!(matches!(
        solver.run_steps(&mut wrong_len, 0),
        Err(CoreError::DimensionMismatch {
            expected: 6,
            actual: 4,
            ..
        })
    ));
    assert_eq!(wrong_len, wrong_snap);
    assert_eq!(solver.status(), SolverStatus::Initialized);
    assert_eq!(solver.iteration(), 0);

    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut nf_layout =
            Layout2d::<f64>::from_xy(vec![0.0, bad, 1.0, 2.0, 3.0, 4.0], vec![0.5; 6]).unwrap();
        let x_bits_before: Vec<u64> = nf_layout.x().iter().map(|v| v.to_bits()).collect();
        let y_bits_before: Vec<u64> = nf_layout.y().iter().map(|v| v.to_bits()).collect();

        assert!(matches!(
            solver.step(&mut nf_layout),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            solver.run_steps(&mut nf_layout, 0),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert_eq!(
            nf_layout
                .x()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            x_bits_before
        );
        assert_eq!(
            nf_layout
                .y()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            y_bits_before
        );
        assert_eq!(solver.status(), SolverStatus::Initialized);
        assert_eq!(solver.iteration(), 0);
    }
}
