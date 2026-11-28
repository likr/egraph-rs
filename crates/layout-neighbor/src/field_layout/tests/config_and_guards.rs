use egraph_core::{CoreError, CsrGraph, IterativeSolver, Layout2d, SolverStatus};
use egraph_neighborhood::{
    DogVaradhanConfig, DogVaradhanLandmarks, SparseProbabilities,
    compute_regularized_dog_varadhan_landmarks,
};

use crate::backends::{BarnesHut, IbFft};

use super::super::{IbUmapConfig, IbUmapSolver, RhoSchedule, TFdpConfig, TFdpSolver};
use super::{grid_graph, seeded_layout};

#[test]
fn test_rho_schedule_evaluation_and_validation() {
    let sched = RhoSchedule::default();
    assert_eq!(sched.start_weight, 1.0);
    assert_eq!(sched.end_weight, 0.1);
    assert_eq!(sched.power, 1.0);
    sched.validate().unwrap();

    assert!((sched.evaluate(-0.5) - 1.0).abs() < 1e-12);
    assert!((sched.evaluate(0.0) - 1.0).abs() < 1e-12);
    assert!((sched.evaluate(0.5) - 0.55).abs() < 1e-12);
    assert!((sched.evaluate(1.0) - 0.1).abs() < 1e-12);
    assert!((sched.evaluate(2.0) - 0.1).abs() < 1e-12);
    assert!((sched.evaluate(f64::NAN) - 1.0).abs() < 1e-12);

    let quad = RhoSchedule::new()
        .with_start_weight(2.0)
        .with_end_weight(0.2)
        .with_power(2.0);
    quad.validate().unwrap();
    assert!((quad.evaluate(0.5) - (2.0 + (0.2 - 2.0) * 0.25)).abs() < 1e-12);

    let bad_cases = [
        sched.clone().with_start_weight(-0.1),
        sched.clone().with_start_weight(f64::NAN),
        sched.clone().with_end_weight(-0.1),
        sched.clone().with_end_weight(f64::INFINITY),
        sched.clone().with_power(0.0),
        sched.clone().with_power(-1.0),
        sched.with_power(f64::NAN),
    ];
    for bad in bad_cases {
        assert!(bad.validate().is_err());
    }
}

#[test]
fn test_field_layout_edge_cases_and_error_guards() {
    let tfdp_cfg = TFdpConfig::default();
    let ib_cfg = IbUmapConfig::default();
    let dog_cfg = DogVaradhanConfig::default();

    // Empty and singleton graphs for both t-FDP and ibUMAP
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let mut se_fdp = TFdpSolver::<Layout2d<f64>>::new(&empty, tfdp_cfg.clone()).unwrap();
    let mut se_umap =
        IbUmapSolver::<f64>::from_graph_dog(&empty, 16, &dog_cfg, ib_cfg.clone()).unwrap();
    let mut le = Layout2d::<f64>::zeros(0);
    assert_eq!(se_fdp.run(&mut le).unwrap(), SolverStatus::Converged);
    assert_eq!(se_umap.run(&mut le).unwrap(), SolverStatus::Converged);

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let mut ss_fdp = TFdpSolver::<Layout2d<f64>>::new(&single, tfdp_cfg.clone()).unwrap();
    let mut ss_umap =
        IbUmapSolver::<f64>::from_graph_dog(&single, 16, &dog_cfg, ib_cfg.clone()).unwrap();
    let mut ls = Layout2d::from_xy(vec![3.5], vec![-2.5]).unwrap();
    assert_eq!(ss_fdp.run(&mut ls).unwrap(), SolverStatus::Converged);
    assert_eq!(ss_umap.run(&mut ls).unwrap(), SolverStatus::Converged);
    assert_eq!(ls.x(), &[3.5]);
    assert_eq!(ls.y(), &[-2.5]);

    // Disconnected graph with isolated vertices runs cleanly
    let disc = CsrGraph::<f64>::from_edges(6, &[(0, 1), (3, 4)], None, false, false).unwrap();
    let mut sd_fdp =
        TFdpSolver::<Layout2d<f64>>::new(&disc, tfdp_cfg.clone().with_iterations(15)).unwrap();
    let mut sd_umap = IbUmapSolver::<f64>::from_graph_dog(
        &disc,
        16,
        &dog_cfg,
        ib_cfg.clone().with_iterations(15),
    )
    .unwrap();
    let mut ld1 = seeded_layout::<f64>(6, 1.0, 31);
    let mut ld2 = ld1.clone();
    assert_eq!(sd_fdp.run(&mut ld1).unwrap(), SolverStatus::Completed);
    assert_eq!(sd_umap.run(&mut ld2).unwrap(), SolverStatus::Completed);
    assert!(ld1.is_all_finite() && ld2.is_all_finite());

    // Zero-iteration solvers complete immediately and still validate run_steps(target, 0)
    let mut zero_fdp =
        TFdpSolver::<Layout2d<f64>>::new(&disc, tfdp_cfg.clone().with_iterations(0)).unwrap();
    assert_eq!(zero_fdp.status(), SolverStatus::Completed);
    assert_eq!(zero_fdp.total_iterations(), Some(0));
    let mut zero_umap =
        IbUmapSolver::<f64>::from_graph_dog(&disc, 16, &dog_cfg, ib_cfg.clone().with_iterations(0))
            .unwrap();
    assert_eq!(zero_umap.status(), SolverStatus::Completed);
    assert_eq!(zero_umap.total_iterations(), Some(0));

    // Dimension mismatch & NaN/Inf layout guards (including run_steps(target, 0))
    let mut l_wrong = Layout2d::<f64>::zeros(4);
    assert!(matches!(
        sd_fdp.step(&mut l_wrong),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        zero_fdp.run_steps(&mut l_wrong, 0),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        zero_umap.run_steps(&mut l_wrong, 0),
        Err(CoreError::DimensionMismatch { .. })
    ));

    let mut l_nan = Layout2d::from_xy(vec![0.0; 6], vec![f64::NAN; 6]).unwrap();
    assert!(matches!(
        sd_umap.step(&mut l_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert_eq!(sd_umap.status(), SolverStatus::Completed);

    // Invalid backend config fails fast at construction.
    assert!(matches!(
        TFdpSolver::<Layout2d<f64>, BarnesHut>::new_with_backend(
            &disc,
            TFdpConfig::new(),
            BarnesHut::new(-0.1),
        ),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        TFdpSolver::<Layout2d<f64>, IbFft>::new_with_backend(
            &disc,
            TFdpConfig::new(),
            IbFft::new().with_intervals(0),
        ),
        Err(CoreError::InvalidInput(_))
    ));

    // Negative or NaN config & weight guards
    let bad_tfdp_configs = [
        tfdp_cfg.clone().with_learning_rate(0.0),
        tfdp_cfg.clone().with_learning_rate(f64::NAN),
        tfdp_cfg.clone().with_momentum(1.0),
        tfdp_cfg.clone().with_alpha(0.0),
        tfdp_cfg.clone().with_gamma(-0.5),
        tfdp_cfg
            .clone()
            .with_rho(RhoSchedule::new().with_power(0.0)),
    ];
    for bad in bad_tfdp_configs {
        assert!(TFdpSolver::<Layout2d<f64>>::new(&disc, bad).is_err());
    }

    let bad_ib_configs = [
        ib_cfg.clone().with_learning_rate(f64::NAN),
        ib_cfg.clone().with_min_dist(-0.1),
        ib_cfg.clone().with_spread(0.0),
        ib_cfg.clone().with_c_k(-1.0),
        ib_cfg.clone().with_tau_scale(0.0),
        ib_cfg.clone().with_theta(f64::INFINITY),
        ib_cfg.clone().with_a(Some(0.0)),
        ib_cfg.clone().with_b(Some(f64::NAN)),
    ];
    for bad in bad_ib_configs {
        assert!(IbUmapSolver::<f64>::from_graph_dog(&disc, 16, &dog_cfg, bad).is_err());
    }

    // Degree slice length mismatch in IbUmapSolver::from_fuzzy_set
    let probs = SparseProbabilities::empty(3);
    assert!(matches!(
        IbUmapSolver::<f64>::from_fuzzy_set(probs, Some(&[1, 2]), ib_cfg),
        Err(CoreError::DimensionMismatch { .. })
    ));

    // Negative edge weight in TFdpSolver
    let neg_w_graph =
        CsrGraph::<f64>::from_edges(2, &[(0, 1)], Some(&[-1.0]), false, false).unwrap();
    assert!(matches!(
        TFdpSolver::<Layout2d<f64>>::new(&neg_w_graph, tfdp_cfg),
        Err(CoreError::InvalidInput(_))
    ));
}

#[test]
fn test_ib_umap_from_landmarks_bit_exact_and_validation() {
    let g = grid_graph(4, 4);
    let n = g.num_nodes();
    let dog_cfg = DogVaradhanConfig::default();
    let cfg = IbUmapConfig::new().with_iterations(15).with_seed(67);
    let (landmarks, _) = compute_regularized_dog_varadhan_landmarks(&g, &dog_cfg).unwrap();

    let mut s_dog = IbUmapSolver::<f64>::from_graph_dog(&g, 32, &dog_cfg, cfg.clone()).unwrap();
    let mut s_lm = IbUmapSolver::<f64>::from_landmarks(&g, &landmarks, 32, cfg.clone()).unwrap();
    let mut l_dog = seeded_layout::<f64>(n, 1.0, 23);
    let mut l_lm = l_dog.clone();
    s_dog.run(&mut l_dog).unwrap();
    s_lm.run(&mut l_lm).unwrap();
    for i in 0..n {
        assert_eq!(l_dog.x()[i].to_bits(), l_lm.x()[i].to_bits());
        assert_eq!(l_dog.y()[i].to_bits(), l_lm.y()[i].to_bits());
    }

    // Unweighted CsrGraph<()> works with from_landmarks
    let g_unit =
        CsrGraph::<()>::from_unweighted_csr(g.indptr(), g.indices(), false, false).unwrap();
    let mut s_unit =
        IbUmapSolver::<f64>::from_landmarks(&g_unit, &landmarks, 32, cfg.clone()).unwrap();
    let mut l_unit = seeded_layout::<f64>(n, 1.0, 23);
    s_unit.run(&mut l_unit).unwrap();
    for i in 0..n {
        assert_eq!(l_unit.x()[i].to_bits(), l_lm.x()[i].to_bits());
        assert_eq!(l_unit.y()[i].to_bits(), l_lm.y()[i].to_bits());
    }

    // Validation errors
    let bad_nodes = DogVaradhanLandmarks::empty(n + 1).unwrap();
    assert!(matches!(
        IbUmapSolver::<f64>::from_landmarks(&g, &bad_nodes, 32, cfg.clone()),
        Err(CoreError::DimensionMismatch { .. })
    ));

    let bad_empty_lm = DogVaradhanLandmarks::empty(n).unwrap();
    assert!(matches!(
        IbUmapSolver::<f64>::from_landmarks(&g, &bad_empty_lm, 32, cfg),
        Err(CoreError::InvalidInput(_))
    ));
}
