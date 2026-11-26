use egraph_core::{CoreError, CsrGraph, IterativeSolver, Layout2d, SolverStatus};
use egraph_neighborhood::{
    DogVaradhanConfig, SparseProbabilities, compute_regularized_dog_varadhan_landmarks,
};

use super::super::{FieldLayoutSolver, IbUmapConfig, RhoSchedule, TFdpConfig};
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

    let quad = RhoSchedule {
        start_weight: 2.0,
        end_weight: 0.2,
        power: 2.0,
    };
    quad.validate().unwrap();
    assert!((quad.evaluate(0.5) - (2.0 + (0.2 - 2.0) * 0.25)).abs() < 1e-12);

    let bad_cases = [
        RhoSchedule {
            start_weight: -0.1,
            ..sched
        },
        RhoSchedule {
            start_weight: f64::NAN,
            ..sched
        },
        RhoSchedule {
            end_weight: -0.1,
            ..sched
        },
        RhoSchedule {
            end_weight: f64::INFINITY,
            ..sched
        },
        RhoSchedule {
            power: 0.0,
            ..sched
        },
        RhoSchedule {
            power: -1.0,
            ..sched
        },
        RhoSchedule {
            power: f64::NAN,
            ..sched
        },
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
    let mut se_fdp = FieldLayoutSolver::<f64>::t_fdp(&empty, tfdp_cfg).unwrap();
    let mut se_umap = FieldLayoutSolver::<f64>::ib_umap(&empty, 16, &dog_cfg, ib_cfg).unwrap();
    let mut le = Layout2d::<f64>::zeros(0);
    assert_eq!(se_fdp.run(&mut le).unwrap(), SolverStatus::Converged);
    assert_eq!(se_umap.run(&mut le).unwrap(), SolverStatus::Converged);

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let mut ss_fdp = FieldLayoutSolver::<f64>::t_fdp(&single, tfdp_cfg).unwrap();
    let mut ss_umap = FieldLayoutSolver::<f64>::ib_umap(&single, 16, &dog_cfg, ib_cfg).unwrap();
    let mut ls = Layout2d::from_xy(vec![3.5], vec![-2.5]).unwrap();
    assert_eq!(ss_fdp.run(&mut ls).unwrap(), SolverStatus::Converged);
    assert_eq!(ss_umap.run(&mut ls).unwrap(), SolverStatus::Converged);
    assert_eq!(ls.x(), &[3.5]);
    assert_eq!(ls.y(), &[-2.5]);

    // Disconnected graph with isolated vertices runs cleanly
    let disc = CsrGraph::<f64>::from_edges(6, &[(0, 1), (3, 4)], None, false, false).unwrap();
    let mut sd_fdp = FieldLayoutSolver::<f64>::t_fdp(
        &disc,
        TFdpConfig {
            iterations: 15,
            ..tfdp_cfg
        },
    )
    .unwrap();
    let mut sd_umap = FieldLayoutSolver::<f64>::ib_umap(
        &disc,
        16,
        &dog_cfg,
        IbUmapConfig {
            iterations: 15,
            ..ib_cfg
        },
    )
    .unwrap();
    let mut ld1 = seeded_layout::<f64>(6, 1.0, 31);
    let mut ld2 = ld1.clone();
    assert_eq!(sd_fdp.run(&mut ld1).unwrap(), SolverStatus::Completed);
    assert_eq!(sd_umap.run(&mut ld2).unwrap(), SolverStatus::Completed);
    assert!(ld1.is_all_finite() && ld2.is_all_finite());

    // Zero-iteration solvers complete immediately
    let zero_fdp = FieldLayoutSolver::<f64>::t_fdp(
        &disc,
        TFdpConfig {
            iterations: 0,
            ..tfdp_cfg
        },
    )
    .unwrap();
    assert_eq!(zero_fdp.status(), SolverStatus::Completed);
    assert_eq!(zero_fdp.total_iterations(), 0);
    let zero_umap = FieldLayoutSolver::<f64>::ib_umap(
        &disc,
        16,
        &dog_cfg,
        IbUmapConfig {
            iterations: 0,
            ..ib_cfg
        },
    )
    .unwrap();
    assert_eq!(zero_umap.status(), SolverStatus::Completed);
    assert_eq!(zero_umap.total_iterations(), 0);

    // Dimension mismatch & NaN/Inf layout guards
    let mut l_wrong = Layout2d::<f64>::zeros(4);
    assert!(matches!(
        sd_fdp.step(&mut l_wrong),
        Err(CoreError::DimensionMismatch { .. })
    ));
    let mut l_nan = Layout2d::from_xy(vec![0.0; 6], vec![f64::NAN; 6]).unwrap();
    assert!(matches!(
        sd_umap.step(&mut l_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));

    // Negative or NaN config & weight guards
    let bad_tfdp_configs = [
        TFdpConfig {
            learning_rate: 0.0,
            ..tfdp_cfg
        },
        TFdpConfig {
            learning_rate: f64::NAN,
            ..tfdp_cfg
        },
        TFdpConfig {
            momentum: 1.0,
            ..tfdp_cfg
        },
        TFdpConfig {
            alpha: 0.0,
            ..tfdp_cfg
        },
        TFdpConfig {
            gamma: -0.5,
            ..tfdp_cfg
        },
        TFdpConfig {
            theta: -0.1,
            ..tfdp_cfg
        },
        TFdpConfig {
            rho: RhoSchedule {
                power: 0.0,
                ..RhoSchedule::default()
            },
            ..tfdp_cfg
        },
    ];
    for bad in bad_tfdp_configs {
        assert!(FieldLayoutSolver::<f64>::t_fdp(&disc, bad).is_err());
    }

    let bad_ib_configs = [
        IbUmapConfig {
            learning_rate: f64::NAN,
            ..ib_cfg
        },
        IbUmapConfig {
            min_dist: -0.1,
            ..ib_cfg
        },
        IbUmapConfig {
            spread: 0.0,
            ..ib_cfg
        },
        IbUmapConfig {
            c_k: -1.0,
            ..ib_cfg
        },
        IbUmapConfig {
            tau_scale: 0.0,
            ..ib_cfg
        },
        IbUmapConfig {
            theta: f64::INFINITY,
            ..ib_cfg
        },
        IbUmapConfig {
            a: Some(0.0),
            ..ib_cfg
        },
        IbUmapConfig {
            b: Some(f64::NAN),
            ..ib_cfg
        },
    ];
    for bad in bad_ib_configs {
        assert!(FieldLayoutSolver::<f64>::ib_umap(&disc, 16, &dog_cfg, bad).is_err());
    }

    // Degree slice length mismatch in ib_umap_from_fuzzy_set
    let probs = SparseProbabilities::empty(3);
    assert!(matches!(
        FieldLayoutSolver::<f64>::ib_umap_from_fuzzy_set(probs, Some(&[1, 2]), ib_cfg),
        Err(CoreError::DimensionMismatch { .. })
    ));

    // Negative edge weight in t_fdp
    let neg_w_graph =
        CsrGraph::<f64>::from_edges(2, &[(0, 1)], Some(&[-1.0]), false, false).unwrap();
    assert!(matches!(
        FieldLayoutSolver::<f64>::t_fdp(&neg_w_graph, tfdp_cfg),
        Err(CoreError::InvalidInput(_))
    ));
}

#[test]
fn test_ib_umap_from_landmarks_bit_exact_and_validation() {
    let g = grid_graph(4, 4);
    let n = g.num_nodes();
    let dog_cfg = DogVaradhanConfig::default();
    let cfg = IbUmapConfig {
        iterations: 15,
        seed: 67,
        ..Default::default()
    };
    let (landmarks, _) = compute_regularized_dog_varadhan_landmarks(&g, &dog_cfg).unwrap();

    let mut s_dog = FieldLayoutSolver::<f64>::ib_umap(&g, 32, &dog_cfg, cfg).unwrap();
    let mut s_lm =
        FieldLayoutSolver::<f64>::ib_umap_from_landmarks(&g, &landmarks, 32, cfg).unwrap();
    let mut l_dog = seeded_layout::<f64>(n, 1.0, 23);
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
        FieldLayoutSolver::<f64>::ib_umap_from_landmarks(&g, &bad_nodes, 32, cfg),
        Err(CoreError::DimensionMismatch { .. })
    ));

    let mut bad_empty_lm = landmarks.clone();
    bad_empty_lm.landmarks.clear();
    bad_empty_lm.d_dog.clear();
    bad_empty_lm.d_coarse = None;
    assert!(matches!(
        FieldLayoutSolver::<f64>::ib_umap_from_landmarks(&g, &bad_empty_lm, 32, cfg),
        Err(CoreError::InvalidInput(_))
    ));

    let mut bad_nan = landmarks.clone();
    bad_nan.d_dog[0] = f64::NAN;
    assert!(matches!(
        FieldLayoutSolver::<f64>::ib_umap_from_landmarks(&g, &bad_nan, 32, cfg),
        Err(CoreError::NonFiniteValue { .. })
    ));

    let mut bad_idx = landmarks;
    bad_idx.landmarks[0] = n as u32;
    assert!(matches!(
        FieldLayoutSolver::<f64>::ib_umap_from_landmarks(&g, &bad_idx, 32, cfg),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
}
