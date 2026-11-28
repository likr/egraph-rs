use super::*;
use egraph_spatial_fft::ScreenedPoissonConfig;

fn dense_disk_layout(side: usize) -> Layout2d<f64> {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for r in 0..side {
        for c in 0..side {
            let x = -0.8 + 1.6 * (c as f64) / ((side - 1) as f64);
            let y = -0.8 + 1.6 * (r as f64) / ((side - 1) as f64);
            if x * x + y * y <= 0.64 {
                xs.push(x);
                ys.push(y);
            }
        }
    }
    Layout2d::from_xy(xs, ys).unwrap()
}

#[test]
fn test_refiner_step_run_equivalence_and_excess_reduction() {
    let layout_init = dense_disk_layout(24);
    let n = layout_init.len();
    let field_cfg = ScreenedPoissonConfig::default()
        .with_grid_size(128)
        .with_padding_factor(2)
        .with_rho_max(Some(0.15))
        .with_sigma(Some(0.10))
        .with_kappa(Some(1.0));
    let cfg = ScreenedPoissonRefinerConfig::new()
        .with_field(field_cfg)
        .with_iterations(8)
        .with_step_size(0.3)
        .with_momentum(0.4);

    let mut r_run = ScreenedPoissonRefiner::<f64>::new(n, cfg.clone()).unwrap();
    let mut r_step = ScreenedPoissonRefiner::<f64>::new(n, cfg).unwrap();
    assert_eq!(r_run.total_iterations(), Some(8));
    let mut l_run = layout_init.clone();
    let mut l_step = layout_init;

    assert_eq!(
        r_step.run_steps(&mut l_step, 0).unwrap(),
        SolverStatus::Initialized
    );
    r_step.step(&mut l_step).unwrap();
    let initial_excess_mass = r_step.last_stats().excess_mass;
    assert!(initial_excess_mass > 0.1);
    r_step.run_steps(&mut l_step, 2).unwrap();
    assert_eq!(r_step.iteration(), 3);
    r_step.run(&mut l_step).unwrap();

    r_run.run(&mut l_run).unwrap();

    assert_eq!(r_run.iteration(), r_step.iteration());
    assert_eq!(r_run.status(), r_step.status());
    assert_eq!(l_run, l_step);
    assert!(r_run.last_energy() >= 0.0);
    assert!(r_run.last_stats().excess_mass < initial_excess_mass);
    // Overdense plateau must expand: final bbox span larger than initial 1.6.
    let (min_x, max_x, min_y, max_y) = bbox_of(&l_run);
    assert!(
        max_x - min_x > 1.9,
        "expected expansion > 1.9, got {}",
        max_x - min_x
    );
    assert!(
        max_y - min_y > 1.9,
        "expected expansion > 1.9, got {}",
        max_y - min_y
    );
}

#[test]
fn test_refiner_deterministic_seed_on_coincident_zero_layout() {
    let n = 32usize;
    let l0 = Layout2d::<f64>::zeros(n);
    let field_cfg = ScreenedPoissonConfig::default()
        .with_grid_size(64)
        .with_padding_factor(2);
    let cfg = ScreenedPoissonRefinerConfig::default()
        .with_field(field_cfg)
        .with_iterations(4)
        .with_seed(99);

    let mut r1 = ScreenedPoissonRefiner::<f64>::new(n, cfg.clone()).unwrap();
    let mut r2 = ScreenedPoissonRefiner::<f64>::new(n, cfg).unwrap();
    let mut l1 = l0.clone();
    let mut l2 = l0;
    r1.run(&mut l1).unwrap();
    r2.run(&mut l2).unwrap();
    assert_eq!(l1, l2);
    let (min_x, max_x, min_y, max_y) = bbox_of(&l1);
    assert!(max_x - min_x > 0.0 || max_y - min_y > 0.0);
}

#[test]
fn test_refiner_with_graph_anchors_and_coarse_tether() {
    let g =
        CsrGraph::<()>::from_unweighted_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0)], false, true)
            .unwrap();
    let mut layout = Layout2d::from_xy(vec![0.0, 0.1, 0.1, 0.0], vec![0.0, 0.0, 0.1, 0.1]).unwrap();
    let coarse = Layout2d::from_xy(vec![5.0, -5.0], vec![2.0, -2.0]).unwrap();
    let n2c = [0u32, 0u32, 1u32, 1u32];

    let field_cfg = ScreenedPoissonConfig::default()
        .with_grid_size(64)
        .with_padding_factor(2)
        .with_rho_max(Some(0.1));
    let cfg = ScreenedPoissonRefinerConfig::default()
        .with_field(field_cfg)
        .with_iterations(3)
        .with_anchor_weight(0.2)
        .with_gamma_tether(1.0)
        .with_recenter(false);
    let mut refiner = ScreenedPoissonRefiner::<f64>::with_graph(&g, cfg)
        .unwrap()
        .with_coarse_tether(&coarse, &n2c)
        .unwrap();
    refiner.run(&mut layout).unwrap();

    let cx0 = 0.5 * (layout.x()[0] + layout.x()[1]);
    let cy0 = 0.5 * (layout.y()[0] + layout.y()[1]);
    let cx1 = 0.5 * (layout.x()[2] + layout.x()[3]);
    let cy1 = 0.5 * (layout.y()[2] + layout.y()[3]);
    assert!((cx0 - 5.0).abs() < 1e-9);
    assert!((cy0 - 2.0).abs() < 1e-9);
    assert!((cx1 + 5.0).abs() < 1e-9);
    assert!((cy1 + 2.0).abs() < 1e-9);
}

#[test]
fn test_screened_poisson_refiner_validation_and_zero_step_contract() {
    let cfg = ScreenedPoissonRefinerConfig::default().with_iterations(2);
    let mut solver = ScreenedPoissonRefiner::<f64>::new(4, cfg.clone()).unwrap();

    let mut wrong_len = Layout2d::<f64>::zeros(3);
    assert!(matches!(
        solver.run_steps(&mut wrong_len, 0),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert_eq!(solver.status(), SolverStatus::Initialized);

    let mut non_finite = Layout2d::from_xy(vec![f64::NAN, 0.0, 0.0, 0.0], vec![0.0; 4]).unwrap();
    assert!(matches!(
        solver.run_steps(&mut non_finite, 0),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert_eq!(solver.status(), SolverStatus::Initialized);

    assert!(matches!(
        cfg.clone().with_step_size(f64::NAN).validate(),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        cfg.clone().with_step_size(0.0).validate(),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        cfg.clone().with_momentum(1.0).validate(),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        cfg.clone().with_gamma_tether(1.2).validate(),
        Err(CoreError::InvalidInput(_))
    ));
}
