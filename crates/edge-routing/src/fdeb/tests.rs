use egraph_core::{
    CoreError, CsrGraph, EdgePolylines, IterativeSolver, Layout2d, Layout3d, LayoutNd, SolverStatus,
};

use super::*;

#[test]
fn test_fdeb_config_builders_and_validation() {
    let cfg = FdebConfig::new()
        .with_cycles(4)
        .with_s0(0.2)
        .with_steps_per_cycle(15)
        .with_s_step(0.6)
        .with_i_step(0.75)
        .with_spring_k(0.05)
        .with_minimum_edge_compatibility(0.4);
    assert_eq!(cfg.cycles, 4);
    assert_eq!(cfg.s0, 0.2);
    assert_eq!(cfg.steps_per_cycle, 15);
    assert_eq!(cfg.s_step, 0.6);
    assert_eq!(cfg.i_step, 0.75);
    assert_eq!(cfg.spring_k, 0.05);
    assert_eq!(cfg.minimum_edge_compatibility, 0.4);
    cfg.validate().unwrap();

    assert!(matches!(
        FdebConfig::new().with_cycles(30).validate(),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        FdebConfig::new()
            .with_minimum_edge_compatibility(1.2)
            .validate(),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        FdebConfig::new().with_s0(-0.1).validate(),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        FdebConfig::new().with_spring_k(f64::NAN).validate(),
        Err(CoreError::NonFiniteValue { .. })
    ));
}

#[test]
fn test_fdeb_short_edge_and_dense_bundle_stability_caps() {
    // 1. Short compatible edges where naive alpha * kp = 0.5 * 1.0 / (1 * 1e-3) = 500.0 >> 0.5.
    let g_short =
        CsrGraph::<()>::from_unweighted_edges(4, &[(0, 1), (2, 3)], false, false).unwrap();
    let layout_short = Layout2d::from_xy(
        vec![0.0_f64, 1e-3, 0.0, 1e-3],
        vec![0.0_f64, 0.0, 2e-4, 2e-4],
    )
    .unwrap();
    let mut poly_short = EdgePolylines::straight_line_from_graph(&g_short, &layout_short).unwrap();
    let cfg_short = FdebConfig::new()
        .with_cycles(3)
        .with_s0(0.5)
        .with_steps_per_cycle(20)
        .with_spring_k(1.0)
        .with_minimum_edge_compatibility(0.5);
    let mut solver_short = FdebSolver::new(&poly_short, cfg_short).unwrap();
    solver_short.run(&mut poly_short).unwrap();
    for (&x, &y) in poly_short
        .points()
        .x()
        .iter()
        .zip(poly_short.points().y().iter())
    {
        assert!(x.is_finite() && y.is_finite());
        assert!(
            (-1e-3..=2e-3).contains(&x),
            "short edge x out of bounds: {x}"
        );
        assert!(
            (-1e-3..=2e-3).contains(&y),
            "short edge y out of bounds: {y}"
        );
    }

    // 2. Dense bundle of 50 parallel short edges: single-step displacement must not exceed
    // ep.length / (num_mid + 1).
    let num_edges = 50usize;
    let edges: Vec<(u32, u32)> = (0..num_edges)
        .map(|i| ((2 * i) as u32, (2 * i + 1) as u32))
        .collect();
    let g_dense =
        CsrGraph::<()>::from_unweighted_edges(num_edges * 2, &edges, false, false).unwrap();
    let mut xs = Vec::with_capacity(num_edges * 2);
    let mut ys = Vec::with_capacity(num_edges * 2);
    for i in 0..num_edges {
        let y_off = (i as f64) * 1e-4;
        xs.push(0.0);
        ys.push(y_off);
        xs.push(1e-2);
        ys.push(y_off);
    }
    let layout_dense = Layout2d::from_xy(xs, ys).unwrap();
    let mut poly_dense = EdgePolylines::straight_line_from_graph(&g_dense, &layout_dense).unwrap();
    let cfg_dense = FdebConfig::new()
        .with_cycles(3)
        .with_s0(1.0)
        .with_steps_per_cycle(15)
        .with_spring_k(1.0)
        .with_minimum_edge_compatibility(0.5);
    let mut solver_dense = FdebSolver::new(&poly_dense, cfg_dense).unwrap();
    solver_dense.step(&mut poly_dense).unwrap();
    let pts_per_edge = (poly_dense.edge_offsets()[1] - poly_dense.edge_offsets()[0]) as usize;
    let max_disp = 1e-2 / ((pts_per_edge - 1) as f64);
    for e in 0..num_edges {
        let y_base = (e as f64) * 1e-4;
        let start = poly_dense.edge_offsets()[e] as usize;
        let mid_y = poly_dense.points().y()[start + 1];
        assert!(mid_y.is_finite());
        assert!(
            (mid_y - y_base).abs() <= max_disp + 1e-12,
            "dense bundle midpoint moved {} > max_disp {max_disp}",
            (mid_y - y_base).abs()
        );
    }
    solver_dense.run(&mut poly_dense).unwrap();
    for (&x, &y) in poly_dense
        .points()
        .x()
        .iter()
        .zip(poly_dense.points().y().iter())
    {
        assert!(x.is_finite() && y.is_finite());
        assert!((-1e-2..=2e-2).contains(&x));
        assert!((-1e-2..=2e-2).contains(&y));
    }
}

#[test]
fn test_fdeb_numerical_breakdown_transitions_to_failed_without_corrupting_target() {
    // 1. f32 overflow test: finite in f64 during init and step, overflows f32 on writeback.
    let pts32 = Layout2d::from_xy(
        vec![3.0e38_f32, 3.0e38_f32, 3.1e38_f32, 3.1e38_f32],
        vec![-3.0e38_f32, 3.0e38_f32, -3.0e38_f32, 3.0e38_f32],
    )
    .unwrap();
    let mut poly32 = EdgePolylines::from_parts(&[0, 2, 4], pts32).unwrap();
    let original32 = poly32.clone();
    let cfg32 = FdebConfig::new()
        .with_cycles(2)
        .with_s0(1.0e38)
        .with_steps_per_cycle(4)
        .with_minimum_edge_compatibility(0.0);
    let mut solver32 = FdebSolver::new(&poly32, cfg32).unwrap();
    let res32 = solver32.step(&mut poly32);
    assert!(matches!(res32, Err(CoreError::NumericalBreakdown(_))));
    assert_eq!(solver32.status(), SolverStatus::Failed);
    assert!(solver32.status().is_failed());
    assert!(solver32.is_finished());
    assert_eq!(solver32.iteration(), 0);
    assert_eq!(poly32, original32);
    assert_eq!(solver32.step(&mut poly32).unwrap(), SolverStatus::Failed);

    // 2. f64 velocity overflow test: four close parallel edges with s0 = 1e308 overflow
    // velocity accumulation cleanly to NumericalBreakdown and SolverStatus::Failed.
    let pts64 = Layout2d::from_xy(
        vec![0.0_f64, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0],
        vec![0.0_f64, 0.0, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03],
    )
    .unwrap();
    let mut poly64 = EdgePolylines::from_parts(&[0, 2, 4, 6, 8], pts64).unwrap();
    let original64 = poly64.clone();
    let cfg64 = FdebConfig::new()
        .with_cycles(2)
        .with_s0(1.0e308)
        .with_steps_per_cycle(5)
        .with_minimum_edge_compatibility(0.0);
    let mut solver64 = FdebSolver::new(&poly64, cfg64).unwrap();
    let res64 = solver64.run(&mut poly64);
    assert!(matches!(res64, Err(CoreError::NumericalBreakdown(_))));
    assert_eq!(solver64.status(), SolverStatus::Failed);
    assert_eq!(poly64, original64);
    assert_eq!(solver64.step(&mut poly64).unwrap(), SolverStatus::Failed);

    // 3D Layout3d constructor & basic compatibility check.
    let pts3 = Layout3d::from_xyz(
        vec![0.0, 1.0, 0.0, 1.0],
        vec![0.0, 0.0, 0.1, 0.1],
        vec![0.0, 0.0, 0.1, 0.1],
    )
    .unwrap();
    let poly3 = EdgePolylines::from_parts(&[0, 2, 4], pts3).unwrap();
    let s3 = FdebSolver::new(&poly3, FdebConfig::new()).unwrap();
    assert_eq!(s3.dims(), 3);
    assert_eq!(s3.num_compatible_pairs(), 1);

    // LayoutNd constructor check.
    let pts4 = LayoutNd::from_flat(
        4,
        4,
        &[
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0,
        ],
    )
    .unwrap();
    let poly4 = EdgePolylines::from_parts(&[0, 2, 4], pts4).unwrap();
    let s4 = FdebSolver::new(&poly4, FdebConfig::new()).unwrap();
    assert_eq!(s4.dims(), 4);
    assert_eq!(s4.num_compatible_pairs(), 1);
}
