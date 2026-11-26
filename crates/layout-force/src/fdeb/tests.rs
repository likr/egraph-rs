use egraph_core::{CsrGraph, IterativeSolver, Layout2d};

use super::*;

#[test]
fn test_fdeb_short_edge_and_dense_bundle_stability_caps() {
    // 1. Short compatible edges where naive alpha * kp = 0.5 * 1.0 / (1 * 1e-3) = 500.0 >> 0.5.
    let g_short = CsrGraph::<f64>::from_edges(4, &[(0, 1), (2, 3)], None, false, false).unwrap();
    let mut layout_short =
        Layout2d::from_xy(vec![0.0, 1e-3, 0.0, 1e-3], vec![0.0, 0.0, 2e-4, 2e-4]).unwrap();
    let cfg_short = FdebConfig {
        cycles: 3,
        s0: 0.5,
        steps_per_cycle: 20,
        spring_k: 1.0,
        minimum_edge_compatibility: 0.5,
        ..Default::default()
    };
    let mut solver_short = FdebSolver::<f64>::new(&g_short, cfg_short).unwrap();
    solver_short.run(&mut layout_short).unwrap();
    for (&x, &y) in solver_short
        .points_x()
        .iter()
        .zip(solver_short.points_y().iter())
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
    let g_dense = CsrGraph::<f64>::from_edges(num_edges * 2, &edges, None, false, false).unwrap();
    let mut xs = Vec::with_capacity(num_edges * 2);
    let mut ys = Vec::with_capacity(num_edges * 2);
    for i in 0..num_edges {
        let y_off = (i as f64) * 1e-4;
        xs.push(0.0);
        ys.push(y_off);
        xs.push(1e-2);
        ys.push(y_off);
    }
    let mut layout_dense = Layout2d::from_xy(xs, ys).unwrap();
    let cfg_dense = FdebConfig {
        cycles: 3,
        s0: 1.0,
        steps_per_cycle: 15,
        spring_k: 1.0,
        minimum_edge_compatibility: 0.5,
        ..Default::default()
    };
    let mut solver_dense = FdebSolver::<f64>::new(&g_dense, cfg_dense).unwrap();
    solver_dense.step(&mut layout_dense).unwrap();
    let pts_per_edge = (solver_dense.edge_offsets()[1] - solver_dense.edge_offsets()[0]) as usize;
    let max_disp = 1e-2 / ((pts_per_edge - 1) as f64);
    for e in 0..num_edges {
        let y_base = (e as f64) * 1e-4;
        let start = solver_dense.edge_offsets()[e] as usize;
        let mid_y = solver_dense.points_y()[start + 1];
        assert!(mid_y.is_finite());
        assert!(
            (mid_y - y_base).abs() <= max_disp + 1e-12,
            "dense bundle midpoint moved {} > max_disp {max_disp}",
            (mid_y - y_base).abs()
        );
    }
    solver_dense.run(&mut layout_dense).unwrap();
    for (&x, &y) in solver_dense
        .points_x()
        .iter()
        .zip(solver_dense.points_y().iter())
    {
        assert!(x.is_finite() && y.is_finite());
        assert!((-1e-2..=2e-2).contains(&x));
        assert!((-1e-2..=2e-2).contains(&y));
    }
}
