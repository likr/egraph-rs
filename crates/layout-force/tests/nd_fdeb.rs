//! Integration tests for 3D and N-D Force-Directed Edge Bundling (`FdebSolver`).

use egraph_core::{CoreError, CsrGraph, IterativeSolver, Layout2d, LayoutNd, SolverStatus};
use egraph_layout_force::fdeb::{FdebConfig, FdebSolver};

#[test]
fn test_fdeb_nd_3d_4d_bundling_and_accessors() {
    let graph =
        CsrGraph::<f64>::from_edges(8, &[(0, 1), (2, 3), (4, 5), (6, 7)], None, true, false)
            .unwrap();
    let init3_64 = LayoutNd::from_flat(
        8,
        3,
        vec![
            0.0, 0.0, 0.0, 10.0, 0.0, 0.0, // parallel e0
            0.0, 0.3, 0.3, 10.0, 0.3, 0.3, // parallel e1 in 3D diagonal
            10.0, 0.3, 0.3, 0.0, 0.3, 0.3, // collinear antiparallel
            5.0, 2.0, 2.0, 5.0, 2.0, 2.0, // zero-length
        ],
    )
    .unwrap();
    let init3_32 =
        LayoutNd::from_flat(8, 3, init3_64.coords().iter().map(|&v| v as f32).collect()).unwrap();
    let cfg = FdebConfig {
        cycles: 4,
        steps_per_cycle: 25,
        ..Default::default()
    };

    // f64 bit-exact step == run
    let mut s_run = FdebSolver::<f64>::new(&graph, cfg).unwrap();
    assert_eq!(s_run.point_dims(), 0);
    assert!(matches!(
        s_run.points_axis(0),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    let mut s_step = FdebSolver::<f64>::new(&graph, cfg).unwrap();
    let mut l_run = init3_64.copy();
    let mut l_step = init3_64.copy();
    assert_eq!(s_run.run(&mut l_run).unwrap(), SolverStatus::Completed);
    s_step.run_steps(&mut l_step, 9).unwrap();
    while !s_step.status().is_finished() {
        s_step.step(&mut l_step).unwrap();
    }
    assert_eq!(s_run.point_dims(), 3);
    assert_eq!(s_run.edge_offsets(), s_step.edge_offsets());
    assert_eq!(s_run.points_nd().len(), s_step.points_nd().len());
    for (&a, &b) in s_run.points_nd().iter().zip(s_step.points_nd().iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
        assert!(a.is_finite());
    }
    assert_eq!(s_run.to_interleaved_points_nd(), s_run.points_nd());
    let xs = s_run.points_axis(0).unwrap();
    let ys = s_run.points_axis(1).unwrap();
    let zs = s_run.points_z().unwrap();
    assert_eq!(xs.len(), s_run.points_nd().len() / 3);
    assert!(matches!(
        s_run.points_axis(3),
        Err(CoreError::IndexOutOfBounds { .. })
    ));

    // Midpoint of e0 and e1 moved closer in (y, z) than initial gap sqrt(0.3^2 + 0.3^2) = 0.424.
    let off = s_run.edge_offsets();
    let mid0 = (off[0] as usize + off[1] as usize) / 2;
    let mid1 = (off[1] as usize + off[2] as usize) / 2;
    let yz_gap = ((ys[mid1] - ys[mid0]).powi(2) + (zs[mid1] - zs[mid0]).powi(2)).sqrt();
    assert!(
        yz_gap < 0.35,
        "expected 3D edge bundling reduction, yz_gap={yz_gap}"
    );

    // f32 bit-exact step == run
    let mut s32_run = FdebSolver::<f32>::new(&graph, cfg).unwrap();
    let mut s32_step = FdebSolver::<f32>::new(&graph, cfg).unwrap();
    let mut l32_run = init3_32.copy();
    let mut l32_step = init3_32.copy();
    s32_run.run(&mut l32_run).unwrap();
    while !s32_step.status().is_finished() {
        s32_step.step(&mut l32_step).unwrap();
    }
    for (&a, &b) in s32_run.points_nd().iter().zip(s32_step.points_nd().iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    // 4D bundling works and exposes axis 3.
    let mut s4 = FdebSolver::<f64>::new(&graph, cfg).unwrap();
    let mut l4 = LayoutNd::<f64>::init(8, 4, 888, 5.0).unwrap();
    assert_eq!(s4.run(&mut l4).unwrap(), SolverStatus::Completed);
    assert_eq!(s4.point_dims(), 4);
    assert_eq!(s4.points_axis(3).unwrap().len(), s4.points_nd().len() / 4);

    // Layout2d sets point_dims=2 and rejects points_z().
    let mut s2 = FdebSolver::<f64>::new(&graph, cfg).unwrap();
    let mut l2 = Layout2d::<f64>::init(8, 42, 2.0).unwrap();
    s2.run(&mut l2).unwrap();
    assert_eq!(s2.point_dims(), 2);
    assert!(matches!(
        s2.points_z(),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    assert_eq!(s2.to_interleaved_points_nd(), s2.to_interleaved_points());
}

#[test]
fn test_fdeb_nd_degenerates_and_error_guards() {
    for n in [0usize, 1] {
        let g = CsrGraph::<f64>::from_edges(n, &[], None, false, false).unwrap();
        let mut l = LayoutNd::<f64>::zeros(n, 3).unwrap();
        let mut s = FdebSolver::<f64>::new(&g, FdebConfig::default()).unwrap();
        assert!(s.run(&mut l).unwrap().is_finished());
        assert_eq!(s.point_dims(), 3);
    }

    let g_edgeless = CsrGraph::<f64>::from_edges(5, &[], None, false, false).unwrap();
    let mut l_zero = LayoutNd::<f64>::zeros(5, 3).unwrap();
    assert!(
        FdebSolver::<f64>::new(&g_edgeless, FdebConfig::default())
            .unwrap()
            .run(&mut l_zero)
            .unwrap()
            .is_finished()
    );

    let g_disc =
        CsrGraph::<f64>::from_edges(6, &[(0, 1), (1, 2), (3, 4)], None, false, false).unwrap();
    let mut fdeb = FdebSolver::<f64>::new(&g_disc, FdebConfig::default()).unwrap();
    let mut bad_len = LayoutNd::<f64>::zeros(2, 3).unwrap();
    assert!(matches!(
        fdeb.step(&mut bad_len),
        Err(CoreError::DimensionMismatch { .. })
    ));

    let mut bad_nan = LayoutNd::from_flat(6, 3, {
        let mut v = vec![0.0; 18];
        v[2] = f64::NAN;
        v
    })
    .unwrap();
    assert!(matches!(
        fdeb.step(&mut bad_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));

    let mut l3 = LayoutNd::<f64>::init(6, 3, 2, 1.0).unwrap();
    let mut l4 = LayoutNd::<f64>::init(6, 4, 1, 1.0).unwrap();
    fdeb.step(&mut l3).unwrap();
    assert!(matches!(
        fdeb.step(&mut l4),
        Err(CoreError::DimensionMismatch { .. })
    ));
}
