//! Integration tests for Force-Directed Edge Bundling (`FdebSolver`) over `EdgePolylines`.

use egraph_core::{
    CoreError, CsrGraph, EdgePolylines, IterativeSolver, Layout2d, Layout3d, LayoutNd, SolverStatus,
};
use egraph_edge_routing::{FdebConfig, FdebSolver};

#[test]
fn test_fdeb_2d_step_vs_run_subdivision_and_bundling() {
    // Two parallel close edges bundle closer together, plus collinear antiparallel and zero-length edges.
    let graph = CsrGraph::<()>::from_unweighted_edges(
        8,
        &[
            (0, 1), // horizontal y = 0
            (2, 3), // horizontal y = 0.3 (parallel, same direction -> bundles!)
            (4, 5), // collinear antiparallel on y = 0.3
            (6, 7), // zero-length edge (same coords!)
        ],
        true,
        false,
    )
    .unwrap();
    let layout64 = Layout2d::from_xy(
        vec![0.0_f64, 10.0, 0.0, 10.0, 10.0, 0.0, 5.0, 5.0],
        vec![0.0_f64, 0.0, 0.3, 0.3, 0.3, 0.3, 2.0, 2.0],
    )
    .unwrap();
    let mut poly64_run = EdgePolylines::straight_line_from_graph(&graph, &layout64).unwrap();
    let mut poly64_step = poly64_run.clone();

    let cfg = FdebConfig::new().with_cycles(4).with_steps_per_cycle(30);

    // f64 bit-exact step == run
    let mut s64_run = FdebSolver::new(&poly64_run, cfg.clone()).unwrap();
    let mut s64_step = FdebSolver::new(&poly64_step, cfg.clone()).unwrap();
    assert_eq!(s64_run.status(), SolverStatus::Initialized);
    assert_eq!(s64_run.dims(), 2);
    assert_eq!(s64_run.num_edges(), 4);
    assert_eq!(s64_run.pts_per_edge(), 2);
    assert_eq!(poly64_run.edge_offsets(), &[0, 2, 4, 6, 8]);

    let expected_total = IterativeSolver::total_iterations(&s64_run).unwrap();
    assert!(expected_total > 0);
    assert_eq!(s64_run.total_iterations(), Some(expected_total));

    assert_eq!(
        s64_run.run(&mut poly64_run).unwrap(),
        SolverStatus::Completed
    );
    assert!(s64_run.is_finished());
    assert_eq!(s64_run.iteration(), expected_total);
    assert_eq!(s64_run.pts_per_edge(), (1usize << 4) + 1);
    assert_eq!(poly64_run.edge_offsets(), &[0, 17, 34, 51, 68]);

    // Verify subdivision progression across cycles on step-by-step solver.
    assert_eq!(
        s64_step.run_steps(&mut poly64_step, 1).unwrap(),
        SolverStatus::Running
    );
    assert_eq!(poly64_step.edge_offsets(), &[0, 3, 6, 9, 12]);
    // Run remaining steps of cycle 0 (30 steps total) and step into cycle 1.
    let _ = s64_step.run_steps(&mut poly64_step, 29).unwrap();
    assert_eq!(poly64_step.edge_offsets(), &[0, 3, 6, 9, 12]);
    let _ = s64_step.step(&mut poly64_step).unwrap();
    assert_eq!(poly64_step.edge_offsets(), &[0, 5, 10, 15, 20]);

    while !s64_step.is_finished() {
        s64_step.step(&mut poly64_step).unwrap();
    }
    assert_eq!(poly64_run.edge_offsets(), poly64_step.edge_offsets());
    for i in 0..poly64_run.total_points() {
        assert_eq!(
            poly64_run.points().x()[i].to_bits(),
            poly64_step.points().x()[i].to_bits()
        );
        assert_eq!(
            poly64_run.points().y()[i].to_bits(),
            poly64_step.points().y()[i].to_bits()
        );
        assert!(poly64_run.points().x()[i].is_finite());
        assert!(poly64_run.points().y()[i].is_finite());
    }

    // Endpoints remain pinned to original node positions.
    let off = poly64_run.edge_offsets();
    assert_eq!(poly64_run.points().x()[off[0] as usize], 0.0);
    assert_eq!(poly64_run.points().x()[off[1] as usize - 1], 10.0);

    // Verify bundling: midpoint of edge 0 (y=0) and edge 1 (y=0.3) moved closer than initial 0.3.
    let mid0 = (off[0] as usize + off[1] as usize) / 2;
    let mid1 = (off[1] as usize + off[2] as usize) / 2;
    let y_gap = (poly64_run.points().y()[mid1] - poly64_run.points().y()[mid0]).abs();
    assert!(
        y_gap < 0.25,
        "expected edges to bundle closer, y_gap={y_gap}"
    );

    // f32 bit-exact step == run
    let layout32 = Layout2d::from_xy(
        layout64.x().iter().map(|&v| v as f32).collect(),
        layout64.y().iter().map(|&v| v as f32).collect(),
    )
    .unwrap();
    let mut poly32_run = EdgePolylines::straight_line_from_graph(&graph, &layout32).unwrap();
    let mut poly32_step = poly32_run.clone();
    let mut s32_run = FdebSolver::new(&poly32_run, cfg.clone()).unwrap();
    let mut s32_step = FdebSolver::new(&poly32_step, cfg).unwrap();
    s32_run.run(&mut poly32_run).unwrap();
    let _ = s32_step.run_steps(&mut poly32_step, 11).unwrap();
    while !s32_step.is_finished() {
        s32_step.step(&mut poly32_step).unwrap();
    }
    assert_eq!(poly32_run.edge_offsets(), poly32_step.edge_offsets());
    for i in 0..poly32_run.total_points() {
        assert_eq!(
            poly32_run.points().x()[i].to_bits(),
            poly32_step.points().x()[i].to_bits()
        );
        assert_eq!(
            poly32_run.points().y()[i].to_bits(),
            poly32_step.points().y()[i].to_bits()
        );
    }
}

#[test]
fn test_fdeb_3d_and_nd_bundling_and_blanket_dispatch() {
    let graph =
        CsrGraph::<()>::from_unweighted_edges(8, &[(0, 1), (2, 3), (4, 5), (6, 7)], true, false)
            .unwrap();
    let layout3d_64 = Layout3d::from_xyz(
        vec![0.0_f64, 10.0, 0.0, 10.0, 10.0, 0.0, 5.0, 5.0],
        vec![0.0_f64, 0.0, 0.3, 0.3, 0.3, 0.3, 2.0, 2.0],
        vec![0.0_f64, 0.0, 0.3, 0.3, 0.3, 0.3, 2.0, 2.0],
    )
    .unwrap();
    let cfg = FdebConfig::new().with_cycles(4).with_steps_per_cycle(25);

    // Layout3d<f64> bit-exact step == run
    let mut poly3_run = EdgePolylines::straight_line_from_graph(&graph, &layout3d_64).unwrap();
    let mut poly3_step = poly3_run.clone();
    let mut s3_run = FdebSolver::new(&poly3_run, cfg.clone()).unwrap();
    let mut s3_step = FdebSolver::new(&poly3_step, cfg.clone()).unwrap();
    assert_eq!(s3_run.run(&mut poly3_run).unwrap(), SolverStatus::Completed);
    s3_step.run_steps(&mut poly3_step, 9).unwrap();
    while !s3_step.is_finished() {
        s3_step.step(&mut poly3_step).unwrap();
    }
    assert_eq!(s3_run.dims(), 3);
    assert_eq!(poly3_run.edge_offsets(), poly3_step.edge_offsets());
    for i in 0..poly3_run.total_points() {
        assert_eq!(
            poly3_run.points().x()[i].to_bits(),
            poly3_step.points().x()[i].to_bits()
        );
        assert_eq!(
            poly3_run.points().y()[i].to_bits(),
            poly3_step.points().y()[i].to_bits()
        );
        assert_eq!(
            poly3_run.points().z()[i].to_bits(),
            poly3_step.points().z()[i].to_bits()
        );
    }
    let off = poly3_run.edge_offsets();
    let mid0 = (off[0] as usize + off[1] as usize) / 2;
    let mid1 = (off[1] as usize + off[2] as usize) / 2;
    let yz_gap = ((poly3_run.points().y()[mid1] - poly3_run.points().y()[mid0]).powi(2)
        + (poly3_run.points().z()[mid1] - poly3_run.points().z()[mid0]).powi(2))
    .sqrt();
    assert!(
        yz_gap < 0.35,
        "expected 3D edge bundling reduction, yz_gap={yz_gap}"
    );

    // Layout3d<f32> bit-exact step == run
    let layout3d_32 = Layout3d::from_xyz(
        layout3d_64.x().iter().map(|&v| v as f32).collect(),
        layout3d_64.y().iter().map(|&v| v as f32).collect(),
        layout3d_64.z().iter().map(|&v| v as f32).collect(),
    )
    .unwrap();
    let mut poly3_32_run = EdgePolylines::straight_line_from_graph(&graph, &layout3d_32).unwrap();
    let mut poly3_32_step = poly3_32_run.clone();
    let mut s3_32_run = FdebSolver::new(&poly3_32_run, cfg.clone()).unwrap();
    let mut s3_32_step = FdebSolver::new(&poly3_32_step, cfg.clone()).unwrap();
    s3_32_run.run(&mut poly3_32_run).unwrap();
    while !s3_32_step.is_finished() {
        s3_32_step.step(&mut poly3_32_step).unwrap();
    }
    for i in 0..poly3_32_run.total_points() {
        assert_eq!(
            poly3_32_run.points().x()[i].to_bits(),
            poly3_32_step.points().x()[i].to_bits()
        );
        assert_eq!(
            poly3_32_run.points().y()[i].to_bits(),
            poly3_32_step.points().y()[i].to_bits()
        );
        assert_eq!(
            poly3_32_run.points().z()[i].to_bits(),
            poly3_32_step.points().z()[i].to_bits()
        );
    }

    // LayoutNd<f64> 4D bundling and blanket IterativeSolver dispatch (`&mut dyn` and `Box<dyn>`).
    let layout4d = LayoutNd::<f64>::init(8, 4, 888, 5.0).unwrap();
    let mut poly4_run = EdgePolylines::straight_line_from_graph(&graph, &layout4d).unwrap();
    let mut poly4_dyn = poly4_run.clone();
    let mut s4 = FdebSolver::new(&poly4_run, cfg.clone()).unwrap();
    assert_eq!(s4.dims(), 4);
    assert_eq!(s4.run(&mut poly4_run).unwrap(), SolverStatus::Completed);

    let mut s4_inner = FdebSolver::new(&poly4_dyn, cfg.clone()).unwrap();
    let mut s4_ref: &mut dyn IterativeSolver<Target = EdgePolylines<LayoutNd<f64>>> = &mut s4_inner;
    assert_eq!(
        (&mut s4_ref).run_steps(&mut poly4_dyn, 5).unwrap(),
        SolverStatus::Running
    );
    let mut s4_box: Box<dyn IterativeSolver<Target = EdgePolylines<LayoutNd<f64>>>> =
        Box::new(s4_inner);
    assert_eq!(s4_box.run(&mut poly4_dyn).unwrap(), SolverStatus::Completed);
    assert_eq!(poly4_run, poly4_dyn);

    // LayoutNd<f64> with dims == 2 matches Layout2d<f64> bundling numerically.
    let mut flat2 = Vec::with_capacity(16);
    for i in 0..8 {
        flat2.push(layout3d_64.x()[i]);
        flat2.push(layout3d_64.y()[i]);
    }
    let layout_nd2 = LayoutNd::from_flat(8, 2, &flat2).unwrap();
    let mut poly_nd2 = EdgePolylines::straight_line_from_graph(&graph, &layout_nd2).unwrap();
    let mut s_nd2 = FdebSolver::new(&poly_nd2, cfg).unwrap();
    assert_eq!(s_nd2.run(&mut poly_nd2).unwrap(), SolverStatus::Completed);
    assert_eq!(poly_nd2.points().dims(), 2);
    poly_nd2.validate_finite().unwrap();
}

#[test]
fn test_fdeb_boundaries_validation_and_zero_step_guards() {
    // 1. Empty (N=0, E=0) and edgeless (N=5, E=0) graphs.
    for n in [0usize, 5] {
        let g = CsrGraph::<()>::from_unweighted_edges(n, &[], false, false).unwrap();
        let l3 = Layout3d::<f64>::zeros(n);
        let mut poly = EdgePolylines::straight_line_from_graph(&g, &l3).unwrap();
        let mut solver = FdebSolver::new(&poly, FdebConfig::new()).unwrap();
        assert_eq!(
            solver.run_steps(&mut poly, 0).unwrap(),
            SolverStatus::Initialized
        );
        assert_eq!(solver.run(&mut poly).unwrap(), SolverStatus::Converged);
        assert!(solver.is_finished());
        assert_eq!(poly.num_edges(), 0);
        assert_eq!(poly.total_points(), 0);
    }

    // 2. Single edge (E=1) remains straight across cycles.
    let g_single = CsrGraph::<()>::from_unweighted_edges(2, &[(0, 1)], false, false).unwrap();
    let l_single = Layout2d::from_xy(vec![0.0, 4.0], vec![1.0, 5.0]).unwrap();
    let mut poly_single = EdgePolylines::straight_line_from_graph(&g_single, &l_single).unwrap();
    let mut s_single = FdebSolver::new(
        &poly_single,
        FdebConfig::new().with_cycles(3).with_steps_per_cycle(10),
    )
    .unwrap();
    assert_eq!(
        s_single.run(&mut poly_single).unwrap(),
        SolverStatus::Completed
    );
    assert_eq!(poly_single.edge_offsets(), &[0, 9]);
    for i in 0..9 {
        let t = (i as f64) / 8.0;
        assert!((poly_single.points().x()[i] - 4.0 * t).abs() < 1e-12);
        assert!((poly_single.points().y()[i] - (1.0 + 4.0 * t)).abs() < 1e-12);
    }

    // 3. Zero-iteration configs (`cycles=0`, `steps_per_cycle=0`) start Completed and leave target intact.
    let g_two = CsrGraph::<()>::from_unweighted_edges(4, &[(0, 1), (2, 3)], false, false).unwrap();
    let l_two = Layout2d::from_xy(vec![0.0, 5.0, 0.0, 5.0], vec![0.0, 0.0, 0.5, 0.5]).unwrap();
    let poly_two = EdgePolylines::straight_line_from_graph(&g_two, &l_two).unwrap();
    for cfg in [
        FdebConfig::new().with_cycles(0),
        FdebConfig::new().with_steps_per_cycle(0),
    ] {
        let mut work = poly_two.clone();
        let mut solver = FdebSolver::new(&work, cfg).unwrap();
        assert_eq!(solver.status(), SolverStatus::Completed);
        assert_eq!(solver.total_iterations(), Some(0));
        assert_eq!(solver.run(&mut work).unwrap(), SolverStatus::Completed);
        assert_eq!(work, poly_two);
    }

    // 4. Already-subdivided polylines (3 points per edge) work cleanly.
    let mut sub_poly = poly_two.with_subdivisions(1).unwrap();
    let mut sub_solver = FdebSolver::new(
        &sub_poly,
        FdebConfig::new().with_cycles(2).with_steps_per_cycle(10),
    )
    .unwrap();
    assert_eq!(sub_solver.pts_per_edge(), 3);
    assert_eq!(
        sub_solver.run(&mut sub_poly).unwrap(),
        SolverStatus::Completed
    );
    assert_eq!(sub_solver.pts_per_edge(), 9);
    assert_eq!(sub_poly.edge_offsets(), &[0, 9, 18]);

    // 5. Invalid polylines rejected at construction (`< 2` points, non-uniform points, NaN).
    let pts_1 = Layout2d::from_xy(vec![0.0, 1.0, 2.0], vec![0.0, 0.0, 0.0]).unwrap();
    let poly_short = EdgePolylines::from_parts(&[0, 1, 3], pts_1.clone()).unwrap();
    assert!(matches!(
        FdebSolver::new(&poly_short, FdebConfig::new()),
        Err(CoreError::InvalidInput(_))
    ));
    let pts_5 = Layout2d::<f64>::zeros(5);
    let poly_nonuniform = EdgePolylines::from_parts(&[0, 2, 5], pts_5).unwrap();
    assert!(matches!(
        FdebSolver::new(&poly_nonuniform, FdebConfig::new()),
        Err(CoreError::InvalidInput(_))
    ));
    let mut poly_nan = poly_two.clone();
    poly_nan.points_mut().x_mut()[1] = f64::NAN;
    assert!(matches!(
        FdebSolver::new(&poly_nan, FdebConfig::new()),
        Err(CoreError::NonFiniteValue { .. })
    ));

    // 6. `run_steps(target, 0)` validates target mismatches without mutating solver or target.
    let l_nd3 = LayoutNd::<f64>::init(4, 3, 17, 2.0).unwrap();
    let poly_nd3 = EdgePolylines::straight_line_from_graph(&g_two, &l_nd3).unwrap();
    let mut guard = FdebSolver::new(&poly_nd3, FdebConfig::new()).unwrap();

    // 6a. LayoutNd dims mismatch (3D vs 4D, same num_edges and total_points).
    let l_nd4 = LayoutNd::<f64>::init(4, 4, 17, 2.0).unwrap();
    let mut bad_dims = EdgePolylines::straight_line_from_graph(&g_two, &l_nd4).unwrap();
    let bad_dims_orig = bad_dims.clone();
    assert!(matches!(
        guard.run_steps(&mut bad_dims, 0),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert_eq!(guard.status(), SolverStatus::Initialized);
    assert_eq!(bad_dims, bad_dims_orig);

    // 6b. Edge count mismatch.
    let mut bad_edges = EdgePolylines::straight_line_from_graph(
        &g_single,
        &LayoutNd::<f64>::init(2, 3, 1, 1.0).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        guard.run_steps(&mut bad_edges, 0),
        Err(CoreError::DimensionMismatch { .. })
    ));

    // 6c. Total points mismatch (same num_edges, wrong points len).
    let mut bad_total = poly_nd3.with_subdivisions(1).unwrap();
    assert!(matches!(
        guard.run_steps(&mut bad_total, 0),
        Err(CoreError::DimensionMismatch { .. })
    ));

    // 6d. Offset mismatch (same num_edges=2, same total_points=4, wrong offsets [0, 1, 4]).
    let mut bad_offsets =
        EdgePolylines::from_parts(&[0, 1, 4], LayoutNd::<f64>::zeros(4, 3).unwrap()).unwrap();
    assert!(matches!(
        guard.run_steps(&mut bad_offsets, 0),
        Err(CoreError::InvalidInput(_))
    ));

    // 6e. Non-finite coordinates in target.
    let mut bad_nan = poly_nd3.clone();
    bad_nan.points_mut().coords_mut()[2] = f64::INFINITY;
    assert!(matches!(
        guard.run_steps(&mut bad_nan, 0),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert_eq!(guard.status(), SolverStatus::Initialized);
}
