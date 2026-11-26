//! Integration tests for Force-Directed Edge Bundling (`FdebSolver`).

use egraph_core::{CoreError, CsrGraph, Layout2d, SolverStatus};
use egraph_layout_force::fdeb::{FdebConfig, FdebSolver};

#[test]
fn test_fdeb_step_vs_run_collinear_zero_length_and_bundling() {
    // 1. Two parallel close edges bundle closer together, plus collinear, antiparallel, and zero-length edges.
    let graph = CsrGraph::<f64>::from_edges(
        8,
        &[
            (0, 1), // horizontal y=0
            (2, 3), // horizontal y=0.3 (parallel, same direction -> bundles!)
            (4, 5), // collinear antiparallel on y=0.3
            (6, 7), // zero-length edge (same coords!)
        ],
        None,
        true,
        false,
    )
    .unwrap();
    let layout64 = Layout2d::from_xy(
        vec![0.0_f64, 10.0, 0.0, 10.0, 10.0, 0.0, 5.0, 5.0],
        vec![0.0_f64, 0.0, 0.3, 0.3, 0.3, 0.3, 2.0, 2.0],
    )
    .unwrap();

    let cfg = FdebConfig {
        cycles: 4,
        i0: 30,
        ..Default::default()
    };

    // f64 bit-exact step == run
    let mut s64_run = FdebSolver::new(&graph, &layout64, cfg).unwrap();
    let mut s64_step = FdebSolver::new(&graph, &layout64, cfg).unwrap();
    assert_eq!(s64_run.status(), SolverStatus::Initialized);
    assert_eq!(s64_run.num_edges(), 4);
    assert_eq!(s64_run.edge_offsets(), &[0, 2, 4, 6, 8]);

    assert_eq!(s64_run.run().unwrap(), SolverStatus::Completed);
    assert!(s64_run.is_finished());
    assert_eq!(s64_run.iteration(), s64_run.total_iterations());

    let _ = s64_step.run_steps(7).unwrap();
    while !s64_step.is_finished() {
        s64_step.step().unwrap();
    }
    assert_eq!(s64_run.edge_offsets(), s64_step.edge_offsets());
    for i in 0..s64_run.points_x().len() {
        assert_eq!(
            s64_run.points_x()[i].to_bits(),
            s64_step.points_x()[i].to_bits()
        );
        assert_eq!(
            s64_run.points_y()[i].to_bits(),
            s64_step.points_y()[i].to_bits()
        );
        assert!(s64_run.points_x()[i].is_finite());
        assert!(s64_run.points_y()[i].is_finite());
    }

    // Verify bundling: midpoint of edge 0 (y=0) and edge 1 (y=0.3) moved closer than initial 0.3!
    let off = s64_run.edge_offsets();
    let mid0 = (off[0] as usize + off[1] as usize) / 2;
    let mid1 = (off[1] as usize + off[2] as usize) / 2;
    let y_gap = (s64_run.points_y()[mid1] - s64_run.points_y()[mid0]).abs();
    assert!(
        y_gap < 0.25,
        "expected edges to bundle closer, y_gap={y_gap}"
    );

    // Interleaved length matches 2 * points_x.len().
    assert_eq!(
        s64_run.to_interleaved_points().len(),
        s64_run.points_x().len() * 2
    );

    // f32 bit-exact step == run
    let layout32 = Layout2d::from_xy(
        layout64.x().iter().map(|&v| v as f32).collect(),
        layout64.y().iter().map(|&v| v as f32).collect(),
    )
    .unwrap();
    let mut s32_run = FdebSolver::new(&graph, &layout32, cfg).unwrap();
    let mut s32_step = FdebSolver::new(&graph, &layout32, cfg).unwrap();
    s32_run.run().unwrap();
    while !s32_step.is_finished() {
        s32_step.step().unwrap();
    }
    for i in 0..s32_run.points_x().len() {
        assert_eq!(
            s32_run.points_x()[i].to_bits(),
            s32_step.points_x()[i].to_bits()
        );
        assert_eq!(
            s32_run.points_y()[i].to_bits(),
            s32_step.points_y()[i].to_bits()
        );
        assert!(s32_run.points_x()[i].is_finite());
        assert!(s32_run.points_y()[i].is_finite());
    }

    // Input layout was passed by reference and remains untouched!
    assert_eq!(layout64.x()[0], 0.0);

    // Invalid config and dimension mismatch checks.
    assert!(matches!(
        FdebSolver::new(
            &graph,
            &layout64,
            FdebConfig {
                minimum_edge_compatibility: 1.5,
                ..Default::default()
            }
        ),
        Err(CoreError::InvalidInput(_))
    ));
    let wrong_layout = Layout2d::<f64>::zeros(2);
    assert!(matches!(
        FdebSolver::new(&graph, &wrong_layout, cfg),
        Err(CoreError::DimensionMismatch { .. })
    ));
}
