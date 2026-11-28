//! Opaque-box E2E tests for `SgdSolver` Problem IR accessors, blocking threshold,
//! and pair-buffer construction boundaries.

use egraph_core::{CoreError, CsrGraph, IterativeSolver, Layout2d, SolverStatus};
use egraph_layout_sgd::{
    SgdPairBuffer, SgdPairKind, SgdScheduleConfig, SgdScheduleKind, SgdSolver, SparseBarrierConfig,
    StreamingNegativeConfig,
};
use egraph_neighborhood::DogVaradhanConfig;

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
fn pair_buffer_large_batch_preserves_insertion_order() {
    let count = 65_600usize;
    let n_nodes = 512u32;
    let mut buf = SgdPairBuffer::<f32>::with_capacity(count);
    for idx in 0..count {
        let u = (idx as u32) % n_nodes;
        let v = (u + 1 + ((idx as u32) / n_nodes)) % n_nodes;
        let v = if v == u { (u + 1) % n_nodes } else { v };
        if idx % 3 == 0 {
            buf.push_hinge(u, v, 1.5, 0.5).unwrap();
        } else {
            buf.push_spring(u, v, 1.0, 1.0).unwrap();
        }
    }
    let collected: Vec<_> = buf.iter().collect();
    assert_eq!(collected.len(), count);
    assert_eq!(collected[0].kind, SgdPairKind::Hinge);
    assert_eq!(collected[1].kind, SgdPairKind::Spring);

    let solver = SgdSolver::<Layout2d<f32>>::from_pair_buffer(
        n_nodes as usize,
        buf,
        SgdScheduleConfig::new(2),
        9,
    )
    .unwrap();
    assert_eq!(solver.iter_pairs().count(), count);
}

#[test]
fn solver_from_pair_buffer_accessors_match_schedule_and_pairs() {
    let mut buf = SgdPairBuffer::<f64>::new();
    buf.push_spring(0, 1, 1.0, 2.0).unwrap();
    buf.push_spring(1, 2, 2.0, 0.5).unwrap();
    buf.push_hinge(0, 2, 1.5, 1.0).unwrap();
    let expected_pairs: Vec<_> = buf.iter().collect();

    let sched = SgdScheduleConfig::new(12)
        .with_eps(0.05)
        .with_kind(SgdScheduleKind::Linear)
        .with_step_scale(1.25);
    let solver = SgdSolver::<Layout2d<f64>>::from_pair_buffer(3, buf, sched.clone(), 42).unwrap();

    assert_eq!(solver.schedule_config(), &sched);
    let (eta_min, eta_max) = solver.eta_bounds();
    assert!((eta_max - 1.0 / 0.5).abs() < 1e-12);
    assert!((eta_min - 0.05 / 2.0).abs() < 1e-12);
    assert_eq!(solver.iter_pairs().collect::<Vec<_>>(), expected_pairs);
}

#[test]
fn solver_from_pair_buffer_rejects_out_of_bounds_vertices() {
    let mut buf_u = SgdPairBuffer::<f64>::new();
    buf_u.push_spring(4, 1, 1.0, 1.0).unwrap();
    assert!(matches!(
        SgdSolver::<Layout2d<f64>>::from_pair_buffer(4, buf_u, SgdScheduleConfig::new(4), 1),
        Err(CoreError::IndexOutOfBounds {
            index: 4,
            len: 4,
            ..
        })
    ));

    let mut buf_v = SgdPairBuffer::<f64>::new();
    buf_v.push_hinge(1, 7, 1.0, 1.0).unwrap();
    assert!(matches!(
        SgdSolver::<Layout2d<f64>>::from_pair_buffer(4, buf_v, SgdScheduleConfig::new(4), 1),
        Err(CoreError::IndexOutOfBounds {
            index: 7,
            len: 4,
            ..
        })
    ));
}

#[test]
fn solver_full_and_random_pair_iter_pairs_invariants() {
    let g = make_grid(3, 3);
    let sched = SgdScheduleConfig::new(10);
    for solver in [
        SgdSolver::<Layout2d<f64>>::full(&g, sched.clone(), 7).unwrap(),
        SgdSolver::<Layout2d<f64>>::random_pair(&g, 4, sched, 7).unwrap(),
    ] {
        let pairs: Vec<_> = solver.iter_pairs().collect();
        assert_eq!(pairs.len(), solver.num_springs());
        assert_eq!(solver.num_hinges(), 0);
        for p in &pairs {
            assert_eq!(p.kind, SgdPairKind::Spring);
            assert_ne!(p.u, p.v);
            assert!(p.d_uv.is_finite() && p.d_uv > 0.0);
            assert!(p.w_uv.is_finite() && p.w_uv > 0.0);
        }
    }
}

#[test]
fn solver_sparse_barrier_dog_iter_pairs_matches_stats() {
    let g = make_grid(4, 4);
    let dog_cfg = DogVaradhanConfig::default()
        .with_n_landmarks(6)
        .with_seed(42);
    let barrier_cfg = SparseBarrierConfig::default()
        .with_num_pivots(4)
        .with_schedule(SgdScheduleConfig::new(8))
        .with_seed(42);
    let (solver, stats, _dog_stats) =
        SgdSolver::<Layout2d<f64>>::sparse_barrier_dog(&g, &dog_cfg, &barrier_cfg).unwrap();

    let pairs: Vec<_> = solver.iter_pairs().collect();
    assert_eq!(pairs.len(), stats.n_pairs_two_sided + stats.n_pairs_hinge);
    assert_eq!(
        pairs
            .iter()
            .filter(|p| p.kind == SgdPairKind::Spring)
            .count(),
        solver.num_springs()
    );
    assert_eq!(
        pairs
            .iter()
            .filter(|p| p.kind == SgdPairKind::Hinge)
            .count(),
        solver.num_hinges()
    );
}

#[test]
fn solver_from_pairs_with_hinges_asymmetric_iter_pairs() {
    let sched = SgdScheduleConfig::new(6);
    #[rustfmt::skip]
    let solver = SgdSolver::<Layout2d<f64>>::from_pairs_with_hinges(
        3, &[0], &[1], &[1.0], &[2.0], &[1.0], &[0.25],
        &[1], &[2], &[1.5], &[1.5], &[0.5], &[0.0], sched, 13,
    ).unwrap();

    let pairs: Vec<_> = solver.iter_pairs().collect();
    assert_eq!(pairs.len(), 2);
    assert_eq!(pairs[0].kind, SgdPairKind::Spring);
    assert_eq!((pairs[0].d_uv, pairs[0].d_vu), (1.0, 2.0));
    assert_eq!((pairs[0].w_uv, pairs[0].w_vu), (1.0, 0.25));
    assert_eq!(pairs[1].kind, SgdPairKind::Hinge);
    assert_eq!((pairs[1].w_uv, pairs[1].w_vu), (0.5, 0.0));
}

#[test]
fn solver_reset_schedule_updates_config_and_eta_bounds() {
    let g = make_grid(2, 3);
    let mut solver =
        SgdSolver::<Layout2d<f64>>::full(&g, SgdScheduleConfig::new(10).with_eps(0.01), 42)
            .unwrap();
    let (emin_old, emax_old) = solver.eta_bounds();
    let mut layout = Layout2d::<f64>::init(6, 5, 1.0).unwrap();
    solver.step(&mut layout).unwrap();
    assert_eq!(solver.iteration(), 1);

    let new_sched = SgdScheduleConfig::new(20)
        .with_eps(0.25)
        .with_kind(SgdScheduleKind::Quadratic)
        .with_step_scale(0.5);
    solver.reset_schedule(new_sched.clone(), 99).unwrap();

    assert_eq!(solver.schedule_config(), &new_sched);
    assert_eq!(solver.iteration(), 0);
    let (emin_new, emax_new) = solver.eta_bounds();
    assert!((emax_new - emax_old).abs() < 1e-12);
    assert!((emin_new - 25.0 * emin_old).abs() < 1e-12);
}

#[test]
fn solver_empty_and_singleton_graphs_have_zero_pairs_and_bounds() {
    for n in [0usize, 1, 4] {
        let g = CsrGraph::<f64>::from_edges(n, &[], None, false, false).unwrap();
        let solver = SgdSolver::<Layout2d<f64>>::full(&g, SgdScheduleConfig::new(5), 7).unwrap();
        assert_eq!(solver.iter_pairs().count(), 0);
        assert_eq!(solver.eta_bounds(), (0.0, 0.0));
    }
}

#[test]
fn solver_streaming_negatives_incorporates_stream_weight_in_eta_bounds() {
    let g = make_grid(2, 3);
    let neg_cfg = StreamingNegativeConfig::new()
        .with_negatives_per_node(2)
        .with_target_distance(2.0)
        .with_weight(0.05);
    let sched = SgdScheduleConfig::new(8).with_eps(0.01);
    let solver = SgdSolver::<Layout2d<f64>>::streaming_negatives(&g, neg_cfg, sched, 11).unwrap();
    assert_eq!(solver.iter_pairs().count(), g.num_edges());
    let (emin, emax) = solver.eta_bounds();
    assert!((emax - 1.0 / 0.05).abs() < 1e-12);
    assert!((emin - 0.01 / 1.0).abs() < 1e-12);
}

#[test]
fn solver_zero_iterations_and_zero_weight_pairs_bounds() {
    let g = make_grid(2, 2);
    let s_zero = SgdSolver::<Layout2d<f64>>::full(&g, SgdScheduleConfig::new(0), 1).unwrap();
    assert_eq!(s_zero.status(), SolverStatus::Completed);
    assert_eq!(s_zero.schedule_config().iterations, 0);
    assert_eq!(s_zero.current_eta(), 0.0);
    let (emin, emax) = s_zero.eta_bounds();
    assert_eq!(s_zero.schedule_config().evaluate(0, emin, emax), 0.0);

    let mut buf = SgdPairBuffer::<f64>::new();
    buf.push_spring(0, 1, 1.0, 0.0).unwrap();
    buf.push_hinge(1, 2, 2.0, 0.0).unwrap();
    let s_w0 =
        SgdSolver::<Layout2d<f64>>::from_pair_buffer(3, buf, SgdScheduleConfig::new(4), 2).unwrap();
    assert_eq!(s_w0.eta_bounds(), (0.0, 0.0));
    assert_eq!(s_w0.current_eta(), 0.0);
    assert_eq!(s_w0.iter_pairs().count(), 2);
}

#[test]
fn pair_buffer_and_solver_preserve_zero_weight_spring_and_hinge_kinds() {
    let mut buf = SgdPairBuffer::<f64>::new();
    buf.push_spring(0, 1, 1.0, 0.0).unwrap();
    buf.push_hinge(0, 1, 2.0, 0.0).unwrap();
    let mut solver =
        SgdSolver::<Layout2d<f64>>::from_pair_buffer(3, buf, SgdScheduleConfig::new(3), 7).unwrap();
    let yielded: Vec<_> = solver.iter_pairs().collect();
    assert_eq!(yielded.len(), 2);
    assert_eq!(
        (yielded[0].kind, yielded[0].w_uv),
        (SgdPairKind::Spring, 0.0)
    );
    assert_eq!(
        (yielded[1].kind, yielded[1].w_uv),
        (SgdPairKind::Hinge, 0.0)
    );
    let mut l2 = Layout2d::<f64>::init(3, 9, 1.0).unwrap();
    assert_eq!(solver.run(&mut l2).unwrap(), SolverStatus::Completed);
}
