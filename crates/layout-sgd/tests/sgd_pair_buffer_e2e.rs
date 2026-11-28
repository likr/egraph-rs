//! Opaque-box E2E tests for `SgdPairBuffer::iter`, `SgdPairRecord`, `SgdPairKind`,
//! and bit-packing encapsulation.

use egraph_core::{CoreError, IterativeSolver, Layout2d, SolverStatus};
use egraph_layout_sgd::{SgdPairBuffer, SgdPairKind, SgdPairRecord, SgdScheduleConfig, SgdSolver};

#[test]
fn pair_buffer_iter_preserves_order_kinds_and_counts() {
    let mut buf = SgdPairBuffer::<f64>::with_capacity(4);
    buf.push_spring(0, 1, 1.0, 2.0).unwrap();
    buf.push_hinge(1, 2, 1.5, 0.5).unwrap();
    buf.push_spring(2, 3, 2.5, 0.75).unwrap();
    buf.push_hinge(0, 3, 3.0, 1.25).unwrap();

    let records: Vec<SgdPairRecord<f64>> = buf.iter().collect();
    assert_eq!(records.len(), buf.len());
    assert_eq!(
        records
            .iter()
            .filter(|r| r.kind == SgdPairKind::Spring)
            .count(),
        buf.num_springs()
    );
    assert_eq!(
        records
            .iter()
            .filter(|r| r.kind == SgdPairKind::Hinge)
            .count(),
        buf.num_hinges()
    );

    assert_eq!(
        (records[0].u, records[0].v, records[0].kind),
        (0, 1, SgdPairKind::Spring)
    );
    assert_eq!((records[0].d_uv, records[0].d_vu), (1.0, 1.0));
    assert_eq!((records[0].w_uv, records[0].w_vu), (2.0, 2.0));

    assert_eq!(
        (records[1].u, records[1].v, records[1].kind),
        (1, 2, SgdPairKind::Hinge)
    );
    assert_eq!((records[1].d_uv, records[1].w_uv), (1.5, 0.5));

    assert_eq!(
        (records[3].u, records[3].v, records[3].kind),
        (0, 3, SgdPairKind::Hinge)
    );
}

#[test]
fn pair_buffer_iter_reflects_condition_weights() {
    let mut buf = SgdPairBuffer::<f64>::new();
    buf.push_spring(0, 1, 1.0, 1000.0).unwrap();
    buf.push_spring(1, 2, 1.0, 10.0).unwrap();
    buf.push_hinge(2, 3, 2.0, 1.0).unwrap();

    buf.condition_weights(10.0).unwrap();
    let records: Vec<SgdPairRecord<f64>> = buf.iter().collect();
    let mut w_min = f64::INFINITY;
    let mut w_max = 0.0_f64;
    for r in &records {
        w_min = w_min.min(r.w_uv);
        w_max = w_max.max(r.w_uv);
    }
    assert!(w_max / w_min <= 10.0 + 1e-12);
    assert_eq!(
        records
            .iter()
            .filter(|r| r.kind == SgdPairKind::Hinge)
            .count(),
        1
    );
}

#[test]
fn pair_buffer_iter_after_clear_and_repopulate() {
    let mut buf = SgdPairBuffer::<f64>::new();
    buf.push_spring(0, 1, 1.0, 1.0).unwrap();
    buf.push_hinge(1, 2, 2.0, 0.5).unwrap();
    buf.clear();
    assert_eq!(buf.iter().count(), 0);
    assert!(buf.is_empty());

    buf.push_hinge(3, 4, 1.25, 0.75).unwrap();
    let records: Vec<_> = buf.iter().collect();
    assert_eq!(records.len(), 1);
    assert_eq!(
        (records[0].u, records[0].v, records[0].kind),
        (3, 4, SgdPairKind::Hinge)
    );
    assert_eq!((records[0].d_uv, records[0].w_uv), (1.25, 0.75));
}

#[test]
fn pair_buffer_iter_f32_precision() {
    let mut buf = SgdPairBuffer::<f32>::new();
    buf.push_spring(0, 1, 1.25_f32, 0.625_f32).unwrap();
    buf.push_hinge(1, 2, 2.5_f32, 0.125_f32).unwrap();
    let records: Vec<SgdPairRecord<f32>> = buf.iter().collect();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].d_uv, 1.25_f32);
    assert_eq!(records[0].w_uv, 0.625_f32);
    assert_eq!(records[1].kind, SgdPairKind::Hinge);
    assert_eq!(records[1].d_uv, 2.5_f32);
    assert_eq!(records[1].w_uv, 0.125_f32);
}

#[test]
fn pair_buffer_non_destructive_before_from_pair_buffer() {
    let mut buf = SgdPairBuffer::<f64>::new();
    buf.push_spring(0, 1, 1.0, 1.0).unwrap();
    buf.push_hinge(1, 2, 1.5, 0.5).unwrap();

    let snap_a: Vec<_> = buf.iter().collect();
    let snap_b: Vec<_> = buf.iter().collect();
    assert_eq!(snap_a, snap_b);

    let solver =
        SgdSolver::<Layout2d<f64>>::from_pair_buffer(3, buf, SgdScheduleConfig::new(5), 42)
            .unwrap();
    let from_solver: Vec<_> = solver.iter_pairs().collect();
    assert_eq!(from_solver, snap_a);
}

#[test]
fn pair_buffer_empty_iter_yields_nothing() {
    let buf = SgdPairBuffer::<f64>::new();
    assert!(buf.is_empty());
    assert_eq!(buf.len(), 0);
    assert_eq!(buf.iter().count(), 0);
}

#[test]
fn pair_buffer_rejects_invalid_pairs_without_mutation() {
    let mut buf = SgdPairBuffer::<f64>::new();
    buf.push_spring(0, 1, 1.0, 1.0).unwrap();
    let before = buf.len();

    for res in [
        buf.push_spring(2, 2, 1.0, 1.0),
        buf.push_hinge(0, 1, 0.0, 1.0),
        buf.push_spring(0, 1, -1.0, 1.0),
        buf.push_hinge(0, 1, 1.0, -0.1),
        buf.push_spring(1u32 << 31, 0, 1.0, 1.0),
    ] {
        assert!(matches!(res, Err(CoreError::InvalidInput(_))));
    }
    assert!(matches!(
        buf.push_spring(0, 1, f64::NAN, 1.0),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        buf.push_hinge(0, 1, 1.0, f64::INFINITY),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert_eq!(buf.len(), before);
}

#[test]
fn pair_buffer_condition_weights_validation_and_zero_weight_noop() {
    let mut buf = SgdPairBuffer::<f64>::new();
    buf.push_spring(0, 1, 1.0, 0.0).unwrap();
    buf.push_hinge(1, 2, 2.0, 100.0).unwrap();
    assert!(matches!(
        buf.condition_weights(0.5),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        buf.condition_weights(f64::NAN),
        Err(CoreError::NonFiniteValue { .. })
    ));
    buf.condition_weights(5.0).unwrap();
    let recs: Vec<_> = buf.iter().collect();
    assert_eq!(recs[0].w_uv, 0.0);
    assert_eq!(recs[1].w_uv, 100.0);
}

#[test]
fn pair_buffer_single_pair_condition_weights_noop() {
    let mut buf = SgdPairBuffer::<f64>::new();
    buf.push_spring(0, 1, 1.5, 2.5).unwrap();
    buf.condition_weights(2.0).unwrap();
    let recs: Vec<_> = buf.iter().collect();
    assert_eq!(recs.len(), 1);
    assert_eq!((recs[0].d_uv, recs[0].w_uv), (1.5, 2.5));
}

#[test]
fn pair_buffer_accepts_max_index_mask_u_and_v_and_rejects_msb() {
    let max_idx = (1u32 << 31) - 1;
    let invalid_idx = 1u32 << 31;
    let mut buf = SgdPairBuffer::<f64>::new();
    buf.push_hinge(max_idx, 0, 1.0, 1.0).unwrap();
    buf.push_hinge(0, max_idx, 1.0, 1.0).unwrap();
    let rec: Vec<_> = buf.iter().collect();
    assert_eq!((rec[0].u, rec[0].kind), (max_idx, SgdPairKind::Hinge));
    assert_eq!((rec[1].v, rec[1].kind), (max_idx, SgdPairKind::Hinge));
    assert!(matches!(
        buf.push_spring(invalid_idx, 0, 1.0, 1.0),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        buf.push_hinge(0, invalid_idx, 1.0, 1.0),
        Err(CoreError::InvalidInput(_))
    ));
}

#[test]
fn pair_buffer_distinguishes_spring_and_hinge_on_same_endpoints() {
    let mut buf = SgdPairBuffer::<f64>::new();
    buf.push_spring(0, 1, 1.0, 1.0).unwrap();
    buf.push_hinge(0, 1, 2.0, 0.5).unwrap();
    assert_eq!(buf.len(), 2);
    assert_eq!((buf.num_springs(), buf.num_hinges()), (1, 1));
    let yielded: Vec<_> = buf.iter().collect();
    assert_eq!(yielded[0].kind, SgdPairKind::Spring);
    assert_eq!(yielded[1].kind, SgdPairKind::Hinge);
}

#[test]
fn pair_record_and_kind_support_non_exhaustive_wildcard_and_opaque_roundtrip() {
    let mut buf = SgdPairBuffer::<f64>::new();
    buf.push_spring(0, 1, 1.0, 2.0).unwrap();
    buf.push_hinge(1, 2, 1.5, 0.5).unwrap();
    let recs: Vec<_> = buf.iter().collect();
    for r in &recs {
        let is_spring = match r.kind {
            SgdPairKind::Spring => true,
            SgdPairKind::Hinge => false,
            _ => unreachable!("unknown SgdPairKind"),
        };
        assert_eq!(is_spring, r.u == 0);
    }
    let mut solver =
        SgdSolver::<Layout2d<f64>>::from_pair_buffer(3, buf, SgdScheduleConfig::new(4), 11)
            .unwrap();
    assert_eq!(solver.iter_pairs().collect::<Vec<_>>(), recs);
    let mut l2 = Layout2d::<f64>::init(3, 12, 1.0).unwrap();
    assert_eq!(solver.run(&mut l2).unwrap(), SolverStatus::Completed);
}
