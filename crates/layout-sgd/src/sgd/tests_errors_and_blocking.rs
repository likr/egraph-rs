use egraph_core::{
    CoreError, CsrGraph, IterativeSolver, Layout2d, ManifoldKind, ManifoldLayout2d, SolverStatus,
};
use egraph_neighborhood::{DogVaradhanConfig, DogVaradhanLandmarks};

use super::super::sgd::{
    SgdScheduleConfig, SgdScheduleKind, SgdSolver, SparseBarrierConfig, StreamingNegativeConfig,
};
use super::pairs::{BLOCK_SIZE, PackedSymPair};
use super::test_support::circle_layout;

#[test]
fn test_adversarial_empty_singleton_oob_and_non_finite_errors() {
    let sched = SgdScheduleConfig::default();
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();

    for g in [&empty, &single] {
        let n = g.num_nodes();
        let mut f = SgdSolver::<f64>::full(g, sched, 42).unwrap();
        let mut rp = SgdSolver::<f64>::random_pair(g, 5, sched, 42).unwrap();
        let mut sn =
            SgdSolver::<f64>::streaming_negatives(g, StreamingNegativeConfig::default(), sched, 42)
                .unwrap();
        let emb = vec![0.0_f64; n * 2];
        let mut om = SgdSolver::<f64>::omega(g, &emb, 2, 5, 1e-3, sched, 42).unwrap();
        let (mut sb, st, _) = SgdSolver::<f64>::sparse_barrier_dog(
            g,
            &DogVaradhanConfig::default(),
            &SparseBarrierConfig::default(),
        )
        .unwrap();
        assert_eq!(st.n_pairs_1hop, 0);

        for solver in [&mut f, &mut rp, &mut sn, &mut om, &mut sb] {
            let mut l = Layout2d::<f64>::zeros(n);
            assert_eq!(solver.run(&mut l).unwrap(), SolverStatus::Completed);
            let mut m = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Torus2d, n);
            assert_eq!(
                solver.run_manifold(&mut m).unwrap(),
                SolverStatus::Completed
            );
        }
    }

    // Out-of-bounds, dimension mismatch, self-loop, and non-finite pair validation.
    assert!(matches!(
        SgdSolver::<f64>::from_pairs(
            2,
            vec![0],
            vec![2],
            vec![1.0],
            vec![1.0],
            vec![1.0],
            vec![1.0],
            sched,
            42
        ),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        SgdSolver::<f64>::from_pairs(
            2,
            vec![0],
            vec![0],
            vec![1.0],
            vec![1.0],
            vec![1.0],
            vec![1.0],
            sched,
            42
        ),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        SgdSolver::<f64>::from_pairs(
            2,
            vec![0],
            vec![1, 0],
            vec![1.0],
            vec![1.0],
            vec![1.0],
            vec![1.0],
            sched,
            42
        ),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        SgdSolver::<f64>::from_pairs(
            2,
            vec![0],
            vec![1],
            vec![f64::NAN],
            vec![1.0],
            vec![1.0],
            vec![1.0],
            sched,
            42
        ),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        SgdSolver::<f64>::from_pairs(
            2,
            vec![0],
            vec![1],
            vec![1.0],
            vec![1.0],
            vec![-0.5],
            vec![1.0],
            sched,
            42
        ),
        Err(CoreError::InvalidInput(_))
    ));

    // Schedule and config errors.
    assert!(matches!(
        SgdSolver::<f64>::full(
            &single,
            SgdScheduleConfig {
                eps: f64::NAN,
                ..sched
            },
            42
        ),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        SgdSolver::<f64>::full(
            &single,
            SgdScheduleConfig {
                step_scale: 0.0,
                ..sched
            },
            42
        ),
        Err(CoreError::InvalidInput(_))
    ));

    // Omega invalid dims / mismatched length / non-finite / min_dist <= 0.
    let tri = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, false, false).unwrap();
    assert!(matches!(
        SgdSolver::<f64>::omega(&tri, &[], 0, 2, 1e-3, sched, 42),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        SgdSolver::<f64>::omega(&tri, &[0.0; 4], 2, 2, 1e-3, sched, 42),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        SgdSolver::<f64>::omega(&tri, &[f64::INFINITY; 6], 2, 2, 1e-3, sched, 42),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        SgdSolver::<f64>::omega(&tri, &[0.0; 6], 2, 2, 0.0, sched, 42),
        Err(CoreError::InvalidInput(_))
    ));

    // Landmark mismatch, out-of-bounds pivot, and NaN landmark table.
    let lm_ok = DogVaradhanLandmarks {
        num_nodes: 3,
        landmarks: vec![0],
        d_dog: vec![0.0, 1.0, 2.0],
        d_coarse: None,
    };
    assert!(matches!(
        SgdSolver::<f64>::sparse_barrier_from_landmarks(
            &tri,
            &lm_ok,
            Some(&[5]),
            &SparseBarrierConfig::default()
        ),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    let lm_bad = DogVaradhanLandmarks {
        num_nodes: 3,
        landmarks: vec![0],
        d_dog: vec![0.0, f64::NAN, 2.0],
        d_coarse: None,
    };
    assert!(matches!(
        SgdSolver::<f64>::sparse_barrier_from_landmarks(
            &tri,
            &lm_bad,
            None,
            &SparseBarrierConfig::default()
        ),
        Err(CoreError::NonFiniteValue { .. })
    ));
    let lm_mismatch = DogVaradhanLandmarks {
        num_nodes: 2,
        landmarks: vec![0],
        d_dog: vec![0.0, 1.0],
        d_coarse: None,
    };
    assert!(matches!(
        SgdSolver::<f64>::sparse_barrier_from_landmarks(
            &tri,
            &lm_mismatch,
            None,
            &SparseBarrierConfig::default()
        ),
        Err(CoreError::DimensionMismatch { .. })
    ));

    // Layout size mismatch and non-finite layout during step / step_manifold.
    let mut s = SgdSolver::<f64>::full(&tri, sched, 42).unwrap();
    let mut wrong_len = Layout2d::<f64>::zeros(2);
    assert!(matches!(
        s.step(&mut wrong_len),
        Err(CoreError::DimensionMismatch { .. })
    ));
    let mut nan_layout = Layout2d::from_xy(vec![0.0, f64::NAN, 1.0], vec![0.0; 3]).unwrap();
    assert!(matches!(
        s.step(&mut nan_layout),
        Err(CoreError::NonFiniteValue { .. })
    ));
    let mut wrong_m = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Spherical, 2);
    assert!(matches!(
        s.step_manifold(&mut wrong_m),
        Err(CoreError::DimensionMismatch { .. })
    ));

    // Negative weights on graph must fail cleanly.
    let neg_wg = CsrGraph::<f64>::from_edges(3, &[(0, 1)], Some(&[-1.0]), false, false).unwrap();
    assert!(matches!(
        SgdSolver::<f64>::full(&neg_wg, sched, 42),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        SgdSolver::<f64>::random_pair(&neg_wg, 2, sched, 42),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        SgdSolver::<f64>::streaming_negatives(
            &neg_wg,
            StreamingNegativeConfig::default(),
            sched,
            42
        ),
        Err(CoreError::InvalidInput(_))
    ));
}

#[test]
fn test_cache_blocked_shuffle_over_threshold_step_eq_run() {
    let n = 400usize;
    let num_pairs = 70_000usize;
    let mut pairs64 = Vec::with_capacity(num_pairs);
    let mut pairs32 = Vec::with_capacity(num_pairs);
    let mut n_springs = 0usize;
    let mut n_hinges = 0usize;
    for idx in 0..num_pairs {
        let u = (idx % n) as u32;
        let step = 1 + (idx / n);
        let v = ((u as usize + step) % n) as u32;
        let is_hinge = idx % 5 == 0;
        if is_hinge {
            n_hinges += 1;
        } else {
            n_springs += 1;
        }
        let d = 1.0 + (step % 7) as f64 * 0.25;
        let w = 1.0 / (d * d);
        pairs64.push(PackedSymPair::new(u, v, is_hinge, d, w));
        pairs32.push(PackedSymPair::new(u, v, is_hinge, d as f32, w as f32));
    }

    let sched = SgdScheduleConfig {
        iterations: 6,
        eps: 0.05,
        kind: SgdScheduleKind::Exponential,
        step_scale: 0.2,
    };

    let mut s64_run =
        SgdSolver::<f64>::from_packed_symmetric(n, pairs64.clone(), n_springs, n_hinges, sched, 42)
            .unwrap();
    let mut s64_step =
        SgdSolver::<f64>::from_packed_symmetric(n, pairs64, n_springs, n_hinges, sched, 42)
            .unwrap();
    assert!(!s64_run.block_order.is_empty());
    assert_eq!(s64_run.block_order.len(), num_pairs.div_ceil(BLOCK_SIZE));

    let mut l64_run = circle_layout::<f64>(n, 2.0);
    let mut l64_step = l64_run.clone();
    s64_run.run(&mut l64_run).unwrap();
    s64_step.run_steps(&mut l64_step, 2).unwrap();
    while !s64_step.status().is_finished() {
        s64_step.step(&mut l64_step).unwrap();
    }
    assert!(l64_run.is_all_finite());
    for i in 0..n {
        assert_eq!(l64_run.x()[i].to_bits(), l64_step.x()[i].to_bits());
        assert_eq!(l64_run.y()[i].to_bits(), l64_step.y()[i].to_bits());
    }

    let mut s32_run =
        SgdSolver::<f32>::from_packed_symmetric(n, pairs32.clone(), n_springs, n_hinges, sched, 42)
            .unwrap();
    let mut s32_step =
        SgdSolver::<f32>::from_packed_symmetric(n, pairs32, n_springs, n_hinges, sched, 42)
            .unwrap();
    let mut l32_run = circle_layout::<f32>(n, 2.0);
    let mut l32_step = l32_run.clone();
    s32_run.run(&mut l32_run).unwrap();
    s32_step.run_steps(&mut l32_step, 3).unwrap();
    while !s32_step.status().is_finished() {
        s32_step.step(&mut l32_step).unwrap();
    }
    assert!(l32_run.is_all_finite());
    for i in 0..n {
        assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
        assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
    }
}
