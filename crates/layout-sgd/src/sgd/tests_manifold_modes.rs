use egraph_core::{
    IterativeSolver, Layout2d, LayoutGenerator, ManifoldKind, ManifoldLayout2d, Scalar,
    SolverStatus,
};
use egraph_linalg_mds::KorenSpectral;
use egraph_neighborhood::{DogVaradhanConfig, compute_regularized_dog_varadhan_landmarks};

use super::super::sgd::{
    OrtmannSparseStressConfig, Sgd2Config, Sgd2Solver, SgdPairBuffer, SgdScheduleConfig, SgdSolver,
    SparseBarrierConfig, StreamingNegativeConfig,
};
use super::test_support::grid_graph;
use super::tests_nd::build_mode_pair;

const ALL_MANIFOLDS: [ManifoldKind; 4] = [
    ManifoldKind::HyperbolicPoincare,
    ManifoldKind::HyperbolicLorentz,
    ManifoldKind::Spherical,
    ManifoldKind::Torus2d,
];

fn assert_manifold_step_eq_run<S: Scalar>(
    mut s_run: SgdSolver<ManifoldLayout2d<S>>,
    mut s_step: SgdSolver<ManifoldLayout2d<S>>,
    kind: ManifoldKind,
    n: usize,
    label: &str,
) {
    let mut l_run = ManifoldLayout2d::<S>::init(kind, n, 42, 0.35).unwrap();
    let mut l_step = l_run.clone();

    let st_run = s_run.run(&mut l_run).unwrap();
    assert_eq!(st_run, SolverStatus::Completed, "{label}");

    let first_k = (s_step.total_iterations().unwrap() / 3).max(1);
    s_step.run_steps(&mut l_step, first_k).unwrap();
    if !s_step.status().is_finished() {
        s_step.step(&mut l_step).unwrap();
    }
    let st_step = s_step.run(&mut l_step).unwrap();
    assert_eq!(st_step, SolverStatus::Completed, "{label}");
    assert_eq!(s_run.iteration(), s_step.iteration(), "{label}");
    assert!(l_run.as_layout().is_all_finite(), "{label}: finite");

    for i in 0..n {
        assert_eq!(
            l_run.x()[i].to_f64().to_bits(),
            l_step.x()[i].to_f64().to_bits(),
            "{label} {kind:?} x[{i}]"
        );
        assert_eq!(
            l_run.y()[i].to_f64().to_bits(),
            l_step.y()[i].to_f64().to_bits(),
            "{label} {kind:?} y[{i}]"
        );
    }
}

#[test]
fn test_all_sgd_and_sgd2_constructors_manifold_bit_exact_f64_and_f32() {
    let g = grid_graph(3, 4);
    let n = g.num_nodes();
    let dog_cfg = DogVaradhanConfig::new().with_n_landmarks(6).with_seed(42);
    let (landmarks, _) = compute_regularized_dog_varadhan_landmarks(&g, &dog_cfg).unwrap();
    let spec: Layout2d<f64> = KorenSpectral::new(&g).generate().unwrap();
    let mut emb_flat = Vec::with_capacity(n * 2);
    for i in 0..n {
        emb_flat.push(spec.x()[i]);
        emb_flat.push(spec.y()[i]);
    }

    for kind in ALL_MANIFOLDS {
        for mode in 0..11 {
            let (s_run, s_step, label) =
                build_mode_pair::<ManifoldLayout2d<f64>>(mode, &g, &landmarks, &emb_flat);
            assert_manifold_step_eq_run::<f64>(s_run, s_step, kind, n, label);

            let (s32_run, s32_step, label32) =
                build_mode_pair::<ManifoldLayout2d<f32>>(mode, &g, &landmarks, &emb_flat);
            assert_manifold_step_eq_run::<f32>(s32_run, s32_step, kind, n, label32);
        }

        let s2_cfg = Sgd2Config::new()
            .with_iterations(12)
            .with_batch_size(5)
            .with_seed(42);
        let mut s2_run = Sgd2Solver::<ManifoldLayout2d<f64>>::new(&g, s2_cfg.clone()).unwrap();
        let mut s2_step = Sgd2Solver::<ManifoldLayout2d<f64>>::new(&g, s2_cfg.clone()).unwrap();
        let mut l2_run = ManifoldLayout2d::<f64>::init(kind, n, 42, 0.35).unwrap();
        let mut l2_step = l2_run.clone();
        s2_run.run(&mut l2_run).unwrap();
        s2_step.run_steps(&mut l2_step, 4).unwrap();
        s2_step.run(&mut l2_step).unwrap();
        assert_eq!(l2_run, l2_step, "sgd2 f64 {kind:?}");

        let mut s2_32_run = Sgd2Solver::<ManifoldLayout2d<f32>>::new(&g, s2_cfg.clone()).unwrap();
        let mut s2_32_step = Sgd2Solver::<ManifoldLayout2d<f32>>::new(&g, s2_cfg).unwrap();
        let mut l2_32_run = ManifoldLayout2d::<f32>::init(kind, n, 42, 0.35).unwrap();
        let mut l2_32_step = l2_32_run.clone();
        s2_32_run.run(&mut l2_32_run).unwrap();
        s2_32_step.run_steps(&mut l2_32_step, 4).unwrap();
        s2_32_step.run(&mut l2_32_step).unwrap();
        assert_eq!(l2_32_run, l2_32_step, "sgd2 f32 {kind:?}");
    }
}

#[test]
fn test_all_modes_reset_schedule_and_windowed_manifold_determinism() {
    let g = grid_graph(3, 4);
    let n = g.num_nodes();
    let dog_cfg = DogVaradhanConfig::new().with_n_landmarks(4).with_seed(7);
    let (landmarks, _) = compute_regularized_dog_varadhan_landmarks(&g, &dog_cfg).unwrap();
    let spec: Layout2d<f64> = KorenSpectral::new(&g).generate().unwrap();
    let mut emb = Vec::with_capacity(n * 2);
    for i in 0..n {
        emb.push(spec.x()[i]);
        emb.push(spec.y()[i]);
    }

    let s_init = SgdScheduleConfig::new(8);
    let s_reset = SgdScheduleConfig::new(10)
        .with_step_scale(0.6)
        .with_eps(0.05);
    let sb_cfg = SparseBarrierConfig::default()
        .with_num_pivots(4)
        .with_schedule(s_init.clone())
        .with_seed(7);

    let make_all = || -> Vec<(SgdSolver<ManifoldLayout2d<f64>>, &'static str)> {
        let mut buf = SgdPairBuffer::new();
        buf.push_spring(0, 1, 1.0, 1.0).unwrap();
        vec![
            (SgdSolver::full(&g, s_init.clone(), 7).unwrap(), "full"),
            (
                SgdSolver::from_pairs(
                    n,
                    &[0, 1],
                    &[1, 2],
                    &[1.0, 1.0],
                    &[1.0, 1.0],
                    &[1.0, 1.0],
                    &[1.0, 1.0],
                    s_init.clone(),
                    7,
                )
                .unwrap(),
                "from_pairs",
            ),
            (
                SgdSolver::from_pair_buffer(n, buf, s_init.clone(), 7).unwrap(),
                "from_pair_buffer",
            ),
            (
                SgdSolver::random_pair(&g, 3, s_init.clone(), 7).unwrap(),
                "random_pair",
            ),
            (
                SgdSolver::omega(&g, &emb, 2, 3, 1e-3, s_init.clone(), 7).unwrap(),
                "omega",
            ),
            (
                SgdSolver::sparse_barrier_from_landmarks(&g, &landmarks, None, &sb_cfg)
                    .unwrap()
                    .0,
                "sparse_barrier_from_landmarks",
            ),
            (
                SgdSolver::sparse_barrier_dog(&g, &dog_cfg, &sb_cfg)
                    .unwrap()
                    .0,
                "sparse_barrier_dog",
            ),
            (
                SgdSolver::streaming_negatives(
                    &g,
                    StreamingNegativeConfig::default(),
                    s_init.clone(),
                    7,
                )
                .unwrap(),
                "streaming_negatives",
            ),
            (
                SgdSolver::ortmann_sparse_stress(
                    &g,
                    OrtmannSparseStressConfig::default()
                        .with_num_pivots(4)
                        .with_iterations(8)
                        .with_seed(7),
                )
                .unwrap(),
                "ortmann_sparse_stress",
            ),
        ]
    };

    let mut set_a = make_all();
    let mut set_b = make_all();
    for ((sa, label), (sb, _)) in set_a.iter_mut().zip(set_b.iter_mut()) {
        let mut ma = ManifoldLayout2d::<f64>::init(ManifoldKind::Torus2d, n, 11, 1.0).unwrap();
        let mut mb = ma.clone();

        sa.run_schedule_window(&mut ma, 0, 3).unwrap();
        sb.run_schedule_window(&mut mb, 0, 3).unwrap();
        assert_eq!(ma, mb, "{label}: pre-reset window");

        sa.reset_schedule(s_reset.clone(), 55).unwrap();
        sb.reset_schedule(s_reset.clone(), 55).unwrap();
        assert_eq!(sa.total_iterations(), Some(10), "{label}");
        assert_eq!(sa.iteration(), 0, "{label}");

        sa.run_schedule_window(&mut ma, 0, 4).unwrap();
        sa.run_schedule_window(&mut ma, 4, 6).unwrap();
        sb.run(&mut mb).unwrap();
        assert_eq!(ma, mb, "{label}: post-reset windowed == full run");
    }
}
