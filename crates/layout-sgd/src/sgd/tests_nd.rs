use egraph_core::{
    CsrGraph, IterativeSolver, Layout2d, Layout3d, LayoutGenerator, LayoutNd, RiemannianLayout,
    Scalar, SolverStatus,
};
use egraph_linalg_mds::KorenSpectral;
use egraph_neighborhood::{
    DogVaradhanConfig, DogVaradhanLandmarks, compute_regularized_dog_varadhan_landmarks,
};

use super::super::sgd::{
    OrtmannSparseStressConfig, Sgd2Config, Sgd2Solver, SgdPairBuffer, SgdScheduleConfig,
    SgdScheduleKind, SgdSolver, SparseBarrierConfig, StreamingNegativeConfig,
};
use super::test_support::{barbell_graph, grid_graph};

pub(super) trait ScalarBits: Scalar {
    fn bits(self) -> u64;
}

impl ScalarBits for f32 {
    fn bits(self) -> u64 {
        self.to_bits() as u64
    }
}

impl ScalarBits for f64 {
    fn bits(self) -> u64 {
        self.to_bits()
    }
}

fn layout_2d_to_nd<S: Scalar>(l2: &Layout2d<S>) -> LayoutNd<S> {
    let n = l2.len();
    let mut flat = Vec::with_capacity(n * 2);
    for i in 0..n {
        flat.push(l2.x()[i]);
        flat.push(l2.y()[i]);
    }
    LayoutNd::from_flat(n, 2, &flat).unwrap()
}

pub(super) fn assert_step_eq_run_nd<S: ScalarBits>(
    mut s_run: SgdSolver<LayoutNd<S>>,
    mut s_step: SgdSolver<LayoutNd<S>>,
    init: &LayoutNd<S>,
    step_chunk: usize,
    label: &str,
) {
    let mut l_run = init.clone();
    let mut l_step = init.clone();
    assert_eq!(s_run.status(), SolverStatus::Initialized, "{label}");
    assert_eq!(
        s_run.run(&mut l_run).unwrap(),
        SolverStatus::Completed,
        "{label}"
    );

    let total = s_step.total_iterations().unwrap();
    if step_chunk > 0 && total > step_chunk {
        assert_eq!(
            s_step.run_steps(&mut l_step, step_chunk).unwrap(),
            SolverStatus::Running,
            "{label}"
        );
        assert_eq!(s_step.iteration(), step_chunk, "{label}");
    }
    while !s_step.status().is_finished() {
        s_step.step(&mut l_step).unwrap();
    }
    assert_eq!(s_step.status(), SolverStatus::Completed, "{label}");
    assert_eq!(s_step.iteration(), total, "{label}");
    assert_eq!(
        s_step.step(&mut l_step).unwrap(),
        SolverStatus::Completed,
        "{label}"
    );
    assert!(l_run.is_all_finite(), "{label} not finite");

    for (idx, (&a, &b)) in l_run
        .coords()
        .iter()
        .zip(l_step.coords().iter())
        .enumerate()
    {
        assert_eq!(a.bits(), b.bits(), "{label}: bit mismatch at coord {idx}");
    }
}

fn assert_nd_2d_matches_layout2d<S: ScalarBits>(
    mut s_2d: SgdSolver<Layout2d<S>>,
    mut s_nd: SgdSolver<LayoutNd<S>>,
    init_2d: &Layout2d<S>,
    label: &str,
) {
    let mut l_2d = init_2d.clone();
    let mut l_nd = layout_2d_to_nd(init_2d);
    s_2d.run(&mut l_2d).unwrap();
    s_nd.run(&mut l_nd).unwrap();
    for i in 0..l_2d.len() {
        assert!(
            (l_2d.x()[i].to_f64() - l_nd.coords()[i * 2].to_f64()).abs() < 1e-12,
            "{label}: x at {i}"
        );
        assert!(
            (l_2d.y()[i].to_f64() - l_nd.coords()[i * 2 + 1].to_f64()).abs() < 1e-12,
            "{label}: y at {i}"
        );
    }
}

pub(super) fn build_mode_pair<L: RiemannianLayout>(
    mode: usize,
    g: &CsrGraph<f64>,
    landmarks: &DogVaradhanLandmarks,
    emb_flat: &[f64],
) -> (SgdSolver<L>, SgdSolver<L>, &'static str) {
    let n = g.num_nodes();
    let sched = SgdScheduleConfig::new(16)
        .with_eps(0.01)
        .with_kind(SgdScheduleKind::Exponential)
        .with_step_scale(0.9);
    let dog_cfg = DogVaradhanConfig::new().with_n_landmarks(6).with_seed(42);
    let sb_cfg = SparseBarrierConfig::default()
        .with_num_pivots(6)
        .with_schedule(sched.clone())
        .with_seed(42);
    let sf = |v: f64| L::Scalar::from_f64(v);
    match mode {
        0 => (
            SgdSolver::full(g, sched.clone(), 42).unwrap(),
            SgdSolver::full(g, sched, 42).unwrap(),
            "full",
        ),
        1 => {
            let mk = || {
                SgdSolver::from_pairs(
                    n,
                    &[0, 1, 2],
                    &[1, 2, 3],
                    &[sf(1.0), sf(1.5), sf(2.0)],
                    &[sf(1.2), sf(1.5), sf(2.5)],
                    &[sf(1.0), sf(0.8), sf(0.5)],
                    &[sf(0.7), sf(0.8), sf(0.4)],
                    sched.clone(),
                    42,
                )
                .unwrap()
            };
            (mk(), mk(), "from_pairs")
        }
        2 => {
            let mk = || {
                SgdSolver::from_pairs_with_hinges(
                    n,
                    &[0, 1],
                    &[1, 2],
                    &[sf(1.0), sf(1.0)],
                    &[sf(1.0), sf(1.0)],
                    &[sf(1.0), sf(1.0)],
                    &[sf(1.0), sf(1.0)],
                    &[0, 2],
                    &[3, 4],
                    &[sf(2.5), sf(2.5)],
                    &[sf(2.5), sf(2.5)],
                    &[sf(0.5), sf(0.5)],
                    &[sf(0.5), sf(0.5)],
                    sched.clone(),
                    42,
                )
                .unwrap()
            };
            (mk(), mk(), "from_pairs_with_hinges")
        }
        3 => {
            let mk = || {
                let mut buf = SgdPairBuffer::with_capacity(2);
                buf.push_spring(0, 1, sf(1.0), sf(1.0)).unwrap();
                buf.push_hinge(1, 2, sf(2.0), sf(0.5)).unwrap();
                SgdSolver::from_pair_buffer(n, buf, sched.clone(), 42).unwrap()
            };
            (mk(), mk(), "from_pair_buffer")
        }
        4 => (
            SgdSolver::random_pair(g, 4, sched.clone(), 42).unwrap(),
            SgdSolver::random_pair(g, 4, sched, 42).unwrap(),
            "random_pair",
        ),
        5 => (
            SgdSolver::omega(g, emb_flat, 2, 4, 1e-3, sched.clone(), 42).unwrap(),
            SgdSolver::omega(g, emb_flat, 2, 4, 1e-3, sched, 42).unwrap(),
            "omega",
        ),
        6 => (
            SgdSolver::sparse_barrier_from_landmarks(g, landmarks, None, &sb_cfg)
                .unwrap()
                .0,
            SgdSolver::sparse_barrier_from_landmarks(g, landmarks, None, &sb_cfg)
                .unwrap()
                .0,
            "sparse_barrier_from_landmarks",
        ),
        7 => (
            SgdSolver::sparse_barrier_dog(g, &dog_cfg, &sb_cfg)
                .unwrap()
                .0,
            SgdSolver::sparse_barrier_dog(g, &dog_cfg, &sb_cfg)
                .unwrap()
                .0,
            "sparse_barrier_dog",
        ),
        8 => {
            let cfg = StreamingNegativeConfig::default();
            (
                SgdSolver::streaming_negatives(g, cfg.clone(), sched.clone(), 42).unwrap(),
                SgdSolver::streaming_negatives(g, cfg, sched, 42).unwrap(),
                "streaming_negatives",
            )
        }
        9 => {
            let cfg = OrtmannSparseStressConfig::default()
                .with_num_pivots(4)
                .with_iterations(16)
                .with_seed(42);
            (
                SgdSolver::ortmann_sparse_stress(g, cfg.clone()).unwrap(),
                SgdSolver::ortmann_sparse_stress(g, cfg).unwrap(),
                "ortmann_sparse_stress",
            )
        }
        10 => (
            SgdSolver::from_pair_buffer(n, SgdPairBuffer::new(), sched.clone(), 42).unwrap(),
            SgdSolver::from_pair_buffer(n, SgdPairBuffer::new(), sched, 42).unwrap(),
            "empty",
        ),
        _ => unreachable!(),
    }
}

#[test]
fn test_all_sgd_and_sgd2_modes_nd_step_eq_run_f64_and_f32_3d_5d() {
    let g = grid_graph(3, 4);
    let n = g.num_nodes();
    let dog_cfg = DogVaradhanConfig::new().with_n_landmarks(6).with_seed(42);
    let (landmarks, _) = compute_regularized_dog_varadhan_landmarks(&g, &dog_cfg).unwrap();
    let spec: Layout2d<f64> = KorenSpectral::new(&g).generate().unwrap();
    let emb_flat: Vec<f64> = (0..n).flat_map(|i| [spec.x()[i], spec.y()[i]]).collect();

    for dims in [3usize, 5usize] {
        let init64 = LayoutNd::<f64>::init(n, dims, 99, 1.0).unwrap();
        let init32 = LayoutNd::<f32>::init(n, dims, 99, 1.0).unwrap();
        for mode in 0..11 {
            let (s1, s2, name) = build_mode_pair::<LayoutNd<f64>>(mode, &g, &landmarks, &emb_flat);
            assert_step_eq_run_nd(s1, s2, &init64, 5, &format!("{name}_f64_{dims}d"));
            let (s1, s2, name) = build_mode_pair::<LayoutNd<f32>>(mode, &g, &landmarks, &emb_flat);
            assert_step_eq_run_nd(s1, s2, &init32, 5, &format!("{name}_f32_{dims}d"));
        }

        let s2_cfg = Sgd2Config::new()
            .with_iterations(16)
            .with_batch_size(5)
            .with_seed(42);
        let mut s2_run = Sgd2Solver::<LayoutNd<f64>>::new(&g, s2_cfg.clone()).unwrap();
        let mut s2_step = Sgd2Solver::<LayoutNd<f64>>::new(&g, s2_cfg.clone()).unwrap();
        let mut l2_run = init64.clone();
        let mut l2_step = init64.clone();
        s2_run.run(&mut l2_run).unwrap();
        s2_step.run_steps(&mut l2_step, 5).unwrap();
        s2_step.run(&mut l2_step).unwrap();
        assert_eq!(l2_run, l2_step);

        let mut s2_32_run = Sgd2Solver::<LayoutNd<f32>>::new(&g, s2_cfg.clone()).unwrap();
        let mut s2_32_step = Sgd2Solver::<LayoutNd<f32>>::new(&g, s2_cfg).unwrap();
        let mut l2_32_run = init32.clone();
        let mut l2_32_step = init32.clone();
        s2_32_run.run(&mut l2_32_run).unwrap();
        s2_32_step.run_steps(&mut l2_32_step, 5).unwrap();
        s2_32_step.run(&mut l2_32_step).unwrap();
        assert_eq!(l2_32_run, l2_32_step);
    }

    // Layout3d convergence + step == run parity.
    let mut full_3d =
        SgdSolver::<Layout3d<f64>>::full(&g, SgdScheduleConfig::default(), 42).unwrap();
    let mut full_3d_step =
        SgdSolver::<Layout3d<f64>>::full(&g, SgdScheduleConfig::default(), 42).unwrap();
    let mut l_3d = Layout3d::<f64>::init(n, 12, 0.5).unwrap();
    let mut l_3d_step = l_3d.clone();
    full_3d.run(&mut l_3d).unwrap();
    full_3d_step.run_steps(&mut l_3d_step, 10).unwrap();
    full_3d_step.run(&mut l_3d_step).unwrap();
    assert_eq!(l_3d, l_3d_step);

    let mut sum_err = 0.0_f64;
    for u in 0..n {
        for &v_u32 in g.neighbors(u) {
            let v = v_u32 as usize;
            if u < v {
                let d2 = (l_3d.x()[u] - l_3d.x()[v]).powi(2)
                    + (l_3d.y()[u] - l_3d.y()[v]).powi(2)
                    + (l_3d.z()[u] - l_3d.z()[v]).powi(2);
                sum_err += (d2.sqrt() - 1.0).abs();
            }
        }
    }
    assert!(sum_err / (g.num_edges() as f64) < 0.15);
}

#[test]
fn test_nd_dims2_agrees_with_layout2d_all_modes() {
    let g = barbell_graph(5);
    let n = g.num_nodes();
    let dog_cfg = DogVaradhanConfig::new().with_n_landmarks(5).with_seed(42);
    let (landmarks, _) = compute_regularized_dog_varadhan_landmarks(&g, &dog_cfg).unwrap();
    let spec: Layout2d<f64> = KorenSpectral::new(&g).generate().unwrap();
    let emb_flat: Vec<f64> = (0..n).flat_map(|i| [spec.x()[i], spec.y()[i]]).collect();
    let init_nd = LayoutNd::<f64>::init(n, 2, 17, 1.0).unwrap();
    let init_2d = Layout2d::from_xy(
        (0..n).map(|i| init_nd.coords()[i * 2]).collect(),
        (0..n).map(|i| init_nd.coords()[i * 2 + 1]).collect(),
    )
    .unwrap();

    for mode in 0..11 {
        let (s2d, snd, name) = (
            build_mode_pair::<Layout2d<f64>>(mode, &g, &landmarks, &emb_flat).0,
            build_mode_pair::<LayoutNd<f64>>(mode, &g, &landmarks, &emb_flat).0,
            build_mode_pair::<Layout2d<f64>>(mode, &g, &landmarks, &emb_flat).2,
        );
        assert_nd_2d_matches_layout2d(s2d, snd, &init_2d, name);
    }

    let s2_cfg = Sgd2Config::new().with_iterations(12).with_seed(42);
    let mut s2_2d = Sgd2Solver::<Layout2d<f64>>::new(&g, s2_cfg.clone()).unwrap();
    let mut s2_nd = Sgd2Solver::<LayoutNd<f64>>::new(&g, s2_cfg).unwrap();
    let mut l2 = init_2d;
    let mut lnd = init_nd;
    s2_2d.run(&mut l2).unwrap();
    s2_nd.run(&mut lnd).unwrap();
    for i in 0..n {
        assert!((l2.x()[i] - lnd.coords()[i * 2]).abs() < 1e-12);
        assert!((l2.y()[i] - lnd.coords()[i * 2 + 1]).abs() < 1e-12);
    }
}
