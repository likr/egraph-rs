use egraph_core::{CsrGraph, IterativeSolver, Layout2d, LayoutNd, Scalar, SolverStatus};
use egraph_linalg_mds::koren_spectral_init;
use egraph_neighborhood::{
    DogVaradhanConfig, DogVaradhanLandmarks, compute_regularized_dog_varadhan_landmarks,
};

use super::super::coarse_backbone::CoarseBackboneConfig;
use super::super::sgd::{
    DrGraphConfig, GraphCneConfig, GraphCneObjective, LargeVisConfig, OrtmannSparseStressConfig,
    PackedSymPair, PacmapGraphConfig, Sgd2Config, SgdScheduleConfig, SgdScheduleKind, SgdSolver,
    SnapTFdpConfig, SparseBarrierConfig, StreamingNegativeConfig,
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
    LayoutNd::from_flat(n, 2, flat).unwrap()
}

pub(super) fn assert_step_eq_run_nd<S: ScalarBits>(
    mut s_run: SgdSolver<S>,
    mut s_step: SgdSolver<S>,
    init: &LayoutNd<S>,
    step_chunk: usize,
    label: &str,
) {
    let mut l_run = init.copy();
    let mut l_step = init.copy();
    assert_eq!(s_run.status(), SolverStatus::Initialized, "{label}");
    assert_eq!(
        s_run.run(&mut l_run).unwrap(),
        SolverStatus::Completed,
        "{label}"
    );

    let total = s_step.total_iterations();
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
    mut s_2d: SgdSolver<S>,
    mut s_nd: SgdSolver<S>,
    init_2d: &Layout2d<S>,
    label: &str,
) {
    let mut l_2d = init_2d.clone();
    let mut l_nd = layout_2d_to_nd(init_2d);
    s_2d.run(&mut l_2d).unwrap();
    s_nd.run(&mut l_nd).unwrap();
    for i in 0..l_2d.len() {
        assert_eq!(
            l_2d.x()[i].bits(),
            l_nd.coords()[i * 2].bits(),
            "{label}: x at {i}"
        );
        assert_eq!(
            l_2d.y()[i].bits(),
            l_nd.coords()[i * 2 + 1].bits(),
            "{label}: y at {i}"
        );
    }
}

pub(super) fn build_nd_mode_pair<S: Scalar>(
    mode: usize,
    g: &CsrGraph<f64>,
    landmarks: &DogVaradhanLandmarks,
    emb_flat: &[f64],
) -> (SgdSolver<S>, SgdSolver<S>, &'static str) {
    let n = g.num_nodes();
    let sched = SgdScheduleConfig {
        iterations: 16,
        eps: 0.01,
        kind: SgdScheduleKind::Exponential,
        step_scale: 0.9,
    };
    let dog_cfg = DogVaradhanConfig {
        n_landmarks: 6,
        seed: 42,
        ..Default::default()
    };
    let sb_cfg = SparseBarrierConfig {
        num_pivots: 6,
        schedule: sched,
        seed: 42,
        ..Default::default()
    };
    let sf = |v: f64| S::from_f64(v);
    match mode {
        0 => (
            SgdSolver::full(g, sched, 42).unwrap(),
            SgdSolver::full(g, sched, 42).unwrap(),
            "full",
        ),
        1 => {
            let mk = || {
                SgdSolver::from_pairs(
                    n,
                    vec![0, 1, 2],
                    vec![1, 2, 3],
                    vec![sf(1.0), sf(1.5), sf(2.0)],
                    vec![sf(1.2), sf(1.5), sf(2.5)],
                    vec![sf(1.0), sf(0.8), sf(0.5)],
                    vec![sf(0.7), sf(0.8), sf(0.4)],
                    sched,
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
                    vec![0, 1],
                    vec![1, 2],
                    vec![sf(1.0), sf(1.0)],
                    vec![sf(1.0), sf(1.0)],
                    vec![sf(1.0), sf(1.0)],
                    vec![sf(1.0), sf(1.0)],
                    vec![0, 2],
                    vec![3, 4],
                    vec![sf(2.5), sf(2.5)],
                    vec![sf(2.5), sf(2.5)],
                    vec![sf(0.5), sf(0.5)],
                    vec![sf(0.5), sf(0.5)],
                    sched,
                    42,
                )
                .unwrap()
            };
            (mk(), mk(), "from_pairs_with_hinges")
        }
        3 => {
            let mk = || {
                let pairs = vec![
                    PackedSymPair::new(0, 1, false, sf(1.0), sf(1.0)),
                    PackedSymPair::new(1, 2, true, sf(2.0), sf(0.5)),
                ];
                SgdSolver::from_packed_symmetric(n, pairs, 1, 1, sched, 42).unwrap()
            };
            (mk(), mk(), "from_packed_symmetric")
        }
        4 => (
            SgdSolver::random_pair(g, 4, sched, 42).unwrap(),
            SgdSolver::random_pair(g, 4, sched, 42).unwrap(),
            "random_pair",
        ),
        5 => (
            SgdSolver::omega(g, emb_flat, 2, 4, 1e-3, sched, 42).unwrap(),
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
                SgdSolver::streaming_negatives(g, cfg, sched, 42).unwrap(),
                SgdSolver::streaming_negatives(g, cfg, sched, 42).unwrap(),
                "streaming_negatives",
            )
        }
        9 => {
            let cfg = OrtmannSparseStressConfig {
                num_pivots: 4,
                iterations: 16,
                seed: 42,
                ..Default::default()
            };
            (
                SgdSolver::ortmann_sparse_stress(g, cfg).unwrap(),
                SgdSolver::ortmann_sparse_stress(g, cfg).unwrap(),
                "ortmann_sparse_stress",
            )
        }
        10 => (
            SgdSolver::from_packed_symmetric(n, Vec::new(), 0, 0, sched, 42).unwrap(),
            SgdSolver::from_packed_symmetric(n, Vec::new(), 0, 0, sched, 42).unwrap(),
            "empty",
        ),
        11 => {
            let cfg = SnapTFdpConfig {
                iterations: 16,
                seed: 42,
                ..Default::default()
            };
            (
                SgdSolver::snap_t_fdp(g, cfg).unwrap(),
                SgdSolver::snap_t_fdp(g, cfg).unwrap(),
                "snap_t_fdp",
            )
        }
        12 => {
            let cfg = PacmapGraphConfig {
                phase_iterations: [5, 5, 6],
                num_landmarks: 6,
                seed: 42,
                ..Default::default()
            };
            (
                SgdSolver::pacmap_graph(g, cfg).unwrap(),
                SgdSolver::pacmap_graph(g, cfg).unwrap(),
                "pacmap_graph",
            )
        }
        13 => {
            let cfg = Sgd2Config {
                iterations: 16,
                batch_size: 5,
                seed: 42,
                ..Default::default()
            };
            (
                SgdSolver::sgd2(g, cfg).unwrap(),
                SgdSolver::sgd2(g, cfg).unwrap(),
                "sgd2",
            )
        }
        14 => {
            let cfg = LargeVisConfig {
                iterations: 16,
                negative_samples: 3,
                seed: 42,
                ..Default::default()
            };
            (
                SgdSolver::largevis(g, cfg).unwrap(),
                SgdSolver::largevis(g, cfg).unwrap(),
                "largevis",
            )
        }
        15 => {
            let cfg = DrGraphConfig {
                k_order: 2,
                iterations: 18,
                negative_samples: 3,
                coarsen_stop_ratio: 0.85,
                seed: 42,
                ..Default::default()
            };
            (
                SgdSolver::drgraph(g, cfg).unwrap(),
                SgdSolver::drgraph(g, cfg).unwrap(),
                "drgraph",
            )
        }
        16 => {
            let cfg = GraphCneConfig {
                objective: GraphCneObjective::InfoNce,
                iterations: 16,
                negative_samples: 3,
                seed: 42,
                ..Default::default()
            };
            (
                SgdSolver::graph_cne(g, cfg).unwrap(),
                SgdSolver::graph_cne(g, cfg).unwrap(),
                "graph_cne",
            )
        }
        _ => unreachable!(),
    }
}

#[test]
fn test_all_17_sgd_modes_nd_step_eq_run_f64_and_f32_3d_5d() {
    let g = grid_graph(3, 4);
    let n = g.num_nodes();
    let dog_cfg = DogVaradhanConfig {
        n_landmarks: 6,
        seed: 42,
        ..Default::default()
    };
    let (landmarks, _) = compute_regularized_dog_varadhan_landmarks(&g, &dog_cfg).unwrap();
    let spec = koren_spectral_init::<f64, f64>(&g, Some(1.0), 1e-6, 150, 42).unwrap();
    let emb_flat: Vec<f64> = (0..n).flat_map(|i| [spec.x()[i], spec.y()[i]]).collect();

    for dims in [3usize, 5usize] {
        let init64 = LayoutNd::<f64>::init(n, dims, 99, 1.0).unwrap();
        let init32 = LayoutNd::<f32>::init(n, dims, 99, 1.0).unwrap();
        for mode in 0..17 {
            let (s1, s2, name) = build_nd_mode_pair::<f64>(mode, &g, &landmarks, &emb_flat);
            assert_step_eq_run_nd(s1, s2, &init64, 5, &format!("{name}_f64_{dims}d"));
            let (s1, s2, name) = build_nd_mode_pair::<f32>(mode, &g, &landmarks, &emb_flat);
            assert_step_eq_run_nd(s1, s2, &init32, 5, &format!("{name}_f32_{dims}d"));
        }
    }

    let pm_cfg = PacmapGraphConfig {
        phase_iterations: [4, 4, 4],
        num_landmarks: 6,
        seed: 42,
        ..Default::default()
    };
    let lm_dist_flat: Vec<f64> = landmarks.d_dog().to_vec();
    let k_lm = landmarks.num_landmarks();
    let bb_cfg = CoarseBackboneConfig::default();
    let neg_tsne_cfg = GraphCneConfig {
        objective: GraphCneObjective::NegTSne,
        iterations: 14,
        row_normalize: false,
        seed: 42,
        ..Default::default()
    };
    for dims in [3usize, 5usize] {
        let init64 = LayoutNd::<f64>::init(n, dims, 7, 1.0).unwrap();
        let init32 = LayoutNd::<f32>::init(n, dims, 7, 1.0).unwrap();
        assert_step_eq_run_nd(
            SgdSolver::<f64>::pacmap_graph_with_landmarks(&g, &landmarks, pm_cfg).unwrap(),
            SgdSolver::<f64>::pacmap_graph_with_landmarks(&g, &landmarks, pm_cfg).unwrap(),
            &init64,
            4,
            "pacmap_with_landmarks",
        );
        assert_step_eq_run_nd(
            SgdSolver::<f64>::pacmap_graph_with_landmark_distances(&g, &lm_dist_flat, k_lm, pm_cfg)
                .unwrap(),
            SgdSolver::<f64>::pacmap_graph_with_landmark_distances(&g, &lm_dist_flat, k_lm, pm_cfg)
                .unwrap(),
            &init64,
            4,
            "pacmap_with_landmark_distances",
        );
        let (bb_s1, _, _) =
            SgdSolver::<f64>::pacmap_graph_with_backbone(&g, pm_cfg, &bb_cfg).unwrap();
        let (bb_s2, _, _) =
            SgdSolver::<f64>::pacmap_graph_with_backbone(&g, pm_cfg, &bb_cfg).unwrap();
        assert_step_eq_run_nd(bb_s1, bb_s2, &init64, 4, "pacmap_with_backbone");
        assert_step_eq_run_nd(
            SgdSolver::<f32>::graph_cne(&g, neg_tsne_cfg).unwrap(),
            SgdSolver::<f32>::graph_cne(&g, neg_tsne_cfg).unwrap(),
            &init32,
            4,
            "graph_cne_neg_tsne",
        );
    }

    let mut full_3d = SgdSolver::<f64>::full(&g, SgdScheduleConfig::default(), 42).unwrap();
    let mut l_3d = LayoutNd::<f64>::init(n, 3, 12, 0.5).unwrap();
    full_3d.run(&mut l_3d).unwrap();
    let mut sum_err = 0.0_f64;
    for u in 0..n {
        for &v_u32 in g.neighbors(u) {
            let v = v_u32 as usize;
            if u < v {
                let d2: f64 = (0..3)
                    .map(|a| (l_3d.coords()[u * 3 + a] - l_3d.coords()[v * 3 + a]).powi(2))
                    .sum();
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
    let dog_cfg = DogVaradhanConfig {
        n_landmarks: 5,
        seed: 42,
        ..Default::default()
    };
    let (landmarks, _) = compute_regularized_dog_varadhan_landmarks(&g, &dog_cfg).unwrap();
    let spec = koren_spectral_init::<f64, f64>(&g, Some(1.0), 1e-6, 150, 42).unwrap();
    let emb_flat: Vec<f64> = (0..n).flat_map(|i| [spec.x()[i], spec.y()[i]]).collect();
    let init_nd = LayoutNd::<f64>::init(n, 2, 17, 1.0).unwrap();
    let init_2d = Layout2d::from_xy(
        (0..n).map(|i| init_nd.coords()[i * 2]).collect(),
        (0..n).map(|i| init_nd.coords()[i * 2 + 1]).collect(),
    )
    .unwrap();

    for mode in 0..17 {
        if mode == 15 {
            continue;
        }
        let (s2d, snd, name) = build_nd_mode_pair::<f64>(mode, &g, &landmarks, &emb_flat);
        assert_nd_2d_matches_layout2d(s2d, snd, &init_2d, name);
    }

    let dr_single = DrGraphConfig {
        k_order: 1,
        iterations: 10,
        coarsen_stop_ratio: 0.01,
        seed: 42,
        ..Default::default()
    };
    assert_nd_2d_matches_layout2d(
        SgdSolver::<f32>::drgraph(&g, dr_single).unwrap(),
        SgdSolver::<f32>::drgraph(&g, dr_single).unwrap(),
        &Layout2d::<f32>::init(n, 17, 1.0).unwrap(),
        "drgraph_single_level_f32",
    );
}
