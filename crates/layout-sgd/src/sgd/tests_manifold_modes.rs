use egraph_core::{
    CsrGraph, IterativeSolver, ManifoldKind, ManifoldLayout2d, Scalar, SolverStatus,
};
use egraph_linalg_mds::koren_spectral_init;
use egraph_neighborhood::{DogVaradhanConfig, compute_regularized_dog_varadhan_landmarks};

use crate::coarse_backbone::CoarseBackboneConfig;

use super::super::sgd::{
    DrGraphConfig, GraphCneConfig, LargeVisConfig, OrtmannSparseStressConfig, PacmapGraphConfig,
    Sgd2Config, SgdScheduleConfig, SgdSolver, SnapTFdpConfig, SparseBarrierConfig,
    StreamingNegativeConfig,
};
use super::pairs::PackedSymPair;
use super::test_support::grid_graph;
use super::tests_nd::build_nd_mode_pair;

const ALL_MANIFOLDS: [ManifoldKind; 4] = [
    ManifoldKind::HyperbolicPoincare,
    ManifoldKind::HyperbolicLorentz,
    ManifoldKind::Spherical,
    ManifoldKind::Torus2d,
];

fn assert_manifold_step_eq_run<S: Scalar>(
    mut s_run: SgdSolver<S>,
    mut s_step: SgdSolver<S>,
    kind: ManifoldKind,
    n: usize,
    label: &str,
) {
    assert!(s_run.supports_manifold(), "{label}: supports_manifold");
    let mut l_run = ManifoldLayout2d::<S>::init(kind, n, 42, 0.35).unwrap();
    let mut l_step = l_run.clone();

    let st_run = IterativeSolver::<S, ManifoldLayout2d<S>>::run(&mut s_run, &mut l_run).unwrap();
    assert_eq!(st_run, SolverStatus::Completed, "{label}");

    let first_k = (s_step.total_iterations() / 3).max(1);
    IterativeSolver::<S, ManifoldLayout2d<S>>::run_steps(&mut s_step, &mut l_step, first_k)
        .unwrap();
    if !s_step.status().is_finished() {
        IterativeSolver::<S, ManifoldLayout2d<S>>::step(&mut s_step, &mut l_step).unwrap();
    }
    let st_step = s_step.run_manifold(&mut l_step).unwrap();
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

fn build_extra_pacmap_constructors<S: Scalar>(
    g: &CsrGraph<f64>,
    landmarks: &egraph_neighborhood::DogVaradhanLandmarks,
) -> Vec<(SgdSolver<S>, SgdSolver<S>, &'static str)> {
    let n = g.num_nodes();
    let pac_cfg = PacmapGraphConfig {
        phase_iterations: [4, 4, 4],
        num_landmarks: 6,
        seed: 42,
        ..Default::default()
    };
    let lm_flat = landmarks.d_dog();
    let k_lm = landmarks.num_landmarks();
    assert_eq!(lm_flat.len(), n * k_lm);

    let (bb_a, _, _) =
        SgdSolver::<S>::pacmap_graph_with_backbone(g, pac_cfg, &CoarseBackboneConfig::default())
            .unwrap();
    let (bb_b, _, _) =
        SgdSolver::<S>::pacmap_graph_with_backbone(g, pac_cfg, &CoarseBackboneConfig::default())
            .unwrap();

    vec![
        (
            SgdSolver::pacmap_graph_with_landmarks(g, landmarks, pac_cfg).unwrap(),
            SgdSolver::pacmap_graph_with_landmarks(g, landmarks, pac_cfg).unwrap(),
            "pacmap_graph_with_landmarks",
        ),
        (
            SgdSolver::pacmap_graph_with_landmark_distances(g, lm_flat, k_lm, pac_cfg).unwrap(),
            SgdSolver::pacmap_graph_with_landmark_distances(g, lm_flat, k_lm, pac_cfg).unwrap(),
            "pacmap_graph_with_landmark_distances",
        ),
        (bb_a, bb_b, "pacmap_graph_with_backbone"),
    ]
}

#[test]
fn test_all_19_constructors_manifold_bit_exact_f64_and_f32() {
    let g = grid_graph(3, 4);
    let n = g.num_nodes();
    let dog_cfg = DogVaradhanConfig {
        n_landmarks: 6,
        seed: 42,
        ..Default::default()
    };
    let (landmarks, _) = compute_regularized_dog_varadhan_landmarks(&g, &dog_cfg).unwrap();
    let spec = koren_spectral_init::<f64, f64>(&g, Some(1.0), 1e-6, 200, 42).unwrap();
    let mut emb_flat = Vec::with_capacity(n * 2);
    for i in 0..n {
        emb_flat.push(spec.x()[i]);
        emb_flat.push(spec.y()[i]);
    }

    for kind in ALL_MANIFOLDS {
        for mode in 0..17 {
            let (s_run, s_step, label) = build_nd_mode_pair::<f64>(mode, &g, &landmarks, &emb_flat);
            assert_manifold_step_eq_run::<f64>(s_run, s_step, kind, n, label);

            let (s32_run, s32_step, label32) =
                build_nd_mode_pair::<f32>(mode, &g, &landmarks, &emb_flat);
            assert_manifold_step_eq_run::<f32>(s32_run, s32_step, kind, n, label32);
        }

        for (s_run, s_step, label) in build_extra_pacmap_constructors::<f64>(&g, &landmarks) {
            assert_manifold_step_eq_run::<f64>(s_run, s_step, kind, n, label);
        }
        for (s32_run, s32_step, label) in build_extra_pacmap_constructors::<f32>(&g, &landmarks) {
            assert_manifold_step_eq_run::<f32>(s32_run, s32_step, kind, n, label);
        }
    }
}

#[test]
fn test_all_modes_reset_schedule_and_windowed_manifold_determinism() {
    let g = grid_graph(3, 4);
    let n = g.num_nodes();
    let dog_cfg = DogVaradhanConfig {
        n_landmarks: 4,
        seed: 7,
        ..Default::default()
    };
    let (landmarks, _) = compute_regularized_dog_varadhan_landmarks(&g, &dog_cfg).unwrap();
    let spec = koren_spectral_init::<f64, f64>(&g, Some(1.0), 1e-6, 200, 7).unwrap();
    let mut emb = Vec::with_capacity(n * 2);
    for i in 0..n {
        emb.push(spec.x()[i]);
        emb.push(spec.y()[i]);
    }

    let s_init = SgdScheduleConfig::new(8);
    let s_reset = SgdScheduleConfig::new(10)
        .with_step_scale(0.6)
        .with_eps(0.05);
    let sb_cfg = SparseBarrierConfig {
        num_pivots: 4,
        schedule: s_init,
        seed: 7,
        ..Default::default()
    };

    let make_all = || -> Vec<(SgdSolver<f64>, &'static str)> {
        vec![
            (SgdSolver::full(&g, s_init, 7).unwrap(), "full"),
            (
                SgdSolver::from_pairs(
                    n,
                    vec![0, 1],
                    vec![1, 2],
                    vec![1.0, 1.0],
                    vec![1.0, 1.0],
                    vec![1.0, 1.0],
                    vec![1.0, 1.0],
                    s_init,
                    7,
                )
                .unwrap(),
                "from_pairs",
            ),
            (
                SgdSolver::from_packed_symmetric(
                    n,
                    vec![PackedSymPair::new(0, 1, false, 1.0, 1.0)],
                    1,
                    0,
                    s_init,
                    7,
                )
                .unwrap(),
                "from_packed_symmetric",
            ),
            (
                SgdSolver::random_pair(&g, 3, s_init, 7).unwrap(),
                "random_pair",
            ),
            (
                SgdSolver::omega(&g, &emb, 2, 3, 1e-3, s_init, 7).unwrap(),
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
                SgdSolver::streaming_negatives(&g, StreamingNegativeConfig::default(), s_init, 7)
                    .unwrap(),
                "streaming_negatives",
            ),
            (
                SgdSolver::ortmann_sparse_stress(
                    &g,
                    OrtmannSparseStressConfig {
                        num_pivots: 4,
                        iterations: 8,
                        seed: 7,
                        ..Default::default()
                    },
                )
                .unwrap(),
                "ortmann_sparse_stress",
            ),
            (
                SgdSolver::snap_t_fdp(
                    &g,
                    SnapTFdpConfig {
                        iterations: 8,
                        seed: 7,
                        ..Default::default()
                    },
                )
                .unwrap(),
                "snap_t_fdp",
            ),
            (
                SgdSolver::pacmap_graph(
                    &g,
                    PacmapGraphConfig {
                        phase_iterations: [3, 3, 2],
                        num_landmarks: 4,
                        seed: 7,
                        ..Default::default()
                    },
                )
                .unwrap(),
                "pacmap_graph",
            ),
            (
                SgdSolver::sgd2(
                    &g,
                    Sgd2Config {
                        iterations: 8,
                        batch_size: 4,
                        seed: 7,
                        ..Default::default()
                    },
                )
                .unwrap(),
                "sgd2",
            ),
            (
                SgdSolver::largevis(
                    &g,
                    LargeVisConfig {
                        iterations: 8,
                        negative_samples: 2,
                        seed: 7,
                        ..Default::default()
                    },
                )
                .unwrap(),
                "largevis",
            ),
            (
                SgdSolver::drgraph(
                    &g,
                    DrGraphConfig {
                        iterations: 8,
                        negative_samples: 2,
                        seed: 7,
                        ..Default::default()
                    },
                )
                .unwrap(),
                "drgraph",
            ),
            (
                SgdSolver::graph_cne(
                    &g,
                    GraphCneConfig {
                        iterations: 8,
                        negative_samples: 2,
                        seed: 7,
                        ..Default::default()
                    },
                )
                .unwrap(),
                "graph_cne",
            ),
        ]
    };

    let mut set_a = make_all();
    let mut set_b = make_all();
    for ((sa, label), (sb, _)) in set_a.iter_mut().zip(set_b.iter_mut()) {
        let mut ma = ManifoldLayout2d::<f64>::init(ManifoldKind::Torus2d, n, 11, 1.0).unwrap();
        let mut mb = ma.clone();

        sa.run_schedule_window(&mut ma, 0, 3).unwrap();
        sb.run_schedule_window_manifold(&mut mb, 0, 3).unwrap();
        assert_eq!(ma, mb, "{label}: pre-reset window");

        sa.reset_schedule(s_reset, 55).unwrap();
        sb.reset_schedule(s_reset, 55).unwrap();
        assert_eq!(sa.total_iterations(), 10, "{label}");
        assert_eq!(sa.iteration(), 0, "{label}");

        sa.run_schedule_window(&mut ma, 0, 4).unwrap();
        sa.run_schedule_window(&mut ma, 4, 6).unwrap();
        sb.run_manifold(&mut mb).unwrap();
        assert_eq!(ma, mb, "{label}: post-reset windowed == full run");
    }
}
