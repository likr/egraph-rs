use egraph_core::{
    CoreError, CsrGraph, IterativeSolver, Layout2d, LayoutNd, ManifoldKind, ManifoldLayout2d,
    Scalar, SolverStatus,
};
use egraph_neighborhood::SparseProbabilities;

use super::super::{TsNetBackend, TsNetConfig, TsNetSolver};
use super::{make_barbell_graph, make_grid_graph};

fn init_manifold<S: Scalar>(kind: ManifoldKind, n: usize, seed: u64) -> ManifoldLayout2d<S> {
    let init_scale = match kind {
        ManifoldKind::HyperbolicPoincare => 0.4,
        ManifoldKind::HyperbolicLorentz => 0.7,
        ManifoldKind::Spherical | ManifoldKind::Torus2d => 1.0,
        _ => 0.5,
    };
    ManifoldLayout2d::<S>::init(kind, n, seed, init_scale).unwrap()
}

fn mean_geodesic_edge_and_all<S: Scalar>(
    graph: &CsrGraph<f64>,
    layout: &ManifoldLayout2d<S>,
) -> (f64, f64) {
    let n = layout.len();
    let mut edge_sum = 0.0;
    let mut edge_cnt = 0usize;
    for u in 0..n {
        for &v in graph.neighbors(u) {
            let vi = v as usize;
            if u < vi {
                edge_sum += layout.geodesic_distance(u, vi).unwrap().to_f64();
                edge_cnt += 1;
            }
        }
    }
    let mut all_sum = 0.0;
    let mut all_cnt = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            all_sum += layout.geodesic_distance(i, j).unwrap().to_f64();
            all_cnt += 1;
        }
    }
    (
        edge_sum / (edge_cnt.max(1) as f64),
        all_sum / (all_cnt.max(1) as f64),
    )
}

#[test]
fn test_tsnet_manifold_all_kinds_bit_exact_and_constructors() {
    let grid = make_grid_graph(4, 4);
    let n = grid.num_nodes();

    for kind in [
        ManifoldKind::HyperbolicPoincare,
        ManifoldKind::HyperbolicLorentz,
        ManifoldKind::Spherical,
        ManifoldKind::Torus2d,
    ] {
        for backend in [TsNetBackend::Exact, TsNetBackend::Auto] {
            let cfg = TsNetConfig {
                backend,
                stage_iterations: [8, 8, 8],
                lambda_r_stage3: 0.5,
                ..TsNetConfig::pure_2hop()
            };

            // f64 bit-exact step == run.
            let mut l_run = init_manifold::<f64>(kind, n, 17);
            if kind == ManifoldKind::Torus2d {
                l_run.set_torus_extents(2.5, 1.5).unwrap();
            } else {
                l_run.set_scale(1.5).unwrap();
            }
            let mut l_step = l_run.clone();
            let mut s_run = TsNetSolver::<f64>::from_graph_pure_2hop(&grid, cfg).unwrap();
            let mut s_step = TsNetSolver::<f64>::from_graph_pure_2hop(&grid, cfg).unwrap();

            assert_eq!(s_run.run(&mut l_run).unwrap(), SolverStatus::Completed);
            assert!(l_run.validate_finite().is_ok());

            assert_eq!(
                s_step.run_steps(&mut l_step, 5).unwrap(),
                SolverStatus::Running
            );
            while !s_step.status().is_finished() {
                s_step.step(&mut l_step).unwrap();
            }
            assert_eq!(s_run.status(), s_step.status());
            assert_eq!(s_run.iteration(), s_step.iteration());
            assert_eq!(s_run.total_fft_2d_calls(), 0);
            for i in 0..n {
                assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
                assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
            }

            // f32 bit-exact step == run.
            let mut l32_run = init_manifold::<f32>(kind, n, 17);
            let mut l32_step = l32_run.clone();
            let mut s32_run = TsNetSolver::<f32>::from_graph_pure_2hop(&grid, cfg).unwrap();
            let mut s32_step = TsNetSolver::<f32>::from_graph_pure_2hop(&grid, cfg).unwrap();
            assert_eq!(s32_run.run(&mut l32_run).unwrap(), SolverStatus::Completed);
            assert!(l32_run.validate_finite().is_ok());
            for _ in 0..24 {
                s32_step.step(&mut l32_step).unwrap();
            }
            assert_eq!(s32_step.status(), SolverStatus::Completed);
            for i in 0..n {
                assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
                assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
            }
        }
    }

    // Verify from_graph_row_normalized*, from_probabilities*, and separation on barbell graph.
    let barbell = make_barbell_graph(6);
    let cfg_bb = TsNetConfig {
        backend: TsNetBackend::Exact,
        learning_rate: Some(12.0),
        stage_iterations: [40, 40, 40],
        lambda_r_stage3: 0.4,
        ..TsNetConfig::pure_2hop()
    };
    let mut s_rn =
        TsNetSolver::<f64>::from_graph_row_normalized(&barbell, 0.5, None, cfg_bb).unwrap();
    let mut l_rn = init_manifold::<f64>(ManifoldKind::Spherical, barbell.num_nodes(), 5);
    assert_eq!(s_rn.run(&mut l_rn).unwrap(), SolverStatus::Completed);
    let (mean_edge, mean_all) = mean_geodesic_edge_and_all(&barbell, &l_rn);
    assert!(
        mean_edge < mean_all,
        "expected shorter geodesic edge lengths than non-edges: edge={mean_edge}, all={mean_all}"
    );

    let probs = SparseProbabilities {
        n_nodes: 3,
        indptr: vec![0, 1, 3, 4],
        indices: vec![1, 0, 2, 1],
        values: vec![0.25, 0.25, 0.25, 0.25],
    };
    let mut s_prob = TsNetSolver::<f64>::from_probabilities(
        probs,
        TsNetConfig {
            backend: TsNetBackend::Auto,
            stage_iterations: [5, 5, 5],
            ..TsNetConfig::pure_2hop()
        },
    )
    .unwrap();
    let mut l_prob = init_manifold::<f64>(ManifoldKind::HyperbolicLorentz, 3, 8);
    assert_eq!(s_prob.run(&mut l_prob).unwrap(), SolverStatus::Completed);
    assert!(l_prob.validate_finite().is_ok());
}

#[test]
fn test_tsnet_manifold_auto_large_and_errors() {
    // N = 272 > 256: Auto resolves to LinearFft at construction, yet manifold step routes cleanly to Exact.
    let large_grid = make_grid_graph(17, 16);
    let n = large_grid.num_nodes();
    let cfg_auto = TsNetConfig {
        backend: TsNetBackend::Auto,
        stage_iterations: [1, 0, 0],
        ..TsNetConfig::pure_2hop()
    };
    let mut s_auto =
        TsNetSolver::<f64>::from_graph_row_normalized(&large_grid, 0.5, Some(16), cfg_auto)
            .unwrap();
    assert_eq!(s_auto.resolved_backend(), TsNetBackend::LinearFft);
    let mut l_auto = init_manifold::<f64>(ManifoldKind::HyperbolicPoincare, n, 11);
    assert_eq!(s_auto.step(&mut l_auto).unwrap(), SolverStatus::Completed);
    assert_eq!(s_auto.total_fft_2d_calls(), 0);
    assert!(l_auto.validate_finite().is_ok());

    // Euclidean-only backends return CoreError::UnsupportedOperation on ManifoldLayout2d.
    let grid = make_grid_graph(3, 3);
    for bad_backend in [
        TsNetBackend::BarnesHut,
        TsNetBackend::FftInterpolation,
        TsNetBackend::LinearFft,
    ] {
        let cfg = TsNetConfig {
            backend: bad_backend,
            stage_iterations: [3, 0, 0],
            ..TsNetConfig::pure_2hop()
        };
        let mut s = TsNetSolver::<f64>::from_graph_pure_2hop(&grid, cfg).unwrap();
        let mut l = init_manifold::<f64>(ManifoldKind::Torus2d, 9, 4);
        assert!(matches!(
            s.step(&mut l),
            Err(CoreError::UnsupportedOperation(_))
        ));
    }

    // Edge cases: n=0, n=1, edgeless, coincident.
    let cfg_ex = TsNetConfig {
        backend: TsNetBackend::Exact,
        stage_iterations: [4, 4, 4],
        ..TsNetConfig::pure_2hop()
    };
    let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let mut s0 = TsNetSolver::<f64>::from_graph_pure_2hop(&g0, cfg_ex).unwrap();
    let mut l0 = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Spherical, 0);
    assert_eq!(s0.run(&mut l0).unwrap(), SolverStatus::Converged);

    let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let mut s1 = TsNetSolver::<f64>::from_graph_pure_2hop(&g1, cfg_ex).unwrap();
    let mut l1 = ManifoldLayout2d::from_xy(ManifoldKind::Torus2d, vec![0.3], vec![0.4]).unwrap();
    assert_eq!(s1.run(&mut l1).unwrap(), SolverStatus::Converged);

    let g_coin = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, false, false).unwrap();
    let mut s_coin = TsNetSolver::<f64>::from_graph_pure_2hop(&g_coin, cfg_ex).unwrap();
    let mut l_coin = ManifoldLayout2d::<f64>::zeros(ManifoldKind::HyperbolicPoincare, 3);
    assert_eq!(s_coin.run(&mut l_coin).unwrap(), SolverStatus::Completed);
    assert!(l_coin.validate_finite().is_ok());

    // Node mismatch and ActiveLayoutKind lock across 2d, nd, and manifold kinds.
    let mut s_err = TsNetSolver::<f64>::from_graph_pure_2hop(&grid, cfg_ex).unwrap();
    let mut l_wrong = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Torus2d, 4);
    assert!(matches!(
        s_err.step(&mut l_wrong),
        Err(CoreError::DimensionMismatch { .. })
    ));

    let mut l2 = Layout2d::<f64>::init(9, 1, 1.0).unwrap();
    let mut lnd = LayoutNd::<f64>::init(9, 3, 1, 1.0).unwrap();
    let mut l_tor = init_manifold::<f64>(ManifoldKind::Torus2d, 9, 2);
    let mut l_sph = init_manifold::<f64>(ManifoldKind::Spherical, 9, 2);

    let mut s_from_2d = TsNetSolver::<f64>::from_graph_pure_2hop(&grid, cfg_ex).unwrap();
    s_from_2d.step(&mut l2).unwrap();
    assert!(matches!(
        s_from_2d.step(&mut lnd),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        s_from_2d.step(&mut l_tor),
        Err(CoreError::InvalidInput(_))
    ));

    let mut s_from_nd = TsNetSolver::<f64>::from_graph_pure_2hop(&grid, cfg_ex).unwrap();
    s_from_nd.step(&mut lnd).unwrap();
    assert!(matches!(
        s_from_nd.step(&mut l2),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        s_from_nd.step(&mut l_tor),
        Err(CoreError::InvalidInput(_))
    ));

    let mut s_from_m = TsNetSolver::<f64>::from_graph_pure_2hop(&grid, cfg_ex).unwrap();
    s_from_m.step(&mut l_tor).unwrap();
    assert!(matches!(
        s_from_m.step(&mut l2),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        s_from_m.step(&mut lnd),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        s_from_m.step(&mut l_sph),
        Err(CoreError::InvalidInput(_))
    ));
}
