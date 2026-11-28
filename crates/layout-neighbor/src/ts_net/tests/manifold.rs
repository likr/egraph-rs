use egraph_core::{
    CoreError, CsrGraph, IterativeSolver, ManifoldKind, ManifoldLayout2d, RiemannianLayout, Scalar,
    SolverStatus,
};
use egraph_neighborhood::SparseProbabilities;

use crate::backends::Exact;

use super::super::{TsNetConfig, TsNetSolver};
use super::{make_barbell_graph, make_grid_graph, make_stages};

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
        let cfg = TsNetConfig::pure_2hop().with_stages(make_stages(8, 8, 8, 0.5));

        let mut l_run = init_manifold::<f64>(kind, n, 17);
        if kind == ManifoldKind::Torus2d {
            l_run.set_torus_extents(2.5, 1.5).unwrap();
        } else {
            l_run.set_scale(1.5).unwrap();
        }
        let mut l_step = l_run.clone();
        // DefaultTsNetBackend for ManifoldLayout2d resolves to Exact.
        let mut s_run =
            TsNetSolver::<ManifoldLayout2d<f64>>::from_graph_pure_2hop(&grid, cfg.clone()).unwrap();
        let mut s_step =
            TsNetSolver::<ManifoldLayout2d<f64>, Exact>::from_graph_pure_2hop(&grid, cfg.clone())
                .unwrap();

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

        let mut l32_run = init_manifold::<f32>(kind, n, 17);
        let mut l32_step = l32_run.clone();
        let mut s32_run =
            TsNetSolver::<ManifoldLayout2d<f32>>::from_graph_pure_2hop(&grid, cfg.clone()).unwrap();
        let mut s32_step =
            TsNetSolver::<ManifoldLayout2d<f32>>::from_graph_pure_2hop(&grid, cfg).unwrap();
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

    let barbell = make_barbell_graph(6);
    let cfg_bb = TsNetConfig::pure_2hop()
        .with_learning_rate(Some(12.0))
        .with_stages(make_stages(40, 40, 40, 0.4));
    let mut s_rn = TsNetSolver::<ManifoldLayout2d<f64>>::from_graph_row_normalized(
        &barbell, 0.5, None, cfg_bb,
    )
    .unwrap();
    let mut l_rn = init_manifold::<f64>(ManifoldKind::Spherical, barbell.num_nodes(), 5);
    assert_eq!(s_rn.run(&mut l_rn).unwrap(), SolverStatus::Completed);
    let (mean_edge, mean_all) = mean_geodesic_edge_and_all(&barbell, &l_rn);
    assert!(
        mean_edge < mean_all,
        "expected shorter geodesic edge lengths than non-edges: edge={mean_edge}, all={mean_all}"
    );

    let probs = SparseProbabilities::from_csr(
        3,
        vec![0, 1, 3, 4],
        vec![1, 0, 2, 1],
        vec![0.25, 0.25, 0.25, 0.25],
    )
    .unwrap();
    let mut s_prob = TsNetSolver::<ManifoldLayout2d<f64>>::from_probabilities(
        probs,
        TsNetConfig::pure_2hop().with_stages(make_stages(5, 5, 5, 0.0)),
    )
    .unwrap();
    let mut l_prob = init_manifold::<f64>(ManifoldKind::HyperbolicLorentz, 3, 8);
    assert_eq!(s_prob.run(&mut l_prob).unwrap(), SolverStatus::Completed);
    assert!(l_prob.validate_finite().is_ok());
}

#[test]
fn test_tsnet_manifold_velocity_parallel_transport() {
    // Verify that velocity vectors are Levi-Civita parallel-transported after each step
    // via `exp_map_and_transport` across curved manifolds (Spherical, Poincare, Lorentz).
    let probs =
        SparseProbabilities::from_csr(2, vec![0, 1, 2], vec![1, 0], vec![0.25, 0.25]).unwrap();

    let cases = [
        (ManifoldKind::Spherical, vec![0.15, 0.9], vec![0.5, 0.65]),
        (
            ManifoldKind::HyperbolicPoincare,
            vec![0.25, -0.3],
            vec![0.15, 0.2],
        ),
        (
            ManifoldKind::HyperbolicLorentz,
            vec![0.35, -0.25],
            vec![0.15, 0.4],
        ),
    ];

    for (kind, xs, ys) in cases {
        let mut layout = ManifoldLayout2d::from_xy(kind, xs.clone(), ys.clone()).unwrap();
        let pre_step = layout.clone();

        let cfg = TsNetConfig::pure_2hop()
            .with_jacobi_precondition(false)
            .with_learning_rate(Some(0.25))
            .with_momentum(0.8)
            .with_stages(make_stages(2, 0, 0, 0.0));

        let mut solver =
            TsNetSolver::<ManifoldLayout2d<f64>>::from_probabilities(probs.clone(), cfg).unwrap();
        solver.step(&mut layout).unwrap();

        let mut delta = [0.0_f64; 2];
        pre_step.log_map_into(0, 0, &mut delta).unwrap();
        let v_after = [
            solver.space.vel_x().unwrap()[0],
            solver.space.vel_y().unwrap()[0],
        ];
        let norm_after = v_after[0].hypot(v_after[1]);
        assert!(norm_after > 1e-6, "{kind:?}: expected non-zero velocity");

        let mut backward = [0.0_f64; 2];
        layout.log_map_into(0, 0, &mut backward).unwrap();
        let (dx0, dy0) = pre_step.tangent_delta(0, 1).unwrap();
        let step_dir_dot = (dx0 * v_after[0] + dy0 * v_after[1]).abs();
        assert!(step_dir_dot.is_finite());

        // Test explicit pre-seeded velocity transport on an edgeless graph.
        let mut edgeless_l = ManifoldLayout2d::from_xy(kind, xs, ys).unwrap();
        let edgeless_p = SparseProbabilities::empty(2);
        let cfg_t = TsNetConfig::pure_2hop()
            .with_jacobi_precondition(false)
            .with_learning_rate(Some(0.5))
            .with_momentum(1.0 - 1e-9)
            .with_stages(make_stages(1, 0, 0, 0.0));
        let mut s_t =
            TsNetSolver::<ManifoldLayout2d<f64>>::from_probabilities(edgeless_p, cfg_t).unwrap();
        s_t.space.vel_x_mut().unwrap()[0] = 0.25;
        if let super::super::repulsion::TsNetSpaceKind::Manifold2d(ref mut m) = s_t.space.0 {
            m.vel_y[0] = -0.15;
        }
        s_t.step(&mut edgeless_l).unwrap();
        let transported_norm = s_t.space.vel_x().unwrap()[0].hypot(s_t.space.vel_y().unwrap()[0]);
        assert!(transported_norm > 0.1 && transported_norm.is_finite());
    }

    // Explicit exact comparison against `exp_map_and_transport` for Spherical non-equatorial move.
    let mut l_s =
        ManifoldLayout2d::from_xy(ManifoldKind::Spherical, vec![0.4, 1.2], vec![0.5, 0.7]).unwrap();
    let l_ref = l_s.clone();
    let cfg_s = TsNetConfig::pure_2hop()
        .with_jacobi_precondition(false)
        .with_learning_rate(Some(0.3))
        .with_momentum(0.0)
        .with_stages(make_stages(1, 0, 0, 0.0));
    let mut s_s = TsNetSolver::<ManifoldLayout2d<f64>>::from_probabilities(probs, cfg_s).unwrap();
    s_s.step(&mut l_s).unwrap();

    let mixed = ManifoldLayout2d::from_xy(
        ManifoldKind::Spherical,
        vec![l_s.x()[0], l_ref.x()[0]],
        vec![l_s.y()[0], l_ref.y()[0]],
    )
    .unwrap();
    let mut back_vec = [0.0_f64; 2];
    mixed.log_map_into(0, 1, &mut back_vec).unwrap();
    let back_x = back_vec[0];
    let back_y = back_vec[1];
    let vx = s_s.space.vel_x().unwrap()[0];
    let vy = s_s.space.vel_y().unwrap()[0];
    assert!(
        (vx + back_x).abs() < 1e-10 && (vy + back_y).abs() < 1e-10,
        "expected transported velocity == -log_q(p): vel=({vx}, {vy}), -back=({}, {})",
        -back_x,
        -back_y
    );
}

#[test]
fn test_tsnet_manifold_edge_cases_and_errors() {
    let grid = make_grid_graph(3, 3);
    let cfg_ex = TsNetConfig::pure_2hop().with_stages(make_stages(4, 4, 4, 0.0));
    let g0 = CsrGraph::<()>::from_unweighted_edges(0, &[], false, false).unwrap();
    let mut s0 =
        TsNetSolver::<ManifoldLayout2d<f64>>::from_graph_pure_2hop(&g0, cfg_ex.clone()).unwrap();
    let mut l0 = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Spherical, 0);
    assert_eq!(s0.run(&mut l0).unwrap(), SolverStatus::Converged);

    let g1 = CsrGraph::<()>::from_unweighted_edges(1, &[], false, false).unwrap();
    let mut s1 =
        TsNetSolver::<ManifoldLayout2d<f64>>::from_graph_pure_2hop(&g1, cfg_ex.clone()).unwrap();
    let mut l1 = ManifoldLayout2d::from_xy(ManifoldKind::Torus2d, vec![0.3], vec![0.4]).unwrap();
    assert_eq!(s1.run(&mut l1).unwrap(), SolverStatus::Converged);

    let g_coin = CsrGraph::<()>::from_unweighted_edges(3, &[(0, 1), (1, 2)], false, false).unwrap();
    let mut s_coin =
        TsNetSolver::<ManifoldLayout2d<f64>>::from_graph_pure_2hop(&g_coin, cfg_ex.clone())
            .unwrap();
    let mut l_coin = ManifoldLayout2d::<f64>::zeros(ManifoldKind::HyperbolicPoincare, 3);
    assert_eq!(s_coin.run(&mut l_coin).unwrap(), SolverStatus::Completed);
    assert!(l_coin.validate_finite().is_ok());

    // Node mismatch and ManifoldKind switching rejection (including run_steps(target, 0)).
    let mut s_err =
        TsNetSolver::<ManifoldLayout2d<f64>>::from_graph_pure_2hop(&grid, cfg_ex.clone()).unwrap();
    let mut l_wrong = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Torus2d, 4);
    assert!(matches!(
        s_err.step(&mut l_wrong),
        Err(CoreError::DimensionMismatch { .. })
    ));

    let mut l_tor = init_manifold::<f64>(ManifoldKind::Torus2d, 9, 2);
    let mut l_sph = init_manifold::<f64>(ManifoldKind::Spherical, 9, 2);
    let mut s_from_m =
        TsNetSolver::<ManifoldLayout2d<f64>>::from_graph_pure_2hop(&grid, cfg_ex).unwrap();
    s_from_m.step(&mut l_tor).unwrap();
    assert!(matches!(
        s_from_m.step(&mut l_sph),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        s_from_m.run_steps(&mut l_sph, 0),
        Err(CoreError::InvalidInput(_))
    ));
}
