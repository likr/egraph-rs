use egraph_core::{
    CoreError, CsrGraph, IterativeSolver, Layout2d, Layout3d, LayoutGenerator, LayoutNd,
    ManifoldLayout2d, SolverStatus,
};
use egraph_linalg_mds::PivotMds;

use crate::sampling_utils::BLOCK_SHUFFLE_THRESHOLD;
use crate::test_support::{
    ALL_MANIFOLDS, assert_manifold_step_eq_run, assert_riemannian_step_eq_run, barbell_graph,
    circle_layout, compute_np2, grid_graph, layout_2d_to_nd,
};
use crate::{SnapTFdpConfig, SnapTFdpSolver};

#[test]
fn test_snap_t_fdp_bit_exact_compact_memory_and_convergence() {
    let barbell = barbell_graph(6);
    let grid = grid_graph(4, 4);
    let def_cfg = SnapTFdpConfig::default();
    assert_eq!(def_cfg.degree_beta, 0.0);
    assert!(def_cfg.bounded_attraction);
    let cfg = SnapTFdpConfig::new()
        .with_iterations(80)
        .with_negatives_per_edge(3)
        .with_learning_rate(1.0)
        .with_alpha(1.0)
        .with_repulsion_scale(1.0)
        .with_rho_start(1.0)
        .with_rho_end(0.1)
        .with_rho_power(1.0)
        .with_degree_beta(0.0)
        .with_bounded_attraction(true)
        .with_seed(42);

    // Verify compact storage (12 bytes/edge unweighted, zero materialized negatives, CSR, or deg_scales).
    let s_check = SnapTFdpSolver::<Layout2d<f64>>::new(&grid, cfg.clone()).unwrap();
    assert_eq!(s_check.edges_u().len(), grid.num_edges());
    assert_eq!(s_check.edges_v().len(), grid.num_edges());
    assert_eq!(s_check.edge_order().len(), grid.num_edges());
    assert!(s_check.edge_weights().is_none());
    assert!(s_check.deg_scales().is_none());
    let bytes_stored = s_check.edge_buffer_bytes();
    assert_eq!(bytes_stored, grid.num_edges() * 12);

    // Bit-exact step() == run() for f64 and f32, plus 2D/3D/ND parity.
    let mut s64_run = SnapTFdpSolver::<Layout2d<f64>>::new(&grid, cfg.clone()).unwrap();
    let mut s64_step = SnapTFdpSolver::<Layout2d<f64>>::new(&grid, cfg.clone()).unwrap();
    let mut l64_run = circle_layout::<f64>(grid.num_nodes(), 1.0);
    let mut l64_step = l64_run.clone();
    assert_eq!(s64_run.run(&mut l64_run).unwrap(), SolverStatus::Completed);
    assert_eq!(s64_run.current_eta(), 0.0);
    s64_step.run_steps(&mut l64_step, 17).unwrap();
    while !s64_step.status().is_finished() {
        s64_step.step(&mut l64_step).unwrap();
    }
    assert!(l64_run.is_all_finite());
    for i in 0..grid.num_nodes() {
        assert_eq!(l64_run.x()[i].to_bits(), l64_step.x()[i].to_bits());
        assert_eq!(l64_run.y()[i].to_bits(), l64_step.y()[i].to_bits());
    }

    let out_nd2 = assert_riemannian_step_eq_run(
        SnapTFdpSolver::<LayoutNd<f64>>::new(&grid, cfg.clone()).unwrap(),
        SnapTFdpSolver::<LayoutNd<f64>>::new(&grid, cfg.clone()).unwrap(),
        &layout_2d_to_nd(&circle_layout::<f64>(grid.num_nodes(), 1.0)),
        17,
        "snap_nd2",
    );
    for i in 0..grid.num_nodes() {
        assert_eq!(l64_run.x()[i].to_bits(), out_nd2.coords()[i * 2].to_bits());
        assert_eq!(
            l64_run.y()[i].to_bits(),
            out_nd2.coords()[i * 2 + 1].to_bits()
        );
    }

    assert_riemannian_step_eq_run(
        SnapTFdpSolver::<Layout2d<f32>>::new(&grid, cfg.clone()).unwrap(),
        SnapTFdpSolver::<Layout2d<f32>>::new(&grid, cfg.clone()).unwrap(),
        &circle_layout::<f32>(grid.num_nodes(), 1.0),
        9,
        "snap_2d_f32",
    );
    assert_riemannian_step_eq_run(
        SnapTFdpSolver::<Layout3d<f64>>::new(&grid, cfg.clone()).unwrap(),
        SnapTFdpSolver::<Layout3d<f64>>::new(&grid, cfg.clone()).unwrap(),
        &Layout3d::<f64>::init(grid.num_nodes(), 7, 1.0).unwrap(),
        9,
        "snap_3d",
    );
    assert_riemannian_step_eq_run(
        SnapTFdpSolver::<LayoutNd<f32>>::new(&grid, cfg.clone()).unwrap(),
        SnapTFdpSolver::<LayoutNd<f32>>::new(&grid, cfg.clone()).unwrap(),
        &LayoutNd::<f32>::init(grid.num_nodes(), 5, 7, 1.0).unwrap(),
        9,
        "snap_5d_f32",
    );

    // Barbell clique separation & grid NP2 preservation.
    let mut bb_solver =
        SnapTFdpSolver::<Layout2d<f64>>::new(&barbell, cfg.clone().with_iterations(150)).unwrap();
    let mut bb_layout: Layout2d<f64> = PivotMds::new(&barbell).generate().unwrap();
    bb_solver.run(&mut bb_layout).unwrap();
    assert!(bb_layout.is_all_finite());
    let c0x = bb_layout.x()[..6].iter().sum::<f64>() / 6.0;
    let c0y = bb_layout.y()[..6].iter().sum::<f64>() / 6.0;
    let c1x = bb_layout.x()[6..].iter().sum::<f64>() / 6.0;
    let c1y = bb_layout.y()[6..].iter().sum::<f64>() / 6.0;
    let centroid_dist = (c0x - c1x).hypot(c0y - c1y);
    assert!(
        centroid_dist > 0.5,
        "expected separated barbell cliques, got {centroid_dist}"
    );
    assert!(compute_np2(&barbell, &bb_layout) >= 0.85);

    let mut g_solver = SnapTFdpSolver::<Layout2d<f64>>::new(&grid, cfg.clone()).unwrap();
    let mut g_layout: Layout2d<f64> = PivotMds::new(&grid).generate().unwrap();
    g_solver.run(&mut g_layout).unwrap();
    assert!(compute_np2(&grid, &g_layout) >= 0.75);

    // Cache-blocked shuffle path (> 65_536 edges) step == run bit-exactness.
    let n_big = 500usize;
    let mut big_edges = Vec::with_capacity(75_000);
    for u in 0..n_big {
        for off in 1..=150 {
            let v = (u + off) % n_big;
            big_edges.push((u.min(v) as u32, u.max(v) as u32));
        }
    }
    big_edges.sort_unstable();
    big_edges.dedup();
    assert!(big_edges.len() > BLOCK_SHUFFLE_THRESHOLD);
    let big_g = CsrGraph::<f64>::from_edges(n_big, &big_edges, None, false, false).unwrap();
    let big_cfg = cfg.clone().with_iterations(3).with_negatives_per_edge(2);
    let mut b_run = SnapTFdpSolver::<Layout2d<f64>>::new(&big_g, big_cfg.clone()).unwrap();
    let mut b_step = SnapTFdpSolver::<Layout2d<f64>>::new(&big_g, big_cfg).unwrap();
    assert!(!b_run.block_order().is_empty());
    let mut bl_run = circle_layout::<f64>(n_big, 1.0);
    let mut bl_step = bl_run.clone();
    b_run.run(&mut bl_run).unwrap();
    b_step.step(&mut bl_step).unwrap();
    b_step.run(&mut bl_step).unwrap();
    for i in 0..n_big {
        assert_eq!(bl_run.x()[i].to_bits(), bl_step.x()[i].to_bits());
        assert_eq!(bl_run.y()[i].to_bits(), bl_step.y()[i].to_bits());
    }

    // Weighted directed graph works cleanly and allocates f32 weights only when weighted.
    let dwg = CsrGraph::<f64>::from_edges(
        4,
        &[(0, 1), (1, 0), (1, 2), (2, 3)],
        Some(&[2.0, 1.0, 1.5, 0.5]),
        true,
        false,
    )
    .unwrap();
    let mut dw_solver = SnapTFdpSolver::<Layout2d<f64>>::new(&dwg, cfg.clone()).unwrap();
    assert!(dw_solver.edge_weights().is_some());
    assert_eq!(dw_solver.num_springs(), 3);
    let mut dw_layout = circle_layout::<f64>(4, 1.0);
    dw_solver.run(&mut dw_layout).unwrap();
    assert!(dw_layout.is_all_finite());

    // Manifold parity across all 4 geometries.
    let m_cfg = cfg.clone().with_iterations(8);
    for kind in ALL_MANIFOLDS {
        assert_manifold_step_eq_run(
            SnapTFdpSolver::<ManifoldLayout2d<f64>>::new(&grid, m_cfg.clone()).unwrap(),
            SnapTFdpSolver::<ManifoldLayout2d<f64>>::new(&grid, m_cfg.clone()).unwrap(),
            kind,
            grid.num_nodes(),
            "snap_manifold_f64",
        );
    }

    // Validation & edge-case guards (including run_steps(0)).
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    assert_eq!(
        SnapTFdpSolver::<Layout2d<f64>>::new(&empty, cfg.clone())
            .unwrap()
            .run(&mut Layout2d::zeros(0))
            .unwrap(),
        SolverStatus::Completed
    );
    assert_eq!(
        SnapTFdpSolver::<Layout2d<f64>>::new(&single, cfg.clone())
            .unwrap()
            .run(&mut Layout2d::zeros(1))
            .unwrap(),
        SolverStatus::Completed
    );
    assert_eq!(
        SnapTFdpSolver::<Layout2d<f64>>::new(&grid, cfg.clone().with_iterations(0))
            .unwrap()
            .status(),
        SolverStatus::Completed
    );
    let mut s_zero = SnapTFdpSolver::<Layout2d<f64>>::new(&grid, cfg.clone()).unwrap();
    let mut bad_l = Layout2d::from_xy(
        vec![f64::NAN; grid.num_nodes()],
        vec![0.0; grid.num_nodes()],
    )
    .unwrap();
    assert!(matches!(
        s_zero.run_steps(&mut bad_l, 0),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert_eq!(s_zero.status(), SolverStatus::Initialized);

    for bad in [
        cfg.clone().with_learning_rate(0.0),
        cfg.clone().with_learning_rate(f64::NAN),
        cfg.clone().with_alpha(-0.1),
        cfg.clone().with_repulsion_scale(-1.0),
        cfg.clone().with_rho_start(-0.5),
        cfg.clone().with_rho_end(f64::INFINITY),
        cfg.clone().with_rho_power(0.0),
        cfg.clone().with_degree_beta(-0.1),
        cfg.clone().with_degree_beta(f64::NAN),
        cfg.with_degree_beta(f64::INFINITY),
    ] {
        assert!(SnapTFdpSolver::<Layout2d<f64>>::new(&grid, bad).is_err());
    }
}

#[test]
fn test_snap_t_fdp_degree_beta_and_bounded_attraction() {
    let n_star = 17usize;
    let star_edges: Vec<(u32, u32)> = (1..n_star).map(|i| (0u32, i as u32)).collect();
    let star = CsrGraph::<f64>::from_edges(n_star, &star_edges, None, false, false).unwrap();

    let cfg_zero = SnapTFdpConfig::new()
        .with_iterations(1)
        .with_negatives_per_edge(0)
        .with_learning_rate(1.0)
        .with_alpha(1.0)
        .with_degree_beta(0.0)
        .with_bounded_attraction(true)
        .with_seed(42);
    let cfg_beta = cfg_zero.clone().with_degree_beta(0.5);
    let mut s_zero = SnapTFdpSolver::<Layout2d<f64>>::new(&star, cfg_zero).unwrap();
    let mut s_beta = SnapTFdpSolver::<Layout2d<f64>>::new(&star, cfg_beta).unwrap();
    assert!(s_zero.deg_scales().is_none());
    let deg_scales = s_beta.deg_scales().unwrap();
    assert_eq!(deg_scales.len(), n_star);

    let mut init_star_x = vec![0.0; n_star];
    let mut init_star_y = vec![0.0; n_star];
    for i in 1..n_star {
        let ang = 2.0 * std::f64::consts::PI * ((i - 1) as f64) / ((n_star - 1) as f64);
        init_star_x[i] = 2.0 * ang.cos();
        init_star_y[i] = 2.0 * ang.sin();
    }
    let mut l_zero = Layout2d::from_xy(init_star_x.clone(), init_star_y.clone()).unwrap();
    let mut l_beta = Layout2d::from_xy(init_star_x, init_star_y).unwrap();
    s_zero.step(&mut l_zero).unwrap();
    s_beta.step(&mut l_beta).unwrap();
    let mean_r_zero: f64 = (1..n_star)
        .map(|i| (l_zero.x()[i] - l_zero.x()[0]).hypot(l_zero.y()[i] - l_zero.y()[0]))
        .sum::<f64>()
        / ((n_star - 1) as f64);
    let mean_r_beta: f64 = (1..n_star)
        .map(|i| (l_beta.x()[i] - l_beta.x()[0]).hypot(l_beta.y()[i] - l_beta.y()[0]))
        .sum::<f64>()
        / ((n_star - 1) as f64);
    assert!(
        mean_r_beta > mean_r_zero,
        "expected degree_beta=0.5 to attenuate pull (mean_r_beta={mean_r_beta} <= mean_r_zero={mean_r_zero})"
    );

    let pair_g = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
    let mut s_bounded = SnapTFdpSolver::<Layout2d<f64>>::new(
        &pair_g,
        SnapTFdpConfig::new()
            .with_iterations(1)
            .with_negatives_per_edge(0)
            .with_learning_rate(1.0)
            .with_alpha(1.0)
            .with_degree_beta(0.0)
            .with_bounded_attraction(true)
            .with_seed(7),
    )
    .unwrap();
    let mut s_unbounded = SnapTFdpSolver::<Layout2d<f64>>::new(
        &pair_g,
        SnapTFdpConfig::new()
            .with_iterations(1)
            .with_negatives_per_edge(0)
            .with_learning_rate(1.0)
            .with_alpha(1.0)
            .with_degree_beta(0.0)
            .with_bounded_attraction(false)
            .with_seed(7),
    )
    .unwrap();
    let mut l_b = Layout2d::from_xy(vec![0.0, 1000.0], vec![0.0, 0.0]).unwrap();
    let mut l_u = Layout2d::from_xy(vec![0.0, 1000.0], vec![0.0, 0.0]).unwrap();
    s_bounded.step(&mut l_b).unwrap();
    s_unbounded.step(&mut l_u).unwrap();
    let disp_b = l_b.x()[0].abs();
    let disp_u = l_u.x()[0].abs();
    assert!(
        disp_b < 0.01,
        "expected bounded long-range displacement O(1/d) << 1, got {disp_b}"
    );
    assert!(
        disp_u > 90.0,
        "expected legacy unbounded linear tail displacement ~100, got {disp_u}"
    );

    let grid = grid_graph(4, 4);
    for (beta, bounded) in [(0.5, true), (0.0, false)] {
        let cfg = SnapTFdpConfig::new()
            .with_iterations(15)
            .with_negatives_per_edge(3)
            .with_degree_beta(beta)
            .with_bounded_attraction(bounded)
            .with_seed(42);
        assert_riemannian_step_eq_run(
            SnapTFdpSolver::<Layout2d<f64>>::new(&grid, cfg.clone()).unwrap(),
            SnapTFdpSolver::<Layout2d<f64>>::new(&grid, cfg.clone()).unwrap(),
            &circle_layout::<f64>(grid.num_nodes(), 1.0),
            5,
            "snap_toggle_f64",
        );
        assert_riemannian_step_eq_run(
            SnapTFdpSolver::<Layout2d<f32>>::new(&grid, cfg.clone()).unwrap(),
            SnapTFdpSolver::<Layout2d<f32>>::new(&grid, cfg).unwrap(),
            &circle_layout::<f32>(grid.num_nodes(), 1.0),
            5,
            "snap_toggle_f32",
        );
    }

    let dwg = CsrGraph::<f64>::from_edges(
        4,
        &[(0, 1), (1, 0), (1, 2), (2, 3)],
        Some(&[2.0, 1.0, 1.5, 0.5]),
        true,
        false,
    )
    .unwrap();
    let mut dw_solver = SnapTFdpSolver::<Layout2d<f64>>::new(
        &dwg,
        SnapTFdpConfig::new()
            .with_iterations(20)
            .with_degree_beta(0.4)
            .with_bounded_attraction(true)
            .with_seed(13),
    )
    .unwrap();
    assert_eq!(dw_solver.deg_scales().map(<[f32]>::len), Some(4));
    let mut dw_layout = circle_layout::<f64>(4, 1.0);
    dw_solver.run(&mut dw_layout).unwrap();
    assert!(dw_layout.is_all_finite());
}
