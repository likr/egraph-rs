use egraph_core::{
    CsrGraph, IterativeSolver, Layout2d, ManifoldKind, ManifoldLayout2d, SolverStatus,
};
use egraph_linalg_mds::{PivotMdsConfig, pivot_mds};

use super::super::sgd::{SgdSolver, SnapTFdpConfig};
use super::pairs::BLOCK_SHUFFLE_THRESHOLD;
use super::test_support::{barbell_graph, circle_layout, compute_np2, grid_graph};

#[test]
fn test_snap_t_fdp_bit_exact_compact_memory_and_convergence() {
    let barbell = barbell_graph(6);
    let grid = grid_graph(4, 4);
    let def_cfg = SnapTFdpConfig::default();
    assert_eq!(def_cfg.degree_beta, 0.0);
    assert!(def_cfg.bounded_attraction);
    let cfg = SnapTFdpConfig {
        iterations: 80,
        negatives_per_edge: 3,
        learning_rate: 1.0,
        alpha: 1.0,
        repulsion_scale: 1.0,
        rho_start: 1.0,
        rho_end: 0.1,
        rho_power: 1.0,
        degree_beta: 0.0,
        bounded_attraction: true,
        seed: 42,
    };

    // Verify compact storage (12 bytes/edge unweighted, zero materialized negatives, CSR, or deg_scales).
    let s_check = SgdSolver::<f64>::snap_t_fdp(&grid, cfg).unwrap();
    let st = s_check.snap_t_fdp.as_ref().unwrap();
    assert_eq!(s_check.pairs.len(), 0);
    assert_eq!(st.edges_u.len(), grid.num_edges());
    assert_eq!(st.edges_v.len(), grid.num_edges());
    assert_eq!(st.edge_order.len(), grid.num_edges());
    assert!(st.edge_weights.is_none());
    assert!(st.deg_scales.is_none());
    let bytes_stored = (st.edges_u.capacity() + st.edges_v.capacity() + st.edge_order.capacity())
        * std::mem::size_of::<u32>();
    assert_eq!(bytes_stored, grid.num_edges() * 12);

    // Bit-exact step() == run() for f64 and f32.
    let mut s64_run = SgdSolver::<f64>::snap_t_fdp(&grid, cfg).unwrap();
    let mut s64_step = SgdSolver::<f64>::snap_t_fdp(&grid, cfg).unwrap();
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

    let mut s32_run = SgdSolver::<f32>::snap_t_fdp(&grid, cfg).unwrap();
    let mut s32_step = SgdSolver::<f32>::snap_t_fdp(&grid, cfg).unwrap();
    let mut l32_run = circle_layout::<f32>(grid.num_nodes(), 1.0);
    let mut l32_step = l32_run.clone();
    s32_run.run(&mut l32_run).unwrap();
    s32_step.run_steps(&mut l32_step, 9).unwrap();
    for _ in 0..5 {
        s32_step.step(&mut l32_step).unwrap();
    }
    s32_step.run(&mut l32_step).unwrap();
    for i in 0..grid.num_nodes() {
        assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
        assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
    }

    // Barbell clique separation & grid NP2 preservation.
    let mut bb_solver = SgdSolver::<f64>::snap_t_fdp(
        &barbell,
        SnapTFdpConfig {
            iterations: 150,
            ..cfg
        },
    )
    .unwrap();
    let mut bb_layout = pivot_mds(&barbell, &PivotMdsConfig::default()).unwrap();
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

    let mut g_solver = SgdSolver::<f64>::snap_t_fdp(&grid, cfg).unwrap();
    let mut g_layout = pivot_mds(&grid, &PivotMdsConfig::default()).unwrap();
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
    let big_cfg = SnapTFdpConfig {
        iterations: 3,
        negatives_per_edge: 2,
        ..cfg
    };
    let mut b_run = SgdSolver::<f64>::snap_t_fdp(&big_g, big_cfg).unwrap();
    let mut b_step = SgdSolver::<f64>::snap_t_fdp(&big_g, big_cfg).unwrap();
    assert!(!b_run.block_order.is_empty());
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
    let mut dw_solver = SgdSolver::<f64>::snap_t_fdp(&dwg, cfg).unwrap();
    assert!(
        dw_solver
            .snap_t_fdp
            .as_ref()
            .unwrap()
            .edge_weights
            .is_some()
    );
    assert_eq!(dw_solver.num_springs(), 3);
    let mut dw_layout = circle_layout::<f64>(4, 1.0);
    dw_solver.run(&mut dw_layout).unwrap();
    assert!(dw_layout.is_all_finite());

    // Validation & edge-case guards.
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    assert_eq!(
        SgdSolver::<f64>::snap_t_fdp(&empty, cfg)
            .unwrap()
            .run(&mut Layout2d::zeros(0))
            .unwrap(),
        SolverStatus::Completed
    );
    assert_eq!(
        SgdSolver::<f64>::snap_t_fdp(&single, cfg)
            .unwrap()
            .run(&mut Layout2d::zeros(1))
            .unwrap(),
        SolverStatus::Completed
    );
    assert_eq!(
        SgdSolver::<f64>::snap_t_fdp(
            &grid,
            SnapTFdpConfig {
                iterations: 0,
                ..cfg
            }
        )
        .unwrap()
        .status(),
        SolverStatus::Completed
    );
    for bad in [
        SnapTFdpConfig {
            learning_rate: 0.0,
            ..cfg
        },
        SnapTFdpConfig {
            learning_rate: f64::NAN,
            ..cfg
        },
        SnapTFdpConfig { alpha: -0.1, ..cfg },
        SnapTFdpConfig {
            repulsion_scale: -1.0,
            ..cfg
        },
        SnapTFdpConfig {
            rho_start: -0.5,
            ..cfg
        },
        SnapTFdpConfig {
            rho_end: f64::INFINITY,
            ..cfg
        },
        SnapTFdpConfig {
            rho_power: 0.0,
            ..cfg
        },
        SnapTFdpConfig {
            degree_beta: -0.1,
            ..cfg
        },
        SnapTFdpConfig {
            degree_beta: f64::NAN,
            ..cfg
        },
        SnapTFdpConfig {
            degree_beta: f64::INFINITY,
            ..cfg
        },
    ] {
        assert!(SgdSolver::<f64>::snap_t_fdp(&grid, bad).is_err());
    }
    let mut m_layout = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Torus2d, grid.num_nodes());
    assert!(
        SgdSolver::<f64>::snap_t_fdp(&grid, cfg)
            .unwrap()
            .step_manifold(&mut m_layout)
            .is_ok()
    );
    assert!(m_layout.as_layout().is_all_finite());
}

#[test]
fn test_snap_t_fdp_degree_beta_and_bounded_attraction() {
    // 1. Star graph: degree_beta = 0.5 allocates 4N-byte deg_scales and attenuates hub-leaf pull.
    let n_star = 17usize;
    let star_edges: Vec<(u32, u32)> = (1..n_star).map(|i| (0u32, i as u32)).collect();
    let star = CsrGraph::<f64>::from_edges(n_star, &star_edges, None, false, false).unwrap();

    let cfg_zero = SnapTFdpConfig {
        iterations: 1,
        negatives_per_edge: 0,
        learning_rate: 1.0,
        alpha: 1.0,
        degree_beta: 0.0,
        bounded_attraction: true,
        seed: 42,
        ..Default::default()
    };
    let cfg_beta = SnapTFdpConfig {
        degree_beta: 0.5,
        ..cfg_zero
    };
    let mut s_zero = SgdSolver::<f64>::snap_t_fdp(&star, cfg_zero).unwrap();
    let mut s_beta = SgdSolver::<f64>::snap_t_fdp(&star, cfg_beta).unwrap();
    assert!(s_zero.snap_t_fdp.as_ref().unwrap().deg_scales.is_none());
    let deg_scales = s_beta
        .snap_t_fdp
        .as_ref()
        .unwrap()
        .deg_scales
        .as_ref()
        .unwrap();
    assert_eq!(deg_scales.len(), n_star);
    assert_eq!(deg_scales.capacity(), n_star);

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

    // 2. Bounded long-range attraction vs legacy linear tail at d = 1000.0.
    let pair_g = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
    let mut s_bounded = SgdSolver::<f64>::snap_t_fdp(
        &pair_g,
        SnapTFdpConfig {
            iterations: 1,
            negatives_per_edge: 0,
            learning_rate: 1.0,
            alpha: 1.0,
            degree_beta: 0.0,
            bounded_attraction: true,
            seed: 7,
            ..Default::default()
        },
    )
    .unwrap();
    let mut s_unbounded = SgdSolver::<f64>::snap_t_fdp(
        &pair_g,
        SnapTFdpConfig {
            iterations: 1,
            negatives_per_edge: 0,
            learning_rate: 1.0,
            alpha: 1.0,
            degree_beta: 0.0,
            bounded_attraction: false,
            seed: 7,
            ..Default::default()
        },
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

    // 3. step == run bit-exactness (f64 & f32) across both (degree_beta=0.5, bounded=true)
    // and (degree_beta=0.0, bounded=false).
    let grid = grid_graph(4, 4);
    for (beta, bounded) in [(0.5, true), (0.0, false)] {
        let cfg = SnapTFdpConfig {
            iterations: 15,
            negatives_per_edge: 3,
            degree_beta: beta,
            bounded_attraction: bounded,
            seed: 42,
            ..Default::default()
        };
        let mut s64_r = SgdSolver::<f64>::snap_t_fdp(&grid, cfg).unwrap();
        let mut s64_s = SgdSolver::<f64>::snap_t_fdp(&grid, cfg).unwrap();
        let mut l64_r = circle_layout::<f64>(grid.num_nodes(), 1.0);
        let mut l64_s = l64_r.clone();
        s64_r.run(&mut l64_r).unwrap();
        s64_s.run_steps(&mut l64_s, 5).unwrap();
        while !s64_s.status().is_finished() {
            s64_s.step(&mut l64_s).unwrap();
        }
        for i in 0..grid.num_nodes() {
            assert_eq!(l64_r.x()[i].to_bits(), l64_s.x()[i].to_bits());
            assert_eq!(l64_r.y()[i].to_bits(), l64_s.y()[i].to_bits());
        }

        let mut s32_r = SgdSolver::<f32>::snap_t_fdp(&grid, cfg).unwrap();
        let mut s32_s = SgdSolver::<f32>::snap_t_fdp(&grid, cfg).unwrap();
        let mut l32_r = circle_layout::<f32>(grid.num_nodes(), 1.0);
        let mut l32_s = l32_r.clone();
        s32_r.run(&mut l32_r).unwrap();
        s32_s.run_steps(&mut l32_s, 5).unwrap();
        while !s32_s.status().is_finished() {
            s32_s.step(&mut l32_s).unwrap();
        }
        for i in 0..grid.num_nodes() {
            assert_eq!(l32_r.x()[i].to_bits(), l32_s.x()[i].to_bits());
            assert_eq!(l32_r.y()[i].to_bits(), l32_s.y()[i].to_bits());
        }
    }

    // 4. Directed weighted graph with degree_beta > 0.0.
    let dwg = CsrGraph::<f64>::from_edges(
        4,
        &[(0, 1), (1, 0), (1, 2), (2, 3)],
        Some(&[2.0, 1.0, 1.5, 0.5]),
        true,
        false,
    )
    .unwrap();
    let mut dw_solver = SgdSolver::<f64>::snap_t_fdp(
        &dwg,
        SnapTFdpConfig {
            iterations: 20,
            degree_beta: 0.4,
            bounded_attraction: true,
            seed: 13,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        dw_solver
            .snap_t_fdp
            .as_ref()
            .unwrap()
            .deg_scales
            .as_ref()
            .map(Vec::len),
        Some(4)
    );
    let mut dw_layout = circle_layout::<f64>(4, 1.0);
    dw_solver.run(&mut dw_layout).unwrap();
    assert!(dw_layout.is_all_finite());
}
