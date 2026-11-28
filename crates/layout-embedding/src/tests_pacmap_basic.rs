use egraph_core::{
    CoreError, CsrGraph, IterativeSolver, Layout2d, Layout3d, LayoutGenerator, LayoutNd,
    ManifoldLayout2d, SolverStatus,
};
use egraph_linalg_mds::PivotMds;
use egraph_neighborhood::extract_1hop_2hop_csr;

use crate::test_support::{
    ALL_MANIFOLDS, assert_manifold_step_eq_run, assert_riemannian_step_eq_run, barbell_graph,
    circle_layout, compute_np2, grid_graph, layout_2d_to_nd,
};
use crate::{PacmapGraphConfig, PacmapGraphSolver};

#[test]
fn test_pacmap_graph_three_tier_three_phase_and_bit_exact() {
    let grid = grid_graph(4, 4);
    let cfg = PacmapGraphConfig::new()
        .with_n_neighbors(6)
        .with_mn_ratio(0.5)
        .with_fp_ratio(2.0)
        .with_phase_iterations([30, 30, 60])
        .with_learning_rate(0.5)
        .with_seed(42)
        .with_keep_all_1hop(false);

    // Check 3-tier pair invariants on grid(4, 4).
    let s_check = PacmapGraphSolver::<Layout2d<f64>>::new(&grid, cfg.clone()).unwrap();
    assert!(!s_check.nb_pairs().is_empty());
    assert!(!s_check.mn_pairs().is_empty());
    assert!(!s_check.fp_pairs().is_empty());
    assert_eq!(s_check.total_iterations(), Some(120));

    let (adj1, adj2) = extract_1hop_2hop_csr(&grid).unwrap();
    for &(u, v) in s_check.nb_pairs() {
        assert_ne!(u, v);
        let ui = u as usize;
        assert!(
            adj1.neighbors(ui).contains(&v) || adj2.neighbors(ui).contains(&v),
            "NB pair ({u}, {v}) must be 1-hop or 2-hop"
        );
    }
    for &(u, v) in s_check.mn_pairs() {
        assert_ne!(u, v);
        let ui = u as usize;
        assert!(!adj1.neighbors(ui).contains(&v));
        assert!(!adj2.neighbors(ui).contains(&v));
    }
    let mut fp_per_u: Vec<Vec<u32>> = vec![Vec::new(); grid.num_nodes()];
    for &(u, v) in s_check.fp_pairs() {
        assert_ne!(u, v);
        let ui = u as usize;
        assert!(!adj1.neighbors(ui).contains(&v));
        fp_per_u[ui].push(v);
    }
    for (u, row) in fp_per_u.iter_mut().enumerate() {
        let expected = (16 - 1 - (adj1.degree(u) + adj2.degree(u)).min(6)).min(12);
        assert_eq!(row.len(), expected);
        row.sort_unstable();
        row.dedup();
        assert_eq!(row.len(), expected);
    }

    // On grid(8, 8) (n = 64, max_deg = 4 <= 16), sufficient non-2-hop vertices exist so
    // FP strictly excludes both 1-hop and 2-hop neighbors.
    let grid8 = grid_graph(8, 8);
    let s8 = PacmapGraphSolver::<Layout2d<f64>>::new(&grid8, cfg.clone()).unwrap();
    let (adj1_8, adj2_8) = extract_1hop_2hop_csr(&grid8).unwrap();
    for &(u, v) in s8.fp_pairs() {
        assert_ne!(u, v);
        let ui = u as usize;
        assert!(!adj1_8.neighbors(ui).contains(&v));
        assert!(
            !adj2_8.neighbors(ui).contains(&v),
            "grid(8, 8) FP pair ({u}, {v}) must not be 2-hop"
        );
    }

    // Bit-exact step() == run() for f64 and f32 across all 3 phases.
    let mut s64_run = PacmapGraphSolver::<Layout2d<f64>>::new(&grid, cfg.clone()).unwrap();
    let mut s64_step = PacmapGraphSolver::<Layout2d<f64>>::new(&grid, cfg.clone()).unwrap();
    let mut l64_run: Layout2d<f64> = PivotMds::new(&grid).generate().unwrap();
    let mut l64_step = l64_run.clone();
    s64_run.run(&mut l64_run).unwrap();
    s64_step.run_steps(&mut l64_step, 35).unwrap(); // across phase 1 -> 2
    s64_step.run_steps(&mut l64_step, 35).unwrap(); // across phase 2 -> 3
    while !s64_step.status().is_finished() {
        s64_step.step(&mut l64_step).unwrap();
    }
    assert_eq!(s64_step.iteration(), 120);
    for i in 0..grid.num_nodes() {
        assert_eq!(l64_run.x()[i].to_bits(), l64_step.x()[i].to_bits());
        assert_eq!(l64_run.y()[i].to_bits(), l64_step.y()[i].to_bits());
    }

    let mut s32_run = PacmapGraphSolver::<Layout2d<f32>>::new(&grid, cfg.clone()).unwrap();
    let mut s32_step = PacmapGraphSolver::<Layout2d<f32>>::new(&grid, cfg.clone()).unwrap();
    let mut l32_run = circle_layout::<f32>(grid.num_nodes(), 1.0);
    let mut l32_step = l32_run.clone();
    s32_run.run(&mut l32_run).unwrap();
    s32_step.run_steps(&mut l32_step, 45).unwrap();
    s32_step.run(&mut l32_step).unwrap();
    for i in 0..grid.num_nodes() {
        assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
        assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
    }

    // Layout3d & LayoutNd (2D matching Layout2d, plus 3D and 5D) parity.
    let init2 = circle_layout::<f64>(grid.num_nodes(), 1.0);
    let out_2d = assert_riemannian_step_eq_run(
        PacmapGraphSolver::<Layout2d<f64>>::new(&grid, cfg.clone()).unwrap(),
        PacmapGraphSolver::<Layout2d<f64>>::new(&grid, cfg.clone()).unwrap(),
        &init2,
        40,
        "pacmap_2d",
    );
    let out_nd2 = assert_riemannian_step_eq_run(
        PacmapGraphSolver::<LayoutNd<f64>>::new(&grid, cfg.clone()).unwrap(),
        PacmapGraphSolver::<LayoutNd<f64>>::new(&grid, cfg.clone()).unwrap(),
        &layout_2d_to_nd(&init2),
        40,
        "pacmap_nd2",
    );
    for i in 0..grid.num_nodes() {
        assert_eq!(out_2d.x()[i].to_bits(), out_nd2.coords()[i * 2].to_bits());
        assert_eq!(
            out_2d.y()[i].to_bits(),
            out_nd2.coords()[i * 2 + 1].to_bits()
        );
    }

    let init3 = Layout3d::<f64>::init(grid.num_nodes(), 7, 1.0).unwrap();
    assert_riemannian_step_eq_run(
        PacmapGraphSolver::<Layout3d<f64>>::new(&grid, cfg.clone()).unwrap(),
        PacmapGraphSolver::<Layout3d<f64>>::new(&grid, cfg.clone()).unwrap(),
        &init3,
        40,
        "pacmap_3d",
    );
    let init5 = LayoutNd::<f32>::init(grid.num_nodes(), 5, 7, 1.0).unwrap();
    assert_riemannian_step_eq_run(
        PacmapGraphSolver::<LayoutNd<f32>>::new(&grid, cfg.clone()).unwrap(),
        PacmapGraphSolver::<LayoutNd<f32>>::new(&grid, cfg.clone()).unwrap(),
        &init5,
        40,
        "pacmap_5d_f32",
    );

    // Barbell separation & NP2 quality with default PaCMAP-Graph parameters on a 20-node barbell.
    let barbell20 = barbell_graph(10);
    let mut bb_solver =
        PacmapGraphSolver::<Layout2d<f64>>::new(&barbell20, PacmapGraphConfig::default()).unwrap();
    let mut bb_layout = circle_layout::<f64>(barbell20.num_nodes(), 0.25);
    bb_solver.run(&mut bb_layout).unwrap();
    assert!(bb_layout.is_all_finite());
    let c0x = bb_layout.x()[..10].iter().sum::<f64>() / 10.0;
    let c0y = bb_layout.y()[..10].iter().sum::<f64>() / 10.0;
    let c1x = bb_layout.x()[10..].iter().sum::<f64>() / 10.0;
    let c1y = bb_layout.y()[10..].iter().sum::<f64>() / 10.0;
    let sep = (c0x - c1x).hypot(c0y - c1y);
    assert!(sep > 1.0, "expected separated barbell cliques, got {sep}");
    assert!(compute_np2(&barbell20, &bb_layout) >= 0.85);

    // Weighted and directed graph paths work cleanly.
    let dwg = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
        Some(&[1.0, 2.0, 1.5, 1.0, 2.5, 0.8]),
        true,
        false,
    )
    .unwrap();
    let mut dw_pm = PacmapGraphSolver::<Layout2d<f64>>::new(&dwg, cfg.clone()).unwrap();
    let mut dw_l = circle_layout::<f64>(6, 1.0);
    dw_pm.run(&mut dw_l).unwrap();
    assert!(dw_l.is_all_finite());

    // All 4 manifolds bit-exact f64 & f32.
    let m_cfg = PacmapGraphConfig::new()
        .with_phase_iterations([4, 4, 4])
        .with_num_landmarks(6)
        .with_seed(42);
    for kind in ALL_MANIFOLDS {
        assert_manifold_step_eq_run(
            PacmapGraphSolver::<ManifoldLayout2d<f64>>::new(&grid, m_cfg.clone()).unwrap(),
            PacmapGraphSolver::<ManifoldLayout2d<f64>>::new(&grid, m_cfg.clone()).unwrap(),
            kind,
            grid.num_nodes(),
            "pacmap_manifold_f64",
        );
        assert_manifold_step_eq_run(
            PacmapGraphSolver::<ManifoldLayout2d<f32>>::new(&grid, m_cfg.clone()).unwrap(),
            PacmapGraphSolver::<ManifoldLayout2d<f32>>::new(&grid, m_cfg.clone()).unwrap(),
            kind,
            grid.num_nodes(),
            "pacmap_manifold_f32",
        );
    }

    // Edge cases, run_steps(0) validation, and invalid configs.
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    assert_eq!(
        PacmapGraphSolver::<Layout2d<f64>>::new(&empty, cfg.clone())
            .unwrap()
            .run(&mut Layout2d::zeros(0))
            .unwrap(),
        SolverStatus::Completed
    );
    assert_eq!(
        PacmapGraphSolver::<Layout2d<f64>>::new(&single, cfg.clone())
            .unwrap()
            .run(&mut Layout2d::zeros(1))
            .unwrap(),
        SolverStatus::Completed
    );
    assert_eq!(
        PacmapGraphSolver::<Layout2d<f64>>::new(
            &grid,
            cfg.clone().with_phase_iterations([0, 0, 0])
        )
        .unwrap()
        .status(),
        SolverStatus::Completed
    );
    let mut zero_step_solver = PacmapGraphSolver::<Layout2d<f64>>::new(&grid, cfg.clone()).unwrap();
    let mut wrong_len = Layout2d::<f64>::zeros(grid.num_nodes() + 1);
    assert!(matches!(
        zero_step_solver.run_steps(&mut wrong_len, 0),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert_eq!(zero_step_solver.status(), SolverStatus::Initialized);

    for bad in [
        cfg.clone().with_n_neighbors(0),
        cfg.clone().with_mn_ratio(-0.1),
        cfg.clone().with_fp_ratio(f64::NAN),
        cfg.clone().with_learning_rate(0.0),
        cfg.clone().with_max_1hop_per_node(Some(0)),
        cfg.clone().with_degree_weight_beta(-0.1),
        cfg.clone().with_degree_weight_beta(f64::NAN),
        cfg.with_num_landmarks(0),
    ] {
        assert!(PacmapGraphSolver::<Layout2d<f64>>::new(&grid, bad).is_err());
    }
}

#[test]
fn test_pacmap_graph_keep_all_1hop_and_fp_2hop_exclusion() {
    let mut edges: Vec<(u32, u32)> = (1..=20).map(|v| (0, v)).collect();
    edges.extend([(20, 21), (21, 22), (22, 23), (23, 24), (24, 25)]);
    let n = 26usize;
    let unweighted = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();
    let weights: Vec<f64> = (0..edges.len()).map(|i| 1.0 + 0.1 * (i as f64)).collect();
    let weighted = CsrGraph::<f64>::from_edges(n, &edges, Some(&weights), false, false).unwrap();

    let (adj1, adj2) = extract_1hop_2hop_csr(&unweighted).unwrap();
    let cfg_trunc = PacmapGraphConfig::new()
        .with_n_neighbors(6)
        .with_mn_ratio(0.5)
        .with_fp_ratio(2.0)
        .with_phase_iterations([5, 5, 5])
        .with_seed(17)
        .with_keep_all_1hop(false);
    let cfg_keep = cfg_trunc.clone().with_keep_all_1hop(true);

    for g in [&unweighted, &weighted] {
        let s_trunc = PacmapGraphSolver::<Layout2d<f64>>::new(g, cfg_trunc.clone()).unwrap();
        for w in s_trunc.nb_pairs().windows(2) {
            assert!(w[0] < w[1], "duplicate or unsorted nb_pair: {w:?}");
        }
        assert!(s_trunc.nb_pairs().iter().all(|&(u, v)| u < v));
        let hub_inc_trunc = s_trunc
            .nb_pairs()
            .iter()
            .filter(|&&(u, v)| u == 0 || v == 0)
            .count();
        assert!(hub_inc_trunc >= 20);

        let s_keep = PacmapGraphSolver::<Layout2d<f64>>::new(g, cfg_keep.clone()).unwrap();
        for w in s_keep.nb_pairs().windows(2) {
            assert!(w[0] < w[1], "duplicate or unsorted nb_pair: {w:?}");
        }
        let hub_nb_keep: Vec<u32> = s_keep
            .nb_pairs()
            .iter()
            .filter_map(|&(u, v)| {
                if u == 0 {
                    Some(v)
                } else if v == 0 {
                    Some(u)
                } else {
                    None
                }
            })
            .collect();
        assert!(hub_nb_keep.len() >= 20);
        for v in 1..=20u32 {
            assert!(hub_nb_keep.contains(&v));
        }
        let leaf_nb_keep = s_keep
            .nb_pairs()
            .iter()
            .filter(|&&(u, v)| u == 1 || v == 1)
            .count();
        assert!(leaf_nb_keep >= 6);

        for &(u, v) in s_trunc.fp_pairs() {
            assert_ne!(u, v);
            let ui = u as usize;
            assert!(!adj1.neighbors(ui).contains(&v));
            if ui >= 22 {
                assert!(!adj2.neighbors(ui).contains(&v));
            }
        }
        for &(u, v) in s_keep.fp_pairs() {
            assert_ne!(u, v);
            let ui = u as usize;
            assert!(!adj1.neighbors(ui).contains(&v));
            if ui >= 22 {
                assert!(!adj2.neighbors(ui).contains(&v));
            }
        }
    }

    let mut nh_edges: Vec<(u32, u32)> = (1..=10u32).map(|v| (0, v)).collect();
    for v in 10..69u32 {
        nh_edges.push((v, v + 1));
    }
    let g_nh = CsrGraph::<f64>::from_edges(70, &nh_edges, None, false, false).unwrap();
    let (nh_adj1, nh_adj2) = extract_1hop_2hop_csr(&g_nh).unwrap();
    let nh_cfg_trunc = PacmapGraphConfig::new()
        .with_n_neighbors(4)
        .with_mn_ratio(0.5)
        .with_fp_ratio(2.0)
        .with_phase_iterations([2, 2, 2])
        .with_seed(17)
        .with_keep_all_1hop(false);
    let nh_cfg_keep = nh_cfg_trunc.clone().with_keep_all_1hop(true);
    for nh_cfg in [nh_cfg_trunc, nh_cfg_keep] {
        let s_nh = PacmapGraphSolver::<Layout2d<f64>>::new(&g_nh, nh_cfg).unwrap();
        assert!(!s_nh.fp_pairs().is_empty());
        for &(u, v) in s_nh.fp_pairs() {
            assert_ne!(u, v);
            let ui = u as usize;
            assert!(!nh_adj1.neighbors(ui).contains(&v));
            assert!(
                !nh_adj2.neighbors(ui).contains(&v),
                "non-hub FP pair ({u}, {v}) must not be 2-hop"
            );
        }
    }
}
