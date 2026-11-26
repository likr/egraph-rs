use egraph_core::{CsrGraph, IterativeSolver, Layout2d};
use egraph_linalg_mds::{PivotMdsConfig, pivot_mds};
use egraph_neighborhood::{DogVaradhanConfig, DogVaradhanLandmarks};

use crate::coarse_backbone::CoarseBackboneConfig;

use super::super::sgd::{
    BarrierSymmetry, PacmapGraphConfig, SgdScheduleConfig, SgdScheduleKind, SgdSolver,
    SparseBarrierConfig, StreamingNegativeConfig,
};
use super::pairs::PairStorage;
use super::test_support::{barbell_graph, circle_layout, compute_np2, grid_graph};

#[test]
fn test_sparse_barrier_from_landmarks_parity_and_star_safety() {
    let path =
        CsrGraph::<f64>::from_edges(5, &[(0, 1), (1, 2), (2, 3), (3, 4)], None, false, false)
            .unwrap();
    let landmarks = DogVaradhanLandmarks {
        num_nodes: 5,
        landmarks: vec![0],
        d_dog: vec![0.0, 1.0, 2.0, 3.0, 4.0],
        d_coarse: None,
    };
    let cfg = SparseBarrierConfig {
        max_2hop_per_node: 32,
        hinge_repulsion: true,
        delta_rel: 0.5,
        barrier_sym: BarrierSymmetry::SymMax,
        omega_1: 2.0,
        omega_2: 0.5,
        gamma_non2: 3.0,
        ht_repulsion: false,
        min_ratio: 0.5,
        eps_dist: 1e-6,
        neg_per_node: Some(8),
        num_pivots: 2,
        core_pivots: true,
        degree_weight_beta: 0.0,
        schedule: SgdScheduleConfig::default(),
        seed: 42,
    };
    let (solver, stats) =
        SgdSolver::<f64>::sparse_barrier_from_landmarks(&path, &landmarks, Some(&[0, 4]), &cfg)
            .unwrap();
    assert_eq!(stats.n_pairs_1hop, 4);
    assert_eq!(stats.n_pairs_2hop, 3);
    assert_eq!(stats.n_pairs_non2, 3);
    assert_eq!(stats.n_pairs_two_sided, 7);
    assert_eq!(stats.n_pairs_hinge, 3);
    assert!((stats.d_floor - 0.5).abs() < 1e-12);
    assert!((stats.gamma_eff - 3.0).abs() < 1e-12);
    assert_eq!(solver.num_springs(), 7);
    assert_eq!(solver.num_hinges(), 3);

    let mut cfg_ht = cfg;
    cfg_ht.ht_repulsion = true;
    let (_solver_ht, stats_ht) =
        SgdSolver::<f64>::sparse_barrier_from_landmarks(&path, &landmarks, Some(&[0, 4]), &cfg_ht)
            .unwrap();
    let expected_gamma = 3.0 * (5.0_f64 / 8.0_f64).sqrt();
    assert!((stats_ht.gamma_eff - expected_gamma).abs() < 1e-12);

    // Star graph on 1000 nodes: hub-safe cap keeps pairs O(N * cap).
    let n_star = 1000usize;
    let star_edges: Vec<(u32, u32)> = (1..n_star as u32).map(|v| (0, v)).collect();
    let star = CsrGraph::<f64>::from_edges(n_star, &star_edges, None, false, false).unwrap();
    let star_cfg = SparseBarrierConfig {
        max_2hop_per_node: 16,
        num_pivots: 16,
        ..Default::default()
    };
    let (star_solver, star_stats, _) =
        SgdSolver::<f64>::sparse_barrier_dog(&star, &DogVaradhanConfig::default(), &star_cfg)
            .unwrap();
    assert_eq!(star_stats.n_pairs_1hop, n_star - 1);
    assert!(star_solver.num_springs() <= (n_star - 1) + n_star * 16);
    assert_eq!(star_solver.num_hinges(), 0);
}

#[test]
fn test_sparse_barrier_dog_preserves_2hop_on_grid_and_barbell() {
    let grid = grid_graph(4, 4);
    let dog_cfg = DogVaradhanConfig::default();
    let barrier_cfg = SparseBarrierConfig::default();
    let (mut grid_solver, grid_stats, _) =
        SgdSolver::<f64>::sparse_barrier_dog(&grid, &dog_cfg, &barrier_cfg).unwrap();
    assert!(grid_stats.n_pairs_1hop > 0);
    assert!(grid_stats.n_pairs_2hop > 0);
    assert!(grid_stats.n_pairs_hinge > 0);

    let mut grid_layout = pivot_mds(&grid, &PivotMdsConfig::default()).unwrap();
    grid_solver.run(&mut grid_layout).unwrap();
    assert!(grid_layout.is_all_finite());
    let np2_grid = compute_np2(&grid, &grid_layout);
    assert!(
        np2_grid >= 0.85,
        "expected grid NP2 >= 0.85, got {np2_grid}"
    );

    let barbell = barbell_graph(6);
    let (mut bb_solver, _, _) =
        SgdSolver::<f64>::sparse_barrier_dog(&barbell, &dog_cfg, &barrier_cfg).unwrap();
    let mut bb_layout = pivot_mds(&barbell, &PivotMdsConfig::default()).unwrap();
    bb_solver.run(&mut bb_layout).unwrap();
    assert!(bb_layout.is_all_finite());
    let np2_bb = compute_np2(&barbell, &bb_layout);
    assert!(np2_bb >= 0.90, "expected barbell NP2 >= 0.90, got {np2_bb}");
}

#[test]
fn test_streaming_negatives_separates_non_edges_and_converges() {
    let barbell = barbell_graph(5);
    let neg_cfg = StreamingNegativeConfig {
        negatives_per_node: 6,
        target_distance: 2.5,
        weight: 0.5,
        hinge_only: true,
    };
    let sched = SgdScheduleConfig {
        t_max: 60,
        eps: 0.01,
        kind: SgdScheduleKind::Exponential,
        step_scale: 1.0,
    };
    let mut solver = SgdSolver::<f64>::streaming_negatives(&barbell, neg_cfg, sched, 42).unwrap();
    assert_eq!(solver.num_springs(), barbell.num_edges());
    assert_eq!(solver.num_hinges(), 0);

    let mut layout = circle_layout::<f64>(barbell.num_nodes(), 0.2);
    solver.run(&mut layout).unwrap();
    assert!(layout.is_all_finite());

    let c0x = (0..5).map(|i| layout.x()[i]).sum::<f64>() / 5.0;
    let c0y = (0..5).map(|i| layout.y()[i]).sum::<f64>() / 5.0;
    let c1x = (5..10).map(|i| layout.x()[i]).sum::<f64>() / 5.0;
    let c1y = (5..10).map(|i| layout.y()[i]).sum::<f64>() / 5.0;
    let sep = (c0x - c1x).hypot(c0y - c1y);
    assert!(sep > 1.5, "expected separated clique centroids, got {sep}");

    let disc = CsrGraph::<f64>::from_edges(2, &[], None, false, false).unwrap();
    let neg_spring = StreamingNegativeConfig {
        negatives_per_node: 3,
        target_distance: 3.0,
        weight: 1.0,
        hinge_only: false,
    };
    let mut disc_solver =
        SgdSolver::<f64>::streaming_negatives(&disc, neg_spring, sched, 7).unwrap();
    let mut disc_layout = Layout2d::from_xy(vec![0.0, 0.1], vec![0.0, 0.0]).unwrap();
    disc_solver.run(&mut disc_layout).unwrap();
    let d_disc =
        (disc_layout.x()[0] - disc_layout.x()[1]).hypot(disc_layout.y()[0] - disc_layout.y()[1]);
    assert!(
        (d_disc - 3.0).abs() < 0.25,
        "expected dist ~3.0, got {d_disc}"
    );
}

#[test]
fn test_streaming_negatives_unsorted_csr_never_repels_1hop_edges() {
    // Triangle 0-1-2 plus leaves (0, 3) and (2, 4) with deliberately unsorted CSR rows.
    let indptr = vec![0u32, 3, 5, 8, 9, 10];
    let indices = vec![
        3, 2, 1, // node 0 (unsorted)
        2, 0, // node 1 (unsorted)
        4, 1, 0, // node 2 (unsorted)
        0, // node 3
        2, // node 4
    ];
    let g_unsorted = CsrGraph::<f64>::from_csr(&indptr, &indices, None, false, false).unwrap();
    let g_sorted = CsrGraph::<f64>::from_csr(&indptr, &indices, None, false, true).unwrap();

    let neg_cfg = StreamingNegativeConfig {
        negatives_per_node: 8,
        target_distance: 3.0,
        weight: 0.25,
        hinge_only: true,
    };
    let sched = SgdScheduleConfig {
        t_max: 60,
        eps: 0.01,
        kind: SgdScheduleKind::Exponential,
        step_scale: 1.0,
    };

    let mut s_unsorted =
        SgdSolver::<f64>::streaming_negatives(&g_unsorted, neg_cfg, sched, 99).unwrap();
    let mut s_sorted =
        SgdSolver::<f64>::streaming_negatives(&g_sorted, neg_cfg, sched, 99).unwrap();

    let st_u = s_unsorted.streaming.as_ref().unwrap();
    let st_s = s_sorted.streaming.as_ref().unwrap();
    assert_eq!(st_u.adj_indptr, st_s.adj_indptr);
    assert_eq!(st_u.adj_indices, st_s.adj_indices);
    for u in 0..5 {
        let row = &st_u.adj_indices[st_u.adj_indptr[u] as usize..st_u.adj_indptr[u + 1] as usize];
        assert!(row.windows(2).all(|w| w[0] < w[1]));
    }

    let mut l_unsorted = circle_layout::<f64>(5, 0.8);
    let mut l_sorted = l_unsorted.clone();
    s_unsorted.run(&mut l_unsorted).unwrap();
    s_sorted.run(&mut l_sorted).unwrap();

    // Unsorted CSR must match sorted CSR bit-for-bit and keep 1-hop edges compact while
    // separating non-edge pair (3, 4).
    for (a, b) in l_unsorted.x().iter().zip(l_sorted.x().iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    for (a, b) in l_unsorted.y().iter().zip(l_sorted.y().iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    for (u, v) in [(0usize, 1usize), (1, 2), (0, 2), (0, 3), (2, 4)] {
        let d_uv =
            (l_unsorted.x()[u] - l_unsorted.x()[v]).hypot(l_unsorted.y()[u] - l_unsorted.y()[v]);
        assert!(
            d_uv < 2.0,
            "1-hop edge ({u}, {v}) suffered excessive repulsion: dist={d_uv}"
        );
    }
    let d_34 = (l_unsorted.x()[3] - l_unsorted.x()[4]).hypot(l_unsorted.y()[3] - l_unsorted.y()[4]);
    assert!(d_34 > 2.0, "expected non-edge (3, 4) separated, got {d_34}");
}

#[test]
fn test_sparse_barrier_degree_weight_beta_and_pacmap_backbone_fem_grid() {
    // 1. Star graph degree_weight_beta = 0.5 reduces spring weight on high-degree hub by (d_u * d_v)^{-0.5}.
    let n_star = 17usize;
    let star_edges: Vec<(u32, u32)> = (1..n_star as u32).map(|v| (0, v)).collect();
    let star = CsrGraph::<f64>::from_edges(n_star, &star_edges, None, false, false).unwrap();
    let lm = DogVaradhanLandmarks {
        num_nodes: n_star,
        landmarks: vec![0, 1],
        d_dog: (0..n_star)
            .flat_map(|i| if i == 0 { [0.0, 1.0] } else { [1.0, 2.0] })
            .collect(),
        d_coarse: None,
    };
    let cfg0 = SparseBarrierConfig {
        omega_1: 1.0,
        omega_2: 0.0,
        gamma_non2: 0.0,
        degree_weight_beta: 0.0,
        neg_per_node: Some(0),
        ..SparseBarrierConfig::default()
    };
    let cfg_beta = SparseBarrierConfig {
        degree_weight_beta: 0.5,
        ..cfg0
    };
    let (s0, _) =
        SgdSolver::<f64>::sparse_barrier_from_landmarks(&star, &lm, Some(&[0, 1]), &cfg0).unwrap();
    let (sb, _) =
        SgdSolver::<f64>::sparse_barrier_from_landmarks(&star, &lm, Some(&[0, 1]), &cfg_beta)
            .unwrap();
    let w0 = match &s0.pairs {
        PairStorage::Symmetric(p) => p[0].w,
        _ => panic!("expected symmetric"),
    };
    let wb = match &sb.pairs {
        PairStorage::Symmetric(p) => p[0].w,
        _ => panic!("expected symmetric"),
    };
    let expected_ratio = (16.0_f64 * 1.0_f64).powf(-0.5);
    assert!(
        ((wb / w0) - expected_ratio).abs() < 1e-12,
        "expected weight ratio {expected_ratio}, got {}",
        wb / w0
    );
    let bad_beta = SparseBarrierConfig {
        degree_weight_beta: -0.1,
        ..cfg0
    };
    assert!(
        SgdSolver::<f64>::sparse_barrier_from_landmarks(&star, &lm, Some(&[0, 1]), &bad_beta)
            .is_err()
    );

    // 2. FEM-like 8x8 mesh grid: pacmap_graph_with_backbone unfolds deterministically and preserves NP2.
    let fem = grid_graph(8, 8);
    let pm_cfg = PacmapGraphConfig {
        n_neighbors: 8,
        mn_ratio: 0.5,
        fp_ratio: 2.0,
        phase1_iters: 40,
        phase2_iters: 40,
        phase3_iters: 60,
        num_landmarks: 24,
        seed: 42,
        ..PacmapGraphConfig::default()
    };
    let mut bb_cfg = CoarseBackboneConfig {
        target_std: Some(5.0),
        coarse_t_max: 40,
        ..CoarseBackboneConfig::default()
    };
    bb_cfg.louvain.max_coarse_nodes = 16;

    let (mut s1, mut l1, bb1) =
        SgdSolver::<f64>::pacmap_graph_with_backbone(&fem, pm_cfg, &bb_cfg).unwrap();
    let (mut s2, mut l2, _) =
        SgdSolver::<f64>::pacmap_graph_with_backbone(&fem, pm_cfg, &bb_cfg).unwrap();
    assert!(bb1.stats.num_coarse_nodes <= 16);
    s1.run_steps(&mut l1, 20).unwrap();
    s1.run(&mut l1).unwrap();
    s2.run(&mut l2).unwrap();
    assert_eq!(l1, l2);
    let np2 = compute_np2(&fem, &l1);
    assert!(
        np2 > 0.70,
        "expected unfolded FEM grid NP2 > 0.70, got {np2}"
    );

    // Invalid num_landmarks = 0 rejected.
    let bad_pm = PacmapGraphConfig {
        num_landmarks: 0,
        ..pm_cfg
    };
    assert!(SgdSolver::<f64>::pacmap_graph(&fem, bad_pm).is_err());
}
