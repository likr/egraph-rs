use egraph_core::{CsrGraph, IterativeSolver, Layout2d, LayoutGenerator};
use egraph_linalg_mds::PivotMds;
use egraph_neighborhood::{DogVaradhanConfig, DogVaradhanLandmarks};

use super::super::sgd::{
    BarrierSymmetry, SgdScheduleConfig, SgdScheduleKind, SgdSolver, SparseBarrierConfig,
    StreamingNegativeConfig,
};
use super::pairs::PairStorage;
use super::test_support::{barbell_graph, circle_layout, compute_np2, grid_graph};

#[test]
fn test_sparse_barrier_from_landmarks_parity_and_star_safety() {
    let path =
        CsrGraph::<f64>::from_edges(5, &[(0, 1), (1, 2), (2, 3), (3, 4)], None, false, false)
            .unwrap();
    let landmarks =
        DogVaradhanLandmarks::new_validated(5, vec![0], vec![0.0, 1.0, 2.0, 3.0, 4.0], None)
            .unwrap();
    let cfg = SparseBarrierConfig::default()
        .with_max_2hop_per_node(32)
        .with_hinge_repulsion(true)
        .with_delta_rel(0.5)
        .with_barrier_sym(BarrierSymmetry::SymMax)
        .with_omega_1(2.0)
        .with_omega_2(0.5)
        .with_gamma_non2(3.0)
        .with_ht_repulsion(false)
        .with_min_ratio(0.5)
        .with_eps_dist(1e-6)
        .with_neg_per_node(Some(8))
        .with_num_pivots(2)
        .with_core_pivots(true)
        .with_degree_weight_beta(0.0)
        .with_schedule(SgdScheduleConfig::default())
        .with_seed(42);

    let (solver, stats) = SgdSolver::<Layout2d<f64>>::sparse_barrier_from_landmarks(
        &path,
        &landmarks,
        Some(&[0, 4]),
        &cfg,
    )
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

    let cfg_ht = cfg.clone().with_ht_repulsion(true);
    let (_solver_ht, stats_ht) = SgdSolver::<Layout2d<f64>>::sparse_barrier_from_landmarks(
        &path,
        &landmarks,
        Some(&[0, 4]),
        &cfg_ht,
    )
    .unwrap();
    let expected_gamma = 3.0 * (5.0_f64 / 8.0_f64).sqrt();
    assert!((stats_ht.gamma_eff - expected_gamma).abs() < 1e-12);

    // Also verifies unweighted CsrGraph<()> works directly without W: Scalar.
    let unweighted =
        CsrGraph::<()>::from_unweighted_edges(5, &[(0, 1), (1, 2), (2, 3), (3, 4)], false, false)
            .unwrap();
    let (u_solver, u_stats) = SgdSolver::<Layout2d<f64>>::sparse_barrier_from_landmarks(
        &unweighted,
        &landmarks,
        Some(&[0, 4]),
        &cfg,
    )
    .unwrap();
    assert_eq!(u_stats.n_pairs_1hop, 4);
    assert_eq!(u_solver.num_springs(), 7);

    // Star graph on 1000 nodes: hub-safe cap keeps pairs O(N * cap).
    let n_star = 1000usize;
    let star_edges: Vec<(u32, u32)> = (1..n_star as u32).map(|v| (0, v)).collect();
    let star = CsrGraph::<f64>::from_edges(n_star, &star_edges, None, false, false).unwrap();
    let star_cfg = SparseBarrierConfig::default()
        .with_max_2hop_per_node(16)
        .with_num_pivots(16);
    let (star_solver, star_stats, _) = SgdSolver::<Layout2d<f64>>::sparse_barrier_dog(
        &star,
        &DogVaradhanConfig::default(),
        &star_cfg,
    )
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
        SgdSolver::<Layout2d<f64>>::sparse_barrier_dog(&grid, &dog_cfg, &barrier_cfg).unwrap();
    assert!(grid_stats.n_pairs_1hop > 0);
    assert!(grid_stats.n_pairs_2hop > 0);
    assert!(grid_stats.n_pairs_hinge > 0);

    let mut grid_layout: Layout2d<f64> = PivotMds::new(&grid).generate().unwrap();
    grid_solver.run(&mut grid_layout).unwrap();
    assert!(grid_layout.is_all_finite());
    let np2_grid = compute_np2(&grid, &grid_layout);
    assert!(
        np2_grid >= 0.85,
        "expected grid NP2 >= 0.85, got {np2_grid}"
    );

    let barbell = barbell_graph(6);
    let (mut bb_solver, _, _) =
        SgdSolver::<Layout2d<f64>>::sparse_barrier_dog(&barbell, &dog_cfg, &barrier_cfg).unwrap();
    let mut bb_layout: Layout2d<f64> = PivotMds::new(&barbell).generate().unwrap();
    bb_solver.run(&mut bb_layout).unwrap();
    assert!(bb_layout.is_all_finite());
    let np2_bb = compute_np2(&barbell, &bb_layout);
    assert!(np2_bb >= 0.90, "expected barbell NP2 >= 0.90, got {np2_bb}");
}

#[test]
fn test_streaming_negatives_separates_non_edges_and_converges() {
    let barbell = barbell_graph(5);
    let neg_cfg = StreamingNegativeConfig::default()
        .with_negatives_per_node(6)
        .with_target_distance(2.5)
        .with_weight(0.5)
        .with_hinge_only(true);
    let sched = SgdScheduleConfig::new(60)
        .with_eps(0.01)
        .with_kind(SgdScheduleKind::Exponential)
        .with_step_scale(1.0);
    let mut solver =
        SgdSolver::<Layout2d<f64>>::streaming_negatives(&barbell, neg_cfg, sched.clone(), 42)
            .unwrap();
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
    let neg_spring = StreamingNegativeConfig::default()
        .with_negatives_per_node(3)
        .with_target_distance(3.0)
        .with_weight(1.0)
        .with_hinge_only(false);
    let mut disc_solver =
        SgdSolver::<Layout2d<f64>>::streaming_negatives(&disc, neg_spring, sched, 7).unwrap();
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
    let indptr = vec![0u32, 3, 5, 8, 9, 10];
    let indices = vec![3, 2, 1, 2, 0, 4, 1, 0, 0, 2];
    let g_unsorted = CsrGraph::<f64>::from_csr(&indptr, &indices, None, false, false).unwrap();
    let g_sorted = CsrGraph::<f64>::from_csr(&indptr, &indices, None, false, true).unwrap();

    let neg_cfg = StreamingNegativeConfig::default()
        .with_negatives_per_node(8)
        .with_target_distance(3.0)
        .with_weight(0.25)
        .with_hinge_only(true);
    let sched = SgdScheduleConfig::new(60)
        .with_eps(0.01)
        .with_kind(SgdScheduleKind::Exponential)
        .with_step_scale(1.0);

    let mut s_unsorted = SgdSolver::<Layout2d<f64>>::streaming_negatives(
        &g_unsorted,
        neg_cfg.clone(),
        sched.clone(),
        99,
    )
    .unwrap();
    let mut s_sorted =
        SgdSolver::<Layout2d<f64>>::streaming_negatives(&g_sorted, neg_cfg, sched, 99).unwrap();

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
fn test_sparse_barrier_degree_weight_beta_and_guards() {
    let n_star = 17usize;
    let star_edges: Vec<(u32, u32)> = (1..n_star as u32).map(|v| (0, v)).collect();
    let star = CsrGraph::<f64>::from_edges(n_star, &star_edges, None, false, false).unwrap();
    let d_dog: Vec<f64> = (0..n_star)
        .flat_map(|i| if i == 0 { [0.0, 1.0] } else { [1.0, 2.0] })
        .collect();
    let lm = DogVaradhanLandmarks::new_validated(n_star, vec![0, 1], d_dog, None).unwrap();

    let cfg0 = SparseBarrierConfig::default()
        .with_omega_1(1.0)
        .with_omega_2(0.0)
        .with_gamma_non2(0.0)
        .with_degree_weight_beta(0.0)
        .with_neg_per_node(Some(0));
    let cfg_beta = cfg0.clone().with_degree_weight_beta(0.5);
    let (s0, _) =
        SgdSolver::<Layout2d<f64>>::sparse_barrier_from_landmarks(&star, &lm, Some(&[0, 1]), &cfg0)
            .unwrap();
    let (sb, _) = SgdSolver::<Layout2d<f64>>::sparse_barrier_from_landmarks(
        &star,
        &lm,
        Some(&[0, 1]),
        &cfg_beta,
    )
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
    let bad_beta = cfg0.with_degree_weight_beta(-0.1);
    assert!(
        SgdSolver::<Layout2d<f64>>::sparse_barrier_from_landmarks(
            &star,
            &lm,
            Some(&[0, 1]),
            &bad_beta
        )
        .is_err()
    );
}
