use super::*;
use ndarray::Array2;
use petgraph::graph::UnGraph;
use petgraph_algorithm_shortest_path::all_sources_dijkstra;
use petgraph_linalg_kernel::{
    PivotedBandPassDiffusionKernelBuilder, PivotedDiffusionKernelBuilder,
    PivotedNegLogSimDistanceBuilder, SparseSymmetricMatrix,
};
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashSet;

fn cycle_graph(n: usize) -> UnGraph<(), ()> {
    let mut g = UnGraph::new_undirected();
    let nodes: Vec<_> = (0..n).map(|_| g.add_node(())).collect();
    for i in 0..n {
        g.add_edge(nodes[i], nodes[(i + 1) % n], ());
    }
    g
}

fn star_graph(n: usize) -> UnGraph<(), ()> {
    let mut g = UnGraph::new_undirected();
    let nodes: Vec<_> = (0..n).map(|_| g.add_node(())).collect();
    for i in 1..n {
        g.add_edge(nodes[0], nodes[i], ());
    }
    g
}

#[test]
fn test_sparse_sgd_with_pivoted_neg_log_sim_distance() {
    let g = cycle_graph(8);
    let lap = SparseSymmetricMatrix::symmetric_normalized_laplacian(&g, |_| 1.0_f64);
    let pivots = vec![0usize, 4usize];

    let mut rng = StdRng::seed_from_u64(42);
    let bp_kernel = PivotedBandPassDiffusionKernelBuilder::dog(
        &lap,
        0.5_f64,
        4.0_f64,
        0.5_f64,
        0.05_f64,
        20,
        pivots.clone(),
    )
    .num_hutchinson(Some(16))
    .build(&mut rng)
    .unwrap();
    let bp_dist = PivotedNegLogSimDistanceBuilder::new()
        .build(&g, bp_kernel)
        .unwrap();
    let bp_sgd = SparseSgd::new()
        .h(2)
        .build_with_pivoted_neg_log_sim_distance(&g, |_| 1.0_f64, &bp_dist);
    assert!(!bp_sgd.node_pairs().is_empty());
    for &(_, _, d1, d2, w1, w2) in bp_sgd.node_pairs() {
        assert!(d1.is_finite() && d2.is_finite());
        assert!(w1.is_finite() && w2.is_finite());
        assert!(d1 > 0.0 && d2 > 0.0);
    }

    let diff_kernel = PivotedDiffusionKernelBuilder::new(&lap, 2.0_f64, 20, pivots)
        .num_hutchinson(Some(16))
        .build(&mut rng)
        .unwrap();
    let diff_dist = PivotedNegLogSimDistanceBuilder::new()
        .build(&g, diff_kernel)
        .unwrap();
    let diff_sgd = SparseSgd::new()
        .h(2)
        .build_with_pivoted_neg_log_sim_distance(&g, |_| 1.0_f64, &diff_dist);
    assert!(!diff_sgd.node_pairs().is_empty());
    for &(_, _, d1, d2, w1, w2) in diff_sgd.node_pairs() {
        assert!(d1.is_finite() && d2.is_finite());
        assert!(w1.is_finite() && w2.is_finite());
    }
}

#[test]
fn test_sparse_sgd_legacy_empty_and_zero_h_guards() {
    let empty = UnGraph::<(), ()>::new_undirected();
    let mut rng = StdRng::seed_from_u64(42);
    let sgd_empty = SparseSgd::new().h(10).build(&empty, |_| 1.0_f64, &mut rng);
    assert_eq!(sgd_empty.total_pair_count(), 0);

    let g = cycle_graph(6);
    let sgd_zero_h = SparseSgd::new().h(0).build(&g, |_| 1.0_f64, &mut rng);
    assert_eq!(sgd_zero_h.total_pair_count(), 0);

    // Legacy mode emits no hinge pairs and matches standard Dijkstra pivot structure
    let sgd_legacy = SparseSgd::new().h(3).build(&g, |_| 1.0_f64, &mut rng);
    assert!(!sgd_legacy.node_pairs().is_empty());
    assert!(sgd_legacy.hinge_pairs().is_empty());
}

#[test]
fn test_sparse_sgd_barrier_1hop_calibration_2hop_and_hinge_on_cycle() {
    let g = cycle_graph(12);
    let dm = all_sources_dijkstra(&g, |_| 3.0_f64);
    let nodes: Vec<_> = g.node_indices().collect();
    let pivots = vec![nodes[0], nodes[4], nodes[8]];

    let sgd = SparseSgd::barrier()
        .h(3)
        .max_2hop_per_node(4)
        .delta_rel(0.25)
        .build_with_pivot_and_distance_matrix(&g, |_| 1.0_f64, &pivots, &dm);

    assert!(!sgd.node_pairs().is_empty());
    assert!(!sgd.hinge_pairs().is_empty());
    // 1-hop pairs should be calibrated to 3.0 (not raw 1.0)
    let has_calibrated_1hop = sgd.node_pairs().iter().any(|&(u, v, d, _, _, _)| {
        (u == 0 && v == 1 || u == 1 && v == 0) && (d - 3.0).abs() < 1e-9
    });
    assert!(has_calibrated_1hop);
    // 2-hop pairs (e.g. distance 6.0) should also be registered as two-sided springs
    let has_2hop = sgd
        .node_pairs()
        .iter()
        .any(|&(_, _, d, _, _, _)| (d - 6.0).abs() < 1e-9);
    assert!(has_2hop);
    // Hinge pairs must enforce barrier >= (1 + delta_rel) * r_2 = 1.25 * 6.0 = 7.5
    for &(_, _, d_bar, _, wij, wji) in sgd.hinge_pairs() {
        assert!(d_bar >= 7.5 - 1e-9);
        assert_eq!(wij, 0.0);
        assert!(wji > 0.0 && wji.is_finite());
    }
}

#[test]
fn test_sparse_sgd_barrier_star_2000_hub_safe_zero_o_n2() {
    let n = 2000;
    let cap = 16;
    let g = star_graph(n);
    let mut rng = StdRng::seed_from_u64(42);
    let sgd = SparseSgd::barrier()
        .h(16)
        .max_2hop_per_node(cap)
        .build(&g, |_| 1.0_f64, &mut rng);

    // Total 1-hop + 2-hop pairs must be O(N * cap), nowhere near N*(N-1)/2 (~2e6)
    assert!(sgd.node_pairs().len() <= (n - 1) + n * cap);
    assert!(sgd.node_pairs().len() < 40_000);
    // In a star graph every pair of leaves is 2-hop; no non-2-hop pairs exist
    assert!(sgd.hinge_pairs().is_empty());

    let mut seen = HashSet::new();
    for &(u, v, d, _, w1, w2) in sgd.node_pairs() {
        assert_ne!(u, v);
        assert!(d.is_finite() && d > 0.0);
        assert!(w1.is_finite() && w2.is_finite());
        let key = if u < v { (u, v) } else { (v, u) };
        assert!(seen.insert(key), "duplicate pair {:?}", key);
    }
}

#[test]
fn test_build_barrier_from_landmark_tables_parity_and_validation() {
    // 5-node path 0-1-2-3-4 matching tests/test_sparse_barrier_sgd.py
    let mut g = UnGraph::<(), ()>::new_undirected();
    let nodes: Vec<_> = (0..5).map(|_| g.add_node(())).collect();
    for i in 0..4 {
        g.add_edge(nodes[i], nodes[i + 1], ());
    }
    // Construct 1D landmark coordinate table D[u, 0] = u -> exact path distance |u - v|
    let d_ms = Array2::from_shape_fn((5, 1), |(i, _)| i as f64);
    let (sgd_hinge, stats) = SparseSgd::barrier()
        .max_2hop_per_node(32)
        .delta_rel(0.5)
        .barrier_sym(BarrierSymmetry::SymMax)
        .omega_1(2.0)
        .omega_2(0.5)
        .gamma_non2(3.0)
        .min_ratio(0.5)
        .build_barrier_from_landmark_tables(&g, d_ms.view(), None, &[0, 4], Some(8))
        .unwrap();

    assert_eq!(stats.n_pairs_1hop, 4);
    assert_eq!(stats.n_pairs_2hop, 3);
    assert_eq!(stats.n_pairs_non2, 3);
    assert_eq!(stats.n_pairs_two_sided, 7);
    assert_eq!(stats.n_pairs_hinge, 3);
    assert!((stats.d_floor - 0.5).abs() < 1e-12);

    // Verify barrier target distances match 1.5 * max(R2[u], R2[v])
    for &(u, v, d_bar, _, w_u, w_v) in sgd_hinge.hinge_pairs() {
        // In path 0-1-2-3-4 with d_ms=[0,1,2,3,4], R2 = [2, 2, 2, 2, 2] -> 1.5 * 2 = 3.0 for (0,3),(1,4), and max(4, 3)=4.0 for (0,4)
        let expected = if (u == 0 && v == 4) || (u == 4 && v == 0) {
            4.0
        } else {
            3.0
        };
        assert!((d_bar - expected).abs() < 1e-12);
        assert!((w_u - 3.0 / (expected * expected)).abs() < 1e-12);
        assert!((w_v - 3.0 / (expected * expected)).abs() < 1e-12);
    }

    // Validation errors on mismatched row count or out-of-range pivot
    let bad_d = Array2::zeros((3, 1));
    assert!(SparseSgd::barrier()
        .build_barrier_from_landmark_tables(&g, bad_d.view(), None, &[0], Some(4))
        .is_err());
    assert!(SparseSgd::barrier()
        .build_barrier_from_landmark_tables(&g, d_ms.view(), None, &[10], Some(4))
        .is_err());
}

#[test]
fn test_sparse_sgd_disconnected_and_zero_weight_graphs_no_panic() {
    let mut g_disc = UnGraph::<(), ()>::new_undirected();
    let dn: Vec<_> = (0..5).map(|_| g_disc.add_node(())).collect();
    g_disc.add_edge(dn[0], dn[1], ());
    g_disc.add_edge(dn[2], dn[3], ());
    let mut rng = StdRng::seed_from_u64(42);
    let sgd_legacy = SparseSgd::new().h(4).build(&g_disc, |_| 1.0_f64, &mut rng);
    assert!(!sgd_legacy.node_pairs().is_empty());
    let sgd_barrier = SparseSgd::barrier()
        .h(4)
        .build(&g_disc, |_| 1.0_f64, &mut rng);
    assert!(!sgd_barrier.node_pairs().is_empty());

    let g_cycle = cycle_graph(5);
    let sgd_zero = SparseSgd::new().h(3).build(&g_cycle, |_| 0.0_f64, &mut rng);
    assert_eq!(sgd_zero.node_pairs().len(), 5);
}

#[test]
fn test_sparse_sgd_barrier_duplicate_pivots_preserve_2hop_partition() {
    let g = cycle_graph(6);
    let nodes: Vec<_> = g.node_indices().collect();
    let sgd_unique = SparseSgd::barrier()
        .h(1)
        .max_2hop_per_node(4)
        .build_with_pivot(&g, |_| 1.0_f64, &[nodes[0]]);
    let sgd_dup = SparseSgd::barrier()
        .h(2)
        .max_2hop_per_node(4)
        .build_with_pivot(&g, |_| 1.0_f64, &[nodes[0], nodes[0]]);
    assert_eq!(sgd_dup.node_pairs().len(), sgd_unique.node_pairs().len());
    assert_eq!(sgd_dup.hinge_pairs().len(), sgd_unique.hinge_pairs().len());
}

#[test]
fn test_build_barrier_from_landmark_tables_n0_and_n1_validation() {
    let empty = UnGraph::<(), ()>::new_undirected();
    let bad_rows = Array2::from_elem((3, 1), f64::NAN);
    assert!(SparseSgd::barrier()
        .build_barrier_from_landmark_tables(&empty, bad_rows.view(), None, &[], None)
        .is_err());

    let ok_empty = Array2::<f64>::zeros((0, 0));
    assert!(SparseSgd::barrier()
        .build_barrier_from_landmark_tables(&empty, ok_empty.view(), None, &[], None)
        .is_ok());

    let mut single = UnGraph::<(), ()>::new_undirected();
    single.add_node(());
    let bad_n1 = Array2::<f64>::zeros((1, 0));
    assert!(SparseSgd::barrier()
        .build_barrier_from_landmark_tables(&single, bad_n1.view(), None, &[], None)
        .is_err());
    let ok_n1 = Array2::<f64>::zeros((1, 1));
    assert!(SparseSgd::barrier()
        .build_barrier_from_landmark_tables(&single, ok_n1.view(), None, &[], None)
        .is_ok());
}

#[test]
fn test_sparse_sgd_try_build_with_mismatched_pivoted_distance_returns_err() {
    let g3 = cycle_graph(3);
    let g10 = cycle_graph(10);
    let lap10 = SparseSymmetricMatrix::symmetric_normalized_laplacian(&g10, |_| 1.0_f64);
    let mut rng = StdRng::seed_from_u64(1);
    let pk10 = PivotedBandPassDiffusionKernelBuilder::dog(
        &lap10,
        0.5_f64,
        2.0_f64,
        0.5_f64,
        0.05_f64,
        10,
        vec![8, 9],
    )
    .build(&mut rng)
    .unwrap();
    let dist10 = PivotedNegLogSimDistanceBuilder::new()
        .build(&g10, pk10)
        .unwrap();
    assert!(SparseSgd::barrier()
        .try_build_with_pivoted_neg_log_sim_distance(&g3, |_| 1.0_f64, &dist10)
        .is_err());
}
