use petgraph::graph::UnGraph;
use petgraph::visit::{IntoNeighbors, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_drawing::DrawingIndex;
use petgraph_layout_ts_net::{
    compute_knn_probabilities_with_params, compute_partial_bfs_probabilities, SparseP,
};
use rand::prelude::*;
use rand::rngs::StdRng;
use std::collections::{HashMap, HashSet};

fn reference_partial_bfs_probabilities<G, R>(
    graph: G,
    perplexity: f64,
    k: usize,
    power: f64,
    sigma_iters: usize,
    sigma_tolerance: f64,
    rng: &mut R,
) -> SparseP<f64>
where
    G: IntoNeighbors + IntoNodeIdentifiers + NodeCount + NodeIndexable,
    G::NodeId: Copy + Eq + std::hash::Hash + DrawingIndex,
    R: Rng + ?Sized,
{
    let n = graph.node_count();
    if n < 2 {
        return vec![Vec::new(); n];
    }
    let k_eff = k.min(n - 1).max(1);
    let node_to_idx: HashMap<G::NodeId, usize> = graph
        .node_identifiers()
        .enumerate()
        .map(|(i, id)| (id, i))
        .collect();

    let mut knn = Vec::with_capacity(n);
    for node_id in graph.node_identifiers() {
        let mut visited: HashSet<G::NodeId> = HashSet::new();
        visited.insert(node_id);
        let mut current_layer = vec![node_id];
        let mut collected = Vec::with_capacity(k_eff);
        let mut current_dist = 0.0f64;

        while !current_layer.is_empty() && collected.len() < k_eff {
            current_dist += 1.0;
            let mut next_layer = Vec::new();
            for &u in &current_layer {
                for neighbor in graph.neighbors(u) {
                    if visited.insert(neighbor) {
                        next_layer.push(neighbor);
                    }
                }
            }
            if next_layer.is_empty() {
                break;
            }
            let remaining = k_eff - collected.len();
            if next_layer.len() <= remaining {
                for &v in &next_layer {
                    collected.push((node_to_idx[&v], current_dist));
                }
                current_layer = next_layer;
            } else {
                next_layer.shuffle(rng);
                for &v in next_layer.iter().take(remaining) {
                    collected.push((node_to_idx[&v], current_dist));
                }
                break;
            }
        }
        knn.push(collected);
    }

    compute_knn_probabilities_with_params(n, &knn, perplexity, power, sigma_iters, sigma_tolerance)
}

fn assert_sparse_p_bit_equal(actual: &SparseP<f64>, expected: &SparseP<f64>, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: row count mismatch");
    for (i, (row_a, row_e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(row_a.len(), row_e.len(), "{label}: row {i} len mismatch");
        for (&(ja, pa), &(je, pe)) in row_a.iter().zip(row_e.iter()) {
            assert_eq!(ja, je, "{label}: row {i} col mismatch");
            assert_eq!(
                pa.to_bits(),
                pe.to_bits(),
                "{label}: row {i} col {ja} value mismatch: {pa} vs {pe}"
            );
        }
    }
}

#[test]
fn test_partial_bfs_matches_naive_reference_bit_for_bit() {
    // 1. Connected 4x4 grid with tie-breaking (k = 5 < 2-hop neighborhood)
    let mut grid = UnGraph::<(), ()>::new_undirected();
    let grid_nodes: Vec<_> = (0..16).map(|_| grid.add_node(())).collect();
    for r in 0..4 {
        for c in 0..4 {
            let u = r * 4 + c;
            if c + 1 < 4 {
                grid.add_edge(grid_nodes[u], grid_nodes[u + 1], ());
            }
            if r + 1 < 4 {
                grid.add_edge(grid_nodes[u], grid_nodes[u + 4], ());
            }
        }
    }
    let actual = compute_partial_bfs_probabilities(
        &grid,
        4.0,
        5,
        2.0,
        50,
        1e-5,
        &mut StdRng::seed_from_u64(42),
    );
    let expected = reference_partial_bfs_probabilities(
        &grid,
        4.0,
        5,
        2.0,
        50,
        1e-5,
        &mut StdRng::seed_from_u64(42),
    );
    assert_sparse_p_bit_equal(&actual, &expected, "grid_4x4_k5");

    // 2. Disconnected components (triangle + edge + isolated node)
    let mut disc = UnGraph::<(), ()>::new_undirected();
    let dn: Vec<_> = (0..6).map(|_| disc.add_node(())).collect();
    disc.add_edge(dn[0], dn[1], ());
    disc.add_edge(dn[1], dn[2], ());
    disc.add_edge(dn[2], dn[0], ());
    disc.add_edge(dn[3], dn[4], ());
    let actual_disc = compute_partial_bfs_probabilities(
        &disc,
        2.0,
        4,
        2.0,
        50,
        1e-5,
        &mut StdRng::seed_from_u64(42),
    );
    let expected_disc = reference_partial_bfs_probabilities(
        &disc,
        2.0,
        4,
        2.0,
        50,
        1e-5,
        &mut StdRng::seed_from_u64(42),
    );
    assert_sparse_p_bit_equal(&actual_disc, &expected_disc, "disconnected");
    assert!(actual_disc[5].is_empty());

    // 3. Self-loops and parallel edges
    let mut multi = UnGraph::<(), ()>::new_undirected();
    let mn: Vec<_> = (0..5).map(|_| multi.add_node(())).collect();
    multi.add_edge(mn[0], mn[0], ());
    multi.add_edge(mn[0], mn[1], ());
    multi.add_edge(mn[0], mn[1], ());
    multi.add_edge(mn[1], mn[2], ());
    multi.add_edge(mn[2], mn[3], ());
    multi.add_edge(mn[3], mn[4], ());
    multi.add_edge(mn[4], mn[2], ());
    let actual_multi = compute_partial_bfs_probabilities(
        &multi,
        2.0,
        3,
        2.0,
        50,
        1e-5,
        &mut StdRng::seed_from_u64(42),
    );
    let expected_multi = reference_partial_bfs_probabilities(
        &multi,
        2.0,
        3,
        2.0,
        50,
        1e-5,
        &mut StdRng::seed_from_u64(42),
    );
    assert_sparse_p_bit_equal(&actual_multi, &expected_multi, "self_loop_parallel");

    // 4. Non-contiguous NodeIndex gaps after remove_node on StableUnGraph
    use petgraph::stable_graph::StableUnGraph;
    let mut gapped = StableUnGraph::<(), ()>::default();
    let gn: Vec<_> = (0..6).map(|_| gapped.add_node(())).collect();
    for i in 0..5 {
        gapped.add_edge(gn[i], gn[i + 1], ());
    }
    gapped.add_edge(gn[0], gn[2], ());
    gapped.add_edge(gn[2], gn[4], ());
    gapped.add_edge(gn[4], gn[5], ());
    gapped.remove_node(gn[1]);
    gapped.remove_node(gn[3]);
    assert!(gapped.node_bound() > gapped.node_count());
    let actual_gapped = compute_partial_bfs_probabilities(
        &gapped,
        2.0,
        3,
        2.0,
        50,
        1e-5,
        &mut StdRng::seed_from_u64(42),
    );
    let expected_gapped = reference_partial_bfs_probabilities(
        &gapped,
        2.0,
        3,
        2.0,
        50,
        1e-5,
        &mut StdRng::seed_from_u64(42),
    );
    assert_sparse_p_bit_equal(&actual_gapped, &expected_gapped, "remove_node_gaps");

    // 5. Edge cases: n = 0, n = 1, k = 0
    let empty = UnGraph::<(), ()>::new_undirected();
    assert!(compute_partial_bfs_probabilities(
        &empty,
        2.0,
        5,
        2.0,
        50,
        1e-5,
        &mut StdRng::seed_from_u64(42)
    )
    .is_empty());

    let mut single = UnGraph::<(), ()>::new_undirected();
    single.add_node(());
    assert_eq!(
        compute_partial_bfs_probabilities(
            &single,
            2.0,
            5,
            2.0,
            50,
            1e-5,
            &mut StdRng::seed_from_u64(42)
        ),
        vec![Vec::<(usize, f64)>::new()]
    );

    let actual_k0 = compute_partial_bfs_probabilities(
        &grid,
        2.0,
        0,
        2.0,
        50,
        1e-5,
        &mut StdRng::seed_from_u64(42),
    );
    let expected_k0 = reference_partial_bfs_probabilities(
        &grid,
        2.0,
        0,
        2.0,
        50,
        1e-5,
        &mut StdRng::seed_from_u64(42),
    );
    assert_sparse_p_bit_equal(&actual_k0, &expected_k0, "k_zero");
}

#[test]
fn test_partial_bfs_medium_large_sparse_sanity() {
    let n = 2000;
    let mut graph = UnGraph::<(), ()>::new_undirected();
    let nodes: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();
    for i in 0..n {
        graph.add_edge(nodes[i], nodes[(i + 1) % n], ());
        graph.add_edge(nodes[i], nodes[(i + 7) % n], ());
    }

    let mut rng = StdRng::seed_from_u64(42);
    let sparse_p = compute_partial_bfs_probabilities(&graph, 10.0f64, 20, 2.0, 50, 1e-5, &mut rng);
    assert_eq!(sparse_p.len(), n);

    let mut total_sum = 0.0;
    for (i, row) in sparse_p.iter().enumerate() {
        assert!(!row.is_empty(), "row {i} unexpectedly empty");
        let mut prev = None;
        for &(j, p_ij) in row {
            assert_ne!(i, j);
            assert!(p_ij.is_finite() && p_ij > 0.0);
            if let Some(p) = prev {
                assert!(p < j);
            }
            prev = Some(j);
            total_sum += p_ij;
        }
    }
    assert!(
        (total_sum - 1.0).abs() < 1e-5,
        "expected probability sum 1.0, got {total_sum}"
    );
}
