use super::{compute_knn_probabilities_with_params, SparseP};
use num_traits::Float;
use petgraph::visit::{IntoNeighbors, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use rand::prelude::*;
use std::collections::HashMap;

/// Computes sparse high-dimensional probabilities via Partial BFS in O(N) time (C0 component of BH-tsNET).
///
/// For each vertex `v`, Partial BFS explores the graph until `k` neighbors have been visited.
/// Ties at equal graph distances are broken randomly to ensure unbiased neighbor sampling.
/// Gaussian similarities are then normalized using binary search over standard deviation `sigma_v`
/// to attain the target perplexity `u`.
pub fn compute_partial_bfs_probabilities<G, S, R>(
    graph: G,
    perplexity: S,
    k: usize,
    power: S,
    sigma_iters: usize,
    sigma_tolerance: S,
    rng: &mut R,
) -> SparseP<S>
where
    G: IntoNeighbors + IntoNodeIdentifiers + NodeCount + NodeIndexable,
    G::NodeId: Copy + Eq + std::hash::Hash + DrawingIndex,
    S: DrawingValue + Float + Default,
    R: Rng + ?Sized,
{
    let n = graph.node_count();
    if n < 2 {
        return vec![Vec::new(); n];
    }

    let k_eff = k.min(n - 1).max(1);

    // Map each node identifier to a dense index 0..n
    let node_to_idx: HashMap<G::NodeId, usize> = graph
        .node_identifiers()
        .enumerate()
        .map(|(i, id)| (id, i))
        .collect();

    let mut knn: Vec<Vec<(usize, S)>> = Vec::with_capacity(n);
    // Allocate stamp and frontier buffers once outside the outer loop to keep per-vertex cost O(k * deg).
    let mut visited_stamp = vec![0usize; n];
    let mut current_layer: Vec<G::NodeId> = Vec::new();
    let mut next_layer: Vec<G::NodeId> = Vec::new();

    // Step 1: Run Partial BFS from each vertex
    for (i, node_id) in graph.node_identifiers().enumerate() {
        let stamp = i + 1;
        visited_stamp[i] = stamp;

        current_layer.clear();
        current_layer.push(node_id);

        let mut collected: Vec<(usize, S)> = Vec::with_capacity(k_eff);
        let mut current_dist = S::zero();

        while !current_layer.is_empty() && collected.len() < k_eff {
            current_dist += S::one();
            next_layer.clear();

            for &u in &current_layer {
                for neighbor in graph.neighbors(u) {
                    let neighbor_idx = node_to_idx[&neighbor];
                    if visited_stamp[neighbor_idx] != stamp {
                        visited_stamp[neighbor_idx] = stamp;
                        next_layer.push(neighbor);
                    }
                }
            }

            if next_layer.is_empty() {
                break;
            }

            let remaining_budget = k_eff - collected.len();
            if next_layer.len() <= remaining_budget {
                for &v_neighbor in &next_layer {
                    collected.push((node_to_idx[&v_neighbor], current_dist));
                }
                std::mem::swap(&mut current_layer, &mut next_layer);
            } else {
                // Break ties randomly when next layer exceeds remaining neighbor budget
                next_layer.shuffle(rng);
                for &v_neighbor in next_layer.iter().take(remaining_budget) {
                    collected.push((node_to_idx[&v_neighbor], current_dist));
                }
                break;
            }
        }

        knn.push(collected);
    }

    compute_knn_probabilities_with_params(n, &knn, perplexity, power, sigma_iters, sigma_tolerance)
}
