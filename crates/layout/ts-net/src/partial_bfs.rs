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
) -> Vec<Vec<(usize, S)>>
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
    let perplexity_clamped = perplexity
        .min(S::from_usize(k_eff).unwrap() - S::from_f32(0.01).unwrap())
        .max(S::one());
    let target_entropy = perplexity_clamped.ln();

    // Map each node identifier to a dense index 0..n
    let node_to_idx: HashMap<G::NodeId, usize> = graph
        .node_identifiers()
        .enumerate()
        .map(|(i, id)| (id, i))
        .collect();

    let mut row_probs: Vec<HashMap<usize, S>> = vec![HashMap::new(); n];

    // Step 1: Run Partial BFS from each vertex
    for (i, node_id) in graph.node_identifiers().enumerate() {
        let mut visited = vec![false; n];
        visited[i] = true;

        let mut current_layer = vec![node_id];

        let mut collected: Vec<(usize, S)> = Vec::with_capacity(k_eff);
        let mut current_dist = S::zero();

        while !current_layer.is_empty() && collected.len() < k_eff {
            current_dist += S::one();
            let mut next_layer = Vec::new();

            for &u in &current_layer {
                for neighbor in graph.neighbors(u) {
                    let neighbor_idx = node_to_idx[&neighbor];
                    if !visited[neighbor_idx] {
                        visited[neighbor_idx] = true;
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
                current_layer = next_layer;
            } else {
                // Break ties randomly when next layer exceeds remaining neighbor budget
                next_layer.shuffle(rng);
                for &v_neighbor in next_layer.iter().take(remaining_budget) {
                    collected.push((node_to_idx[&v_neighbor], current_dist));
                }
                break;
            }
        }

        if collected.is_empty() {
            continue;
        }

        // Step 2: Binary search on beta = 1 / (2 * sigma_i^2) to achieve target perplexity
        let m = collected.len();
        let mut beta = S::one();
        let mut beta_min = S::zero();
        let mut beta_max = S::from_f32(1e12).unwrap();

        for _ in 0..sigma_iters {
            let mut max_neg_dp = S::neg_infinity();
            for &(_, d) in &collected {
                let dp = d.powf(power);
                let neg_dp = -beta * dp;
                if neg_dp > max_neg_dp {
                    max_neg_dp = neg_dp;
                }
            }

            let mut sum_w = S::zero();
            let mut weights = vec![S::zero(); m];
            for (idx, &(_, d)) in collected.iter().enumerate() {
                let dp = d.powf(power);
                let w = (-beta * dp - max_neg_dp).exp();
                weights[idx] = w;
                sum_w += w;
            }

            sum_w = sum_w.max(S::from_f32(1e-12).unwrap());

            let mut sum_dp_p = S::zero();
            for (idx, &(_, d)) in collected.iter().enumerate() {
                let prob = weights[idx] / sum_w;
                let dp = d.powf(power);
                sum_dp_p += prob * dp;
            }

            let h = beta * sum_dp_p + max_neg_dp + sum_w.ln();
            let h_diff = h - target_entropy;

            if h_diff.abs() < sigma_tolerance {
                break;
            }

            if h_diff > S::zero() {
                beta_min = beta;
                if beta_max == S::from_f32(1e12).unwrap() {
                    beta *= S::from_f32(2.0).unwrap();
                } else {
                    beta = (beta_min + beta_max) / S::from_f32(2.0).unwrap();
                }
            } else {
                beta_max = beta;
                beta = (beta_min + beta_max) / S::from_f32(2.0).unwrap();
            }
        }

        // Compute final conditional probabilities p_{j|i}
        let mut max_neg_dp = S::neg_infinity();
        for &(_, d) in &collected {
            let dp = d.powf(power);
            let neg_dp = -beta * dp;
            if neg_dp > max_neg_dp {
                max_neg_dp = neg_dp;
            }
        }

        let mut sum_w = S::zero();
        let mut weights = vec![S::zero(); m];
        for (idx, &(_, d)) in collected.iter().enumerate() {
            let dp = d.powf(power);
            let w = (-beta * dp - max_neg_dp).exp();
            weights[idx] = w;
            sum_w += w;
        }

        sum_w = sum_w.max(S::from_f32(1e-12).unwrap());
        for (idx, &(j, _)) in collected.iter().enumerate() {
            row_probs[i].insert(j, weights[idx] / sum_w);
        }
    }

    // Step 3: Compute symmetrized joint probabilities p_{ij} = (p_{j|i} + p_{i|j}) / (2 * N)
    let two_n = S::from_usize(2 * n).unwrap();
    let mut sparse_p: Vec<Vec<(usize, S)>> = vec![Vec::new(); n];

    // Collect all undirected pairs
    let mut all_pairs: HashMap<(usize, usize), S> = HashMap::new();
    for i in 0..n {
        for (&j, &p_ji) in &row_probs[i] {
            let p_ij = row_probs[j].get(&i).copied().unwrap_or(S::zero());
            let p_sym = (p_ji + p_ij) / two_n;
            if p_sym > S::zero() {
                all_pairs.insert((i, j), p_sym);
            }
        }
    }

    for ((i, j), p_val) in all_pairs {
        sparse_p[i].push((j, p_val));
    }

    sparse_p
}
