use egraph_core::{DeterministicRng, VisitStamp};

use super::level_graph::{LeidenLevelGraph, canonicalize_labels};

/// `RefinePartition` / `MergeNodesSubset` phase (Traag et al., 2019, Algorithm A.2, lines 26-43).
///
/// Refines each outer community `S` into internally well-connected sub-communities using
/// positive-edge filtering (`E(v, C) > 0.0`), $O(1)$ `ext_in_s` updates, deterministic sorted
/// candidate ordering, and log-sum-exp stabilized softmax sampling.
pub(super) fn refine_partition(
    level: &LeidenLevelGraph,
    comm: &[u32],
    num_outer: usize,
    gamma_eff: f64,
    randomness: f64,
    rng: &mut DeterministicRng,
) -> (Vec<u32>, usize) {
    let nl = level.num_nodes;
    let mut sub_comm: Vec<u32> = (0..nl as u32).collect();
    let mut sub_size = level.node_sizes.clone();
    let mut sub_members = vec![1u32; nl];

    let mut outer_counts = vec![0usize; num_outer];
    for &c in comm {
        outer_counts[c as usize] += 1;
    }
    let mut outer_groups: Vec<Vec<u32>> = outer_counts
        .iter()
        .map(|&count| Vec::with_capacity(count))
        .collect();
    for (u, &c) in comm.iter().enumerate() {
        outer_groups[c as usize].push(u as u32);
    }

    let mut ext_in_s = vec![0.0_f64; nl];
    let mut r_nodes: Vec<u32> = Vec::new();
    let mut sub_stamp = VisitStamp::new(nl);
    let mut sub_edge_weights = vec![0.0_f64; nl];
    let mut touched_subs: Vec<u32> = Vec::new();
    let mut candidates: Vec<(u32, f64)> = Vec::new();

    for (s_id, members) in outer_groups.iter().enumerate() {
        if members.len() <= 1 {
            continue;
        }

        let mut total_s_size = 0.0_f64;
        for &u_u32 in members {
            let u = u_u32 as usize;
            total_s_size += level.node_sizes[u];
            let mut sum_s = 0.0_f64;
            let start = level.indptr[u] as usize;
            let end = level.indptr[u + 1] as usize;
            for pos in start..end {
                let w = level.weights[pos];
                if w <= 0.0 {
                    continue;
                }
                let v = level.indices[pos] as usize;
                if comm[v] as usize == s_id {
                    sum_s += w;
                }
            }
            ext_in_s[u] = sum_s;
        }

        r_nodes.clear();
        for &u_u32 in members {
            let u = u_u32 as usize;
            let size_u = level.node_sizes[u];
            let rhs = gamma_eff * size_u * (total_s_size - size_u).max(0.0);
            if ext_in_s[u] + 1e-12 * rhs.abs() >= rhs {
                r_nodes.push(u_u32);
            }
        }

        rng.shuffle(&mut r_nodes);

        for &v_u32 in &r_nodes {
            let v = v_u32 as usize;
            let c_v = sub_comm[v] as usize;
            if sub_members[c_v] != 1 {
                continue;
            }

            sub_stamp.clear();
            touched_subs.clear();
            let start = level.indptr[v] as usize;
            let end = level.indptr[v + 1] as usize;
            for pos in start..end {
                let w = level.weights[pos];
                if w <= 0.0 {
                    continue;
                }
                let nbr = level.indices[pos] as usize;
                if comm[nbr] as usize != s_id {
                    continue;
                }
                let cn = sub_comm[nbr];
                if cn as usize == c_v {
                    continue;
                }
                if sub_stamp.visit(cn as usize) {
                    touched_subs.push(cn);
                }
                sub_edge_weights[cn as usize] += w;
            }

            if touched_subs.is_empty() {
                continue;
            }

            touched_subs.sort_unstable();
            candidates.clear();
            let size_v = level.node_sizes[v];
            let mut max_gain = 0.0_f64;

            for &cand_c in &touched_subs {
                let c_idx = cand_c as usize;
                let e_v_c = sub_edge_weights[c_idx];
                if e_v_c <= 0.0 {
                    continue;
                }
                let size_c = sub_size[c_idx];
                let rhs = gamma_eff * size_c * (total_s_size - size_c).max(0.0);
                if ext_in_s[c_idx] + 1e-12 * rhs.abs() < rhs {
                    continue;
                }
                let penalty = gamma_eff * size_v * size_c;
                let gain = e_v_c - penalty;
                let tol = 1e-12 * (e_v_c.abs() + penalty.abs());
                if gain >= -tol {
                    let clamped_gain = gain.max(0.0);
                    if clamped_gain > max_gain {
                        max_gain = clamped_gain;
                    }
                    candidates.push((cand_c, clamped_gain));
                }
            }

            if !candidates.is_empty() {
                // Include staying singleton ({v}, 0.0) and apply log-sum-exp max-subtraction.
                let mut total_weight = (-max_gain / randomness).exp();
                for &(_, g) in &candidates {
                    total_weight += ((g - max_gain) / randomness).exp();
                }

                let target_draw = rng.next_f64() * total_weight;
                let mut cum = 0.0_f64;
                let mut chosen_c = c_v as u32;
                for &(cand_c, g) in &candidates {
                    cum += ((g - max_gain) / randomness).exp();
                    if target_draw < cum {
                        chosen_c = cand_c;
                        break;
                    }
                }

                if chosen_c != c_v as u32 {
                    let chosen_idx = chosen_c as usize;
                    let e_v_chosen = sub_edge_weights[chosen_idx];
                    sub_comm[v] = chosen_c;
                    sub_members[c_v] = 0;
                    sub_members[chosen_idx] += 1;
                    sub_size[chosen_idx] += size_v;
                    ext_in_s[chosen_idx] =
                        (ext_in_s[chosen_idx] + ext_in_s[v] - 2.0 * e_v_chosen).max(0.0);
                }
            }

            for &cand_c in &touched_subs {
                sub_edge_weights[cand_c as usize] = 0.0;
            }
        }
    }

    let num_refined = canonicalize_labels(&mut sub_comm, nl);
    (sub_comm, num_refined)
}
