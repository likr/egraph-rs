use std::collections::VecDeque;

use egraph_core::{DeterministicRng, VisitStamp};

use super::level_graph::LeidenLevelGraph;

/// Queue-based `MoveNodesFast` phase (Traag et al., 2019, Algorithm A.2, lines 13-25).
///
/// Considers neighbor communities and splitting to an empty community `∅` when `u` is not a
/// singleton, enforcing a relative ULP strictness guard to prevent floating-point oscillation.
pub(super) fn move_nodes_fast(
    level: &LeidenLevelGraph,
    comm: &mut [u32],
    gamma_eff: f64,
    rng: &mut DeterministicRng,
) -> bool {
    let nl = level.num_nodes;
    let mut comm_size = vec![0.0_f64; nl];
    let mut comm_members = vec![0u32; nl];
    for u in 0..nl {
        let c = comm[u] as usize;
        comm_size[c] += level.node_sizes[u];
        comm_members[c] += 1;
    }

    let mut free_comm_ids = Vec::with_capacity(nl);
    for c in (0..nl).rev() {
        if comm_members[c] == 0 {
            free_comm_ids.push(c as u32);
        }
    }

    let mut order: Vec<u32> = (0..nl as u32).collect();
    rng.shuffle(&mut order);
    let mut queue: VecDeque<u32> = VecDeque::from(order);
    let mut in_queue = vec![true; nl];

    let mut comm_stamp = VisitStamp::new(nl);
    let mut comm_weights = vec![0.0_f64; nl];
    let mut touched_comms: Vec<u32> = Vec::new();
    let mut any_moved = false;

    while let Some(u_u32) = queue.pop_front() {
        let u = u_u32 as usize;
        in_queue[u] = false;

        let c_old = comm[u];
        let c_old_idx = c_old as usize;
        let rem_old_members = comm_members[c_old_idx] - 1;

        comm_stamp.clear();
        touched_comms.clear();

        let start = level.indptr[u] as usize;
        let end = level.indptr[u + 1] as usize;
        for pos in start..end {
            let w = level.weights[pos];
            if w <= 0.0 {
                continue;
            }
            let v = level.indices[pos] as usize;
            let cv = comm[v];
            let cv_idx = cv as usize;
            if comm_stamp.visit(cv_idx) {
                touched_comms.push(cv);
            }
            comm_weights[cv_idx] += w;
        }

        if touched_comms.is_empty() && rem_old_members == 0 {
            continue;
        }

        touched_comms.sort_unstable();

        let size_u = level.node_sizes[u];
        let prev_old_size = comm_size[c_old_idx];
        let rem_old_size = if rem_old_members == 0 {
            0.0
        } else {
            (prev_old_size - size_u).max(0.0)
        };
        let e_u_old = if comm_stamp.is_visited(c_old_idx) {
            comm_weights[c_old_idx]
        } else {
            0.0
        };
        let old_term = e_u_old - gamma_eff * size_u * rem_old_size;

        let mut best_delta_h = 0.0_f64;
        let mut best_cand: Option<u32> = None;
        let mut move_to_empty = false;

        for &cand in &touched_comms {
            if cand == c_old {
                continue;
            }
            let cand_idx = cand as usize;
            let cand_term = comm_weights[cand_idx] - gamma_eff * size_u * comm_size[cand_idx];
            let delta_h = cand_term - old_term;
            let min_gain = 1e-12 * (old_term.abs() + cand_term.abs());
            if delta_h > min_gain
                && (best_cand.is_none() || delta_h > best_delta_h + 1e-12 * best_delta_h.abs())
            {
                best_delta_h = delta_h;
                best_cand = Some(cand);
            }
        }

        if rem_old_members > 0 {
            let delta_h_empty = -old_term;
            let min_gain = 1e-12 * old_term.abs();
            if delta_h_empty > min_gain
                && (best_cand.is_none()
                    || delta_h_empty > best_delta_h + 1e-12 * best_delta_h.abs())
            {
                best_cand = None;
                move_to_empty = true;
            }
        }

        for &cand in &touched_comms {
            comm_weights[cand as usize] = 0.0;
        }

        let chosen_target = if move_to_empty {
            free_comm_ids.pop()
        } else {
            best_cand
        };

        if let Some(target_c) = chosen_target {
            any_moved = true;
            comm_members[c_old_idx] = rem_old_members;
            comm_size[c_old_idx] = rem_old_size;
            if rem_old_members == 0 {
                free_comm_ids.push(c_old);
            }

            let target_idx = target_c as usize;
            comm[u] = target_c;
            comm_members[target_idx] += 1;
            comm_size[target_idx] += size_u;

            for pos in start..end {
                if level.weights[pos] <= 0.0 {
                    continue;
                }
                let v = level.indices[pos] as usize;
                if comm[v] != target_c && !in_queue[v] {
                    in_queue[v] = true;
                    queue.push_back(v as u32);
                }
            }
        }
    }

    any_moved
}
