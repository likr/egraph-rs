use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, Scalar, VisitStamp};

use super::level_graph::{WeightedLevelGraph, canonicalize_labels};
use super::{LouvainConfig, LouvainResult};

/// Runs two-phase multi-level Louvain community detection on an undirected [`CsrGraph`].
///
/// Phase 1 sweeps vertices in deterministic shuffled order (via `DeterministicRng`) and
/// accumulates neighbor community weights using a `VisitStamp` epoch array and flat `Vec`
/// scratch buffers. Phase 2 contracts communities into a super-graph without hash maps.
/// `LouvainResult::modularity` reports $Q_\gamma$ evaluated with `config.resolution`.
pub fn louvain<W: Scalar>(
    graph: &CsrGraph<W>,
    config: &LouvainConfig,
) -> CoreResult<LouvainResult> {
    if graph.is_directed() {
        return Err(CoreError::invalid_input(
            "louvain requires an undirected graph",
        ));
    }
    if config.max_passes == 0 {
        return Err(CoreError::invalid_input("max_passes must be >= 1"));
    }
    if !config.resolution.is_finite() {
        return Err(CoreError::non_finite("LouvainConfig::resolution"));
    }
    if config.resolution < 0.0 {
        return Err(CoreError::invalid_input("resolution must be >= 0.0"));
    }
    if !config.min_modularity_gain.is_finite() {
        return Err(CoreError::non_finite("LouvainConfig::min_modularity_gain"));
    }
    if config.min_modularity_gain < 0.0 {
        return Err(CoreError::invalid_input(
            "min_modularity_gain must be >= 0.0",
        ));
    }

    let n = graph.num_nodes();
    if n == 0 {
        return Ok(LouvainResult {
            labels: Vec::new(),
            num_communities: 0,
            modularity: 0.0,
            passes: 0,
        });
    }

    let mut level = WeightedLevelGraph::from_csr(graph)?;
    let mut global_labels: Vec<u32> = (0..n as u32).collect();

    if level.total_weight_2m <= 0.0 {
        return Ok(LouvainResult {
            labels: global_labels,
            num_communities: n,
            modularity: 0.0,
            passes: 0,
        });
    }

    let mut rng = DeterministicRng::from_seed_u64(config.seed);
    let mut passes = 0usize;
    let mut current_q = {
        let init_comm: Vec<u32> = (0..n as u32).collect();
        level.modularity(&init_comm, &level.degrees, config.resolution)
    };

    for _pass in 0..config.max_passes {
        let nl = level.num_nodes;
        if nl <= 1 {
            break;
        }

        let mut comm: Vec<u32> = (0..nl as u32).collect();
        let mut sigma_tot = level.degrees.clone();
        let mut order: Vec<u32> = (0..nl as u32).collect();
        let mut comm_stamp = VisitStamp::new(nl);
        let mut comm_weights = vec![0.0_f64; nl];
        let mut touched_comms: Vec<u32> = Vec::new();

        let inv_2m = 1.0 / level.total_weight_2m;
        let max_sweeps = nl.clamp(64, 500);
        let mut any_moved_in_pass = false;

        for _sweep in 0..max_sweeps {
            rng.shuffle(&mut order);
            let mut moved_in_sweep = false;

            for &u_u32 in &order {
                let u = u_u32 as usize;
                let k_u = level.degrees[u];
                let c_old = comm[u];
                let c_old_idx = c_old as usize;

                comm_stamp.clear();
                touched_comms.clear();

                let start = level.indptr[u] as usize;
                let end = level.indptr[u + 1] as usize;
                for pos in start..end {
                    let v = level.indices[pos] as usize;
                    let cv = comm[v];
                    let cv_idx = cv as usize;
                    if comm_stamp.visit(cv_idx) {
                        touched_comms.push(cv);
                    }
                    comm_weights[cv_idx] += level.weights[pos];
                }

                if touched_comms.is_empty() {
                    continue;
                }

                // Temporarily remove vertex `u` from its current community.
                sigma_tot[c_old_idx] -= k_u;
                let k_u_old = if comm_stamp.is_visited(c_old_idx) {
                    comm_weights[c_old_idx]
                } else {
                    0.0
                };
                // Proportional to modularity insertion gain relative to singleton {u}:
                // gain_term(c) = k_{u, in}(c) - gamma * k_u * sigma_tot[c] / (2m).
                let old_term = k_u_old - config.resolution * k_u * sigma_tot[c_old_idx] * inv_2m;

                let mut best_c = c_old;
                let mut best_delta_term = 0.0_f64;
                for &cand in &touched_comms {
                    if cand == c_old {
                        continue;
                    }
                    let cand_idx = cand as usize;
                    let cand_term = comm_weights[cand_idx]
                        - config.resolution * k_u * sigma_tot[cand_idx] * inv_2m;
                    let delta_term = cand_term - old_term;
                    if delta_term > best_delta_term + 1e-12
                        || ((delta_term - best_delta_term).abs() <= 1e-12
                            && delta_term > 1e-12
                            && cand < best_c)
                    {
                        best_delta_term = delta_term;
                        best_c = cand;
                    }
                }

                for &cand in &touched_comms {
                    comm_weights[cand as usize] = 0.0;
                }

                if best_c != c_old {
                    comm[u] = best_c;
                    sigma_tot[best_c as usize] += k_u;
                    moved_in_sweep = true;
                    any_moved_in_pass = true;
                } else {
                    sigma_tot[c_old_idx] += k_u;
                }
            }

            if !moved_in_sweep {
                break;
            }
        }

        if !any_moved_in_pass {
            break;
        }

        let next_q = level.modularity(&comm, &sigma_tot, config.resolution);
        if next_q - current_q <= config.min_modularity_gain {
            break;
        }
        current_q = next_q;
        passes += 1;

        let num_new = canonicalize_labels(&mut comm, nl);
        for g in &mut global_labels {
            *g = comm[*g as usize];
        }

        if num_new == nl {
            break;
        }
        level = level.aggregate(&comm, num_new);
    }

    let num_communities = canonicalize_labels(&mut global_labels, n);
    Ok(LouvainResult {
        labels: global_labels,
        num_communities,
        modularity: current_q,
        passes,
    })
}
