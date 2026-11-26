use std::collections::VecDeque;

use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, Scalar};

use super::config::{LeidenConfig, LeidenObjective, LeidenResult};
use super::level_graph::{LeidenLevelGraph, canonicalize_labels};
use super::local_move::move_nodes_fast;
use super::refine::refine_partition;

/// Splits any disconnected community into its positive-weight connected components in $O(V + E)$
/// and returns the canonical community count `0..num_communities`.
fn split_disconnected_communities<W: Scalar>(graph: &CsrGraph<W>, labels: &mut [u32]) -> usize {
    let n = graph.num_nodes();
    if n == 0 {
        return 0;
    }

    let mut visited = vec![false; n];
    let mut new_labels = vec![0u32; n];
    let mut next_comm = 0u32;
    let mut queue = VecDeque::new();
    let weights = graph.weights();

    for start in 0..n {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let target_label = labels[start];
        let comp_id = next_comm;
        next_comm += 1;
        new_labels[start] = comp_id;
        queue.push_back(start as u32);

        while let Some(u_u32) = queue.pop_front() {
            let u = u_u32 as usize;
            let row_start = graph.indptr()[u] as usize;
            let row_end = graph.indptr()[u + 1] as usize;
            for pos in row_start..row_end {
                if let Some(ws) = weights
                    && ws[pos].to_f64() <= 0.0
                {
                    continue;
                }
                let v = graph.indices()[pos] as usize;
                if !visited[v] && labels[v] == target_label {
                    visited[v] = true;
                    new_labels[v] = comp_id;
                    queue.push_back(v as u32);
                }
            }
        }
    }

    labels.copy_from_slice(&new_labels);
    next_comm as usize
}

/// Computes exact `(quality, modularity)` directly on the base graph for canonical `labels`.
fn evaluate_base_partition<W: Scalar>(
    graph: &CsrGraph<W>,
    labels: &[u32],
    num_communities: usize,
    objective: LeidenObjective,
    resolution: f64,
) -> (f64, f64) {
    let n = graph.num_nodes();
    let mut sigma_in = vec![0.0_f64; num_communities];
    let mut sigma_tot = vec![0.0_f64; num_communities];
    let mut comm_counts = vec![0usize; num_communities];
    let mut total_weight_2m = 0.0_f64;

    if let Some(ws) = graph.weights() {
        for u in 0..n {
            let cu = labels[u] as usize;
            comm_counts[cu] += 1;
            let start = graph.indptr()[u] as usize;
            let end = graph.indptr()[u + 1] as usize;
            for pos in start..end {
                let wf = ws[pos].to_f64();
                sigma_tot[cu] += wf;
                total_weight_2m += wf;
                if labels[graph.indices()[pos] as usize] as usize == cu {
                    sigma_in[cu] += wf;
                }
            }
        }
    } else {
        for u in 0..n {
            let cu = labels[u] as usize;
            comm_counts[cu] += 1;
            let start = graph.indptr()[u] as usize;
            let end = graph.indptr()[u + 1] as usize;
            let deg = (end - start) as f64;
            sigma_tot[cu] += deg;
            total_weight_2m += deg;
            for pos in start..end {
                if labels[graph.indices()[pos] as usize] as usize == cu {
                    sigma_in[cu] += 1.0;
                }
            }
        }
    }

    let modularity = if total_weight_2m <= 0.0 {
        0.0
    } else {
        let mut q = 0.0_f64;
        for c in 0..num_communities {
            let st = sigma_tot[c];
            if st > 0.0 || sigma_in[c] > 0.0 {
                q += (sigma_in[c] / total_weight_2m) - resolution * (st / total_weight_2m).powi(2);
            }
        }
        q
    };

    let quality = match objective {
        LeidenObjective::Modularity => modularity,
        LeidenObjective::Cpm => {
            let mut h = 0.0_f64;
            for c in 0..num_communities {
                let sz = comm_counts[c] as f64;
                h += 0.5 * sigma_in[c] - resolution * (sz * (sz - 1.0) * 0.5);
            }
            h
        }
    };

    (quality, modularity)
}

/// Runs multi-level Leiden community detection on an undirected [`CsrGraph`] following
/// Traag, Waltman, and van Eck (2019), Algorithm A.2.
///
/// Supports both [`LeidenObjective::Modularity`] and [`LeidenObjective::Cpm`], guarantees that
/// every returned community's positive-weight induced subgraph is internally connected, and
/// returns canonical `0..num_communities` labels in first-appearance vertex order.
pub fn leiden<W: Scalar>(graph: &CsrGraph<W>, config: &LeidenConfig) -> CoreResult<LeidenResult> {
    if graph.is_directed() {
        return Err(CoreError::invalid_input(
            "leiden requires an undirected graph",
        ));
    }
    if config.max_passes == 0 {
        return Err(CoreError::invalid_input("max_passes must be >= 1"));
    }
    if !config.resolution.is_finite() {
        return Err(CoreError::non_finite("LeidenConfig::resolution"));
    }
    if config.resolution < 0.0 {
        return Err(CoreError::invalid_input("resolution must be >= 0.0"));
    }
    if !config.randomness.is_finite() {
        return Err(CoreError::non_finite("LeidenConfig::randomness"));
    }
    if config.randomness <= 0.0 {
        return Err(CoreError::invalid_input("randomness must be > 0.0"));
    }
    if !config.min_quality_gain.is_finite() {
        return Err(CoreError::non_finite("LeidenConfig::min_quality_gain"));
    }
    if config.min_quality_gain < 0.0 {
        return Err(CoreError::invalid_input("min_quality_gain must be >= 0.0"));
    }

    let n = graph.num_nodes();
    if n == 0 {
        return Ok(LeidenResult {
            labels: Vec::new(),
            num_communities: 0,
            quality: 0.0,
            modularity: 0.0,
            passes: 0,
        });
    }

    let base_level = LeidenLevelGraph::from_csr(graph, config.objective)?;
    let mut best_labels: Vec<u32> = (0..n as u32).collect();

    if base_level.total_weight_2m <= 0.0 {
        return Ok(LeidenResult {
            labels: best_labels,
            num_communities: n,
            quality: 0.0,
            modularity: 0.0,
            passes: 0,
        });
    }

    let gamma_eff = match config.objective {
        LeidenObjective::Modularity => config.resolution / base_level.total_weight_2m,
        LeidenObjective::Cpm => config.resolution,
    };
    if !gamma_eff.is_finite() {
        return Err(CoreError::numerical_breakdown(
            "effective resolution overflows f64",
        ));
    }

    let mut rng = DeterministicRng::from_seed_u64(config.seed);
    let mut best_quality =
        base_level.evaluate_quality(&best_labels, n, config.objective, config.resolution);
    if !best_quality.is_finite() {
        return Err(CoreError::numerical_breakdown(
            "initial leiden quality is not finite",
        ));
    }
    let mut passes = 0usize;
    let mut total_steps = 0usize;
    let mut outer_iter = 0usize;

    while total_steps < config.max_passes {
        let outer_start_quality = best_quality;
        let mut level = base_level.clone();
        let mut base_to_super: Vec<u32> = (0..n as u32).collect();
        let mut comm = best_labels.clone();
        let mut current_q = best_quality;
        let mut level_depth = 0usize;

        while total_steps < config.max_passes {
            let nl = level.num_nodes;
            if nl <= 1 {
                break;
            }

            total_steps += 1;
            let any_moved = move_nodes_fast(&level, &mut comm, gamma_eff, &mut rng);
            let mut num_outer = canonicalize_labels(&mut comm, nl);

            if any_moved {
                let next_q =
                    level.evaluate_quality(&comm, num_outer, config.objective, config.resolution);
                if !next_q.is_finite() {
                    return Err(CoreError::numerical_breakdown(
                        "leiden intermediate quality is not finite",
                    ));
                }
                if next_q - current_q > config.min_quality_gain {
                    current_q = next_q;
                    passes += 1;
                    best_quality = next_q;
                    for i in 0..n {
                        best_labels[i] = comm[base_to_super[i] as usize];
                    }
                } else if level_depth > 0 || outer_iter == 0 {
                    break;
                } else if next_q > best_quality {
                    current_q = next_q;
                    best_quality = next_q;
                    for i in 0..n {
                        best_labels[i] = comm[base_to_super[i] as usize];
                    }
                } else {
                    comm.copy_from_slice(&best_labels);
                    num_outer = canonicalize_labels(&mut comm, nl);
                }
            } else if level_depth > 0 || outer_iter == 0 {
                break;
            }

            if num_outer == nl || total_steps >= config.max_passes {
                break;
            }

            let (refined_comm, num_refined) = refine_partition(
                &level,
                &comm,
                num_outer,
                gamma_eff,
                config.randomness,
                &mut rng,
            );
            if num_refined == nl {
                break;
            }

            let mut lifted_comm = vec![0u32; num_refined];
            for u in 0..nl {
                lifted_comm[refined_comm[u] as usize] = comm[u];
            }
            for s in &mut base_to_super {
                *s = refined_comm[*s as usize];
            }

            level = level.aggregate(&refined_comm, num_refined);
            comm = lifted_comm;
            level_depth += 1;
        }

        if best_quality - outer_start_quality <= config.min_quality_gain {
            break;
        }
        outer_iter += 1;
    }

    let num_communities = split_disconnected_communities(graph, &mut best_labels);
    let (quality, modularity) = evaluate_base_partition(
        graph,
        &best_labels,
        num_communities,
        config.objective,
        config.resolution,
    );
    if !quality.is_finite() || !modularity.is_finite() {
        return Err(CoreError::numerical_breakdown(
            "leiden quality or modularity is not finite",
        ));
    }

    Ok(LeidenResult {
        labels: best_labels,
        num_communities,
        quality,
        modularity,
        passes,
    })
}
