use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar};

use super::config::{
    CoarseLouvainConfig, CoarsenedGraph, CoarsenedHierarchy, CoarsenedHierarchyLevel,
};
use super::contract::coarsen_graph;
use crate::louvain::{LouvainConfig, louvain};

/// Runs multi-level Louvain coarsening with resolution decay until
/// `coarsest.num_clusters <= config.max_coarse_nodes` or the connected-component floor is reached.
pub fn louvain_coarsen<W: Scalar>(
    graph: &CsrGraph<W>,
    config: &CoarseLouvainConfig,
) -> CoreResult<CoarsenedHierarchy> {
    config.validate()?;
    if graph.is_directed() {
        return Err(CoreError::invalid_input(
            "louvain_coarsen requires an undirected graph",
        ));
    }

    let n = graph.num_nodes();
    if n == 0 {
        let coarsest = coarsen_graph(graph, &[])?;
        return Ok(CoarsenedHierarchy {
            levels: Vec::new(),
            coarsest,
        });
    }

    if n == 1 || graph.nnz() == 0 {
        let labels: Vec<u32> = (0..n as u32).collect();
        let coarse = coarsen_graph(graph, &labels)?;
        let level = CoarsenedHierarchyLevel {
            coarse: coarse.clone(),
            resolution: config.resolution,
            modularity: 0.0,
        };
        return Ok(CoarsenedHierarchy {
            levels: vec![level],
            coarsest: coarse,
        });
    }

    let mut cur_res = config.resolution;
    let mut cur_min_gain = config.min_modularity_gain;
    let l0_cfg = LouvainConfig {
        max_passes: config.max_passes,
        resolution: cur_res,
        min_modularity_gain: cur_min_gain,
        seed: config.seed,
    };
    let res0 = louvain(graph, &l0_cfg)?;
    let coarse0 = coarsen_graph(graph, &res0.labels)?;
    let mut fine_to_coarsest = coarse0.node_to_coarse.clone();
    let mut cumulative_sizes = coarse0.coarse_sizes.clone();
    let mut num_clusters = coarse0.num_clusters;
    let mut current_graph = coarse0.graph.clone();
    let mut levels = vec![CoarsenedHierarchyLevel {
        coarse: coarse0,
        resolution: cur_res,
        modularity: res0.modularity,
    }];

    let mut stalled_prev = num_clusters == n;
    for round in 1..config.max_outer_rounds {
        if num_clusters <= config.max_coarse_nodes || current_graph.num_edges() == 0 {
            break;
        }
        if stalled_prev {
            cur_res *= config.resolution_decay;
            if cur_res < 1e-6 {
                cur_res = 0.0;
                cur_min_gain = 0.0;
            }
        }
        let l_cfg = LouvainConfig {
            max_passes: config.max_passes,
            resolution: cur_res,
            min_modularity_gain: cur_min_gain,
            seed: config.seed.wrapping_add(round as u64),
        };
        let res = louvain(&current_graph, &l_cfg)?;
        if res.num_communities >= current_graph.num_nodes() {
            if cur_res == 0.0 && cur_min_gain == 0.0 {
                break;
            }
            stalled_prev = true;
            continue;
        }

        let coarse = coarsen_graph(&current_graph, &res.labels)?;
        for cid in &mut fine_to_coarsest {
            *cid = coarse.node_to_coarse[*cid as usize];
        }
        num_clusters = coarse.num_clusters;
        cumulative_sizes.clear();
        cumulative_sizes.resize(num_clusters, 0);
        for &cid in &fine_to_coarsest {
            cumulative_sizes[cid as usize] += 1;
        }
        current_graph = coarse.graph.clone();
        levels.push(CoarsenedHierarchyLevel {
            coarse,
            resolution: cur_res,
            modularity: res.modularity,
        });
        stalled_prev = false;
    }

    Ok(CoarsenedHierarchy {
        levels,
        coarsest: CoarsenedGraph {
            graph: current_graph,
            node_to_coarse: fine_to_coarsest,
            coarse_sizes: cumulative_sizes,
            num_clusters,
        },
    })
}
