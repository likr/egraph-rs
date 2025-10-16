//! petgraph-clustering is a library for detecting communities in graphs.
//!
//! This crate provides algorithms for community detection (clustering) in graphs,
//! implemented for the `petgraph` library. It includes several algorithms with
//! a common interface that makes it easy to try different approaches.

mod algorithms;
mod utils;

pub use algorithms::*;
pub use utils::*;

use petgraph::EdgeType;
use petgraph::graph::{EdgeIndex, Graph, IndexType, NodeIndex};
use petgraph::visit::{EdgeCount, IntoNeighbors, IntoNodeIdentifiers};
use std::collections::HashMap;
use std::hash::Hash;

/// Trait for community detection algorithms.
///
/// Implementations of this trait provide methods to detect communities in graphs.
/// All algorithms return a mapping from node identifiers to community IDs (as usize).
pub trait CommunityDetection<G>
where
    G: IntoNodeIdentifiers + EdgeCount + IntoNeighbors,
    G::NodeId: Eq + Hash + Clone,
{
    /// Detect communities in the input graph
    ///
    /// # Returns
    ///
    /// A `HashMap` mapping each node's `NodeId` to its community ID (as usize)
    fn detect_communities(&self, graph: G) -> HashMap<G::NodeId, usize>;
}

type CoarsenedGraph<N2, E2, Ty, Ix> = Graph<N2, E2, Ty, Ix>;
type NodeMap<Ix> = HashMap<usize, NodeIndex<Ix>>;

/// Creates a coarser graph representation by grouping nodes from an original graph.
///
/// This function takes an input graph and grouping information to produce a new,
/// smaller graph where each node represents a group of nodes from the original graph.
/// Edges in the coarsened graph represent the connections between groups in the
/// original graph.
///
/// # Arguments
///
/// * `graph` - A reference to the original `petgraph::Graph`.
/// * `node_groups` - A mutable closure `FnMut(&Graph<...>, NodeIndex<Ix>) -> usize`.
///   It takes the original graph and a `NodeIndex` and returns the `usize` ID
///   of the group that node belongs to.
/// * `shrink_node` - A mutable closure `FnMut(&Graph<...>, &Vec<NodeIndex<Ix>>) -> N2`.
///   It takes the original graph and a `Vec` of `NodeIndex` belonging to a single
///   group and returns the node weight (`N2`) for the corresponding node in the
///   coarsened graph.
/// * `shrink_edge` - A mutable closure `FnMut(&Graph<...>, &Vec<EdgeIndex<Ix>>) -> E2`.
///   It takes the original graph and a `Vec` of `EdgeIndex` connecting two
///   specific groups and returns the edge weight (`E2`) for the corresponding
///   edge in the coarsened graph.
///
/// # Returns
///
/// A tuple containing:
/// 1. The `CoarsenedGraph<N2, E2, Ty, Ix>`: The newly created `petgraph::Graph`
///    representing the coarsened structure.
/// 2. The `NodeMap<Ix>`: A `HashMap` mapping the group ID (`usize`) from the
///    original graph's grouping to the `NodeIndex` of the corresponding node
///    in the coarsened graph.
pub fn coarsen<
    N1,
    N2,
    E1,
    E2,
    Ty: EdgeType,
    Ix: IndexType,
    GF: FnMut(&Graph<N1, E1, Ty, Ix>, NodeIndex<Ix>) -> usize,
    NF: FnMut(&Graph<N1, E1, Ty, Ix>, &Vec<NodeIndex<Ix>>) -> N2,
    EF: FnMut(&Graph<N1, E1, Ty, Ix>, &Vec<EdgeIndex<Ix>>) -> E2,
>(
    graph: &Graph<N1, E1, Ty, Ix>,
    node_groups: &mut GF,
    shrink_node: &mut NF,
    shrink_edge: &mut EF,
) -> (CoarsenedGraph<N2, E2, Ty, Ix>, NodeMap<Ix>) {
    let node_groups = graph
        .node_indices()
        .map(|u| (u, node_groups(graph, u)))
        .collect::<HashMap<_, _>>();
    let mut groups = HashMap::<usize, Vec<NodeIndex<Ix>>>::new();
    for u in graph.node_indices() {
        let g = node_groups[&u];
        groups.entry(g).or_default().push(u);
    }
    let mut group_edges: HashMap<(usize, usize), Vec<EdgeIndex<Ix>>> = HashMap::new();
    for e in graph.edge_indices() {
        let (u, v) = graph.edge_endpoints(e).unwrap();
        let key = {
            let source_group = node_groups[&u];
            let target_group = node_groups[&v];
            if source_group == target_group {
                continue;
            }
            if source_group < target_group {
                (source_group, target_group)
            } else {
                (target_group, source_group)
            }
        };
        group_edges.entry(key).or_default().push(e);
    }

    let mut coarsened_graph = Graph::with_capacity(0, 0);
    let mut coarsened_node_ids = HashMap::new();
    for (&group_id, node_ids) in groups.iter() {
        coarsened_node_ids.insert(
            group_id,
            coarsened_graph.add_node(shrink_node(graph, node_ids)),
        );
    }
    for (&(u, v), edge_ids) in group_edges.iter() {
        coarsened_graph.add_edge(
            coarsened_node_ids[&u],
            coarsened_node_ids[&v],
            shrink_edge(graph, edge_ids),
        );
    }
    (coarsened_graph, coarsened_node_ids)
}

// For backward compatibility
pub use algorithms::louvain::louvain_step_legacy as louvain_step;
