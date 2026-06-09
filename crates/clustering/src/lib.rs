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
use petgraph::graph::{Graph, IndexType, NodeIndex};
use petgraph::visit::{
    EdgeCount, EdgeRef, GraphProp, IntoEdgeReferences, IntoNeighbors, IntoNodeIdentifiers,
};
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
/// * `graph` - A reference to the original graph.
/// * `node_groups` - A mutable closure `FnMut(G, G::NodeId) -> usize`.
///   It takes the original graph and a node identifier and returns the `usize` ID
///   of the group that node belongs to.
/// * `shrink_node` - A mutable closure `FnMut(G, &Vec<G::NodeId>) -> N2`.
///   It takes the original graph and a `Vec` of node identifiers belonging to a single
///   group and returns the node weight (`N2`) for the corresponding node in the
///   coarsened graph.
/// * `shrink_edge` - A mutable closure `FnMut(G, &Vec<G::EdgeId>) -> E2`.
///   It takes the original graph and a `Vec` of edge identifiers connecting two
///   specific groups and returns the edge weight (`E2`) for the corresponding
///   edge in the coarsened graph.
///
/// # Returns
///
/// A tuple containing:
/// 1. The coarsened graph `Graph<N2, E2, Ty, Ix>`.
/// 2. The `NodeMap<Ix>`: A `HashMap` mapping the group ID (`usize`) from the
///    original graph's grouping to the `NodeIndex` of the corresponding node
///    in the coarsened graph.
pub fn coarsen<G, N2, E2, Ty, Ix, GF, NF, EF>(
    graph: G,
    node_groups: &mut GF,
    shrink_node: &mut NF,
    shrink_edge: &mut EF,
) -> (Graph<N2, E2, Ty, Ix>, NodeMap<Ix>)
where
    G: IntoNodeIdentifiers + IntoEdgeReferences + GraphProp<EdgeType = Ty> + Copy,
    G::NodeId: Eq + Hash + Copy,
    G::EdgeId: Eq + Hash,
    Ty: EdgeType,
    Ix: IndexType,
    GF: FnMut(G, G::NodeId) -> usize,
    NF: FnMut(G, &Vec<G::NodeId>) -> N2,
    EF: FnMut(G, &Vec<G::EdgeId>) -> E2,
{
    let node_groups = graph
        .node_identifiers()
        .map(|u| (u, node_groups(graph, u)))
        .collect::<HashMap<_, _>>();
    let mut groups = HashMap::<usize, Vec<G::NodeId>>::new();
    for u in graph.node_identifiers() {
        let g = node_groups[&u];
        groups.entry(g).or_default().push(u);
    }
    let mut group_edges: HashMap<(usize, usize), Vec<G::EdgeId>> = HashMap::new();
    for e in graph.edge_references() {
        let u = e.source();
        let v = e.target();
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
        group_edges.entry(key).or_default().push(e.id());
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
