use petgraph::graph::{EdgeIndex, Graph, IndexType, NodeIndex};
use petgraph::visit::{EdgeCount, IntoNeighbors, IntoNodeIdentifiers};
use petgraph::EdgeType;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// Performs one step of the Louvain algorithm for community detection.
///
/// This function iterates through each node in the graph and evaluates the
/// modularity gain of moving the node to one of its neighboring communities.
/// If moving a node increases the overall modularity, the node's community
/// assignment is updated.
///
/// # Arguments
///
/// * `graph` - A reference to a graph `G` implementing `petgraph`'s `EdgeCount`,
///   `IntoNeighbors`, and `IntoNodeIdentifiers` traits. `G::NodeId` must
///   implement `Eq` and `Hash`.
///
/// # Returns
///
/// * `Some(communities)` - If at least one node was moved to a different
///   community (improving modularity), returns a `HashMap` mapping each node's
///   `NodeId` to the `NodeId` representing its assigned community.
/// * `None` - If no node movement improved modularity during this step.
pub fn louvain_step<G>(graph: &G) -> Option<HashMap<G::NodeId, G::NodeId>>
where
    G: EdgeCount + IntoNeighbors + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash,
{
    let m = graph.edge_count() as f32;
    let k = graph
        .node_identifiers()
        .map(|u| (u, graph.neighbors(u).count() as f32))
        .collect::<HashMap<_, _>>();
    let mut sigma_total = k.clone();
    let mut communities = graph
        .node_identifiers()
        .map(|u| (u, u))
        .collect::<HashMap<_, _>>();
    let mut community_nodes = graph
        .node_identifiers()
        .map(|u| (u, HashSet::new()))
        .collect::<HashMap<_, _>>();
    let mut improve = false;

    for u in graph.node_identifiers() {
        let mut neighboring_communities = HashSet::new();
        for v in graph.neighbors(u) {
            neighboring_communities.insert(communities[&v]);
        }
        neighboring_communities.remove(&communities[&u]);
        for &c in neighboring_communities.iter() {
            let prev_c = communities[&u];
            community_nodes.get_mut(&prev_c).unwrap().remove(&u);

            let mut k_in = 0.;
            for v in graph.neighbors(u) {
                if communities[&v] == c {
                    k_in += 1.;
                }
            }
            let delta_q = 0.5 * (k_in - k[&u] * sigma_total[&c] / m) / m;
            if delta_q > 0. {
                *sigma_total.get_mut(&c).unwrap() += k[&u];
                *sigma_total.get_mut(&prev_c).unwrap() -= k[&u];
                *communities.get_mut(&u).unwrap() = c;
                community_nodes.get_mut(&c).unwrap().insert(u);
                improve = true;
            } else {
                community_nodes.get_mut(&prev_c).unwrap().insert(u);
            }
        }
    }
    if improve {
        Some(communities)
    } else {
        None
    }
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
