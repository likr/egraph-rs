use petgraph::unionfind::UnionFind;
use petgraph::visit::{EdgeRef, IntoEdgeReferences, IntoNodeIdentifiers, NodeCount};
use std::collections::HashMap;
use std::hash::Hash;

/// Finds the connected components of a given graph using the Union-Find algorithm.
///
/// This function iterates through the edges of the graph, uniting the components
/// of connected nodes. It then maps each node index to the representative ID of
/// the component it belongs to.
///
/// # Arguments
///
/// * `graph` - A reference to a graph implementing required traits. The graph can be directed or
///   undirected, and node/edge weights are generic.
///
/// # Returns
///
/// A `HashMap` where keys are `NodeId` and values are `usize` component identifiers.
/// Nodes within the same connected component will have the same component identifier.
/// The identifier corresponds to the representative element of the set in the Union-Find structure.
///
/// # Examples
///
/// ```
/// use petgraph::graph::Graph;
/// use petgraph_algorithm_connected_components::connected_components;
///
/// let mut graph = Graph::new_undirected();
/// let u1 = graph.add_node(());
/// let u2 = graph.add_node(());
/// let u3 = graph.add_node(());
/// let u4 = graph.add_node(());
/// graph.add_edge(u1, u2, ());
/// graph.add_edge(u2, u3, ());
/// // u4 is isolated
///
/// let components = connected_components(&graph);
///
/// assert_eq!(components[&u1], components[&u2]);
/// assert_eq!(components[&u2], components[&u3]);
/// assert_ne!(components[&u1], components[&u4]);
/// assert_eq!(components.values().collect::<std::collections::HashSet<_>>().len(), 2); // 2 components
/// ```
pub fn connected_components<G>(graph: G) -> HashMap<G::NodeId, usize>
where
    G: NodeCount + IntoNodeIdentifiers + IntoEdgeReferences,
    G::NodeId: Eq + Hash,
{
    let mut components = UnionFind::new(graph.node_count());
    let indices = graph
        .node_identifiers()
        .enumerate()
        .map(|(i, u)| (u, i))
        .collect::<HashMap<G::NodeId, usize>>();
    for e in graph.edge_references() {
        components.union(indices[&e.source()], indices[&e.target()]);
    }
    let mut result = HashMap::new();
    for u in graph.node_identifiers() {
        result.insert(u, components.find(indices[&u]));
    }
    result
}

#[cfg(test)]
mod test {
    use super::*;
    use petgraph::Graph;

    #[test]
    fn test_connected_components() {
        let mut graph = Graph::new_undirected();
        let u1 = graph.add_node(());
        let u2 = graph.add_node(());
        let u3 = graph.add_node(());
        let u4 = graph.add_node(());
        let u5 = graph.add_node(());
        graph.add_edge(u1, u2, ());
        graph.add_edge(u1, u3, ());
        graph.add_edge(u2, u3, ());
        graph.add_edge(u4, u5, ());
        let components = connected_components(&graph);
        assert_eq!(components[&u1], components[&u2]);
        assert_eq!(components[&u1], components[&u3]);
        assert_ne!(components[&u3], components[&u4]);
        assert_eq!(components[&u4], components[&u5]);
    }
}
