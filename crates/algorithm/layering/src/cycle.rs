use fixedbitset::FixedBitSet;
use petgraph::graph::{IndexType, NodeIndex};
use petgraph::visit::{VisitMap, Visitable};
use petgraph::{Directed, Graph};
use std::collections::HashSet;

/// Performs a depth-first search to detect cycles in a directed graph.
///
/// This function is used internally by `cycle_edges` to identify all edges that
/// form part of a cycle in the graph.
///
/// # Arguments
///
/// * `graph` - The directed graph to search
/// * `map` - A bit set to track visited nodes
/// * `ancestors` - A set to track ancestor nodes in the current search path
/// * `result` - A vector to accumulate the cycle edges
/// * `u` - The current node being visited
fn dfs_cycle<N, E, Ix: IndexType>(
    graph: &Graph<N, E, Directed, Ix>,
    map: &mut FixedBitSet,
    ancestors: &mut HashSet<NodeIndex<Ix>>,
    result: &mut Vec<(NodeIndex<Ix>, NodeIndex<Ix>)>,
    u: NodeIndex<Ix>,
) {
    if map.is_visited(&u) {
        return;
    }
    map.visit(u);
    ancestors.insert(u);
    for v in graph.neighbors(u) {
        if ancestors.contains(&v) {
            result.push((u, v));
        } else {
            dfs_cycle(graph, map, ancestors, result, v)
        }
    }
    ancestors.remove(&u);
}

/// Identifies all edges that form cycles in a directed graph.
///
/// This function performs a depth-first search from each node in the graph
/// to identify edges that complete a cycle.
///
/// # Arguments
///
/// * `graph` - The directed graph to analyze
///
/// # Returns
///
/// A vector of tuples `(u, v)` where `(u, v)` is an edge that completes a cycle.
///
/// # Examples
///
/// ```
/// use petgraph::Graph;
/// use petgraph_algorithm_layering::cycle::cycle_edges;
///
/// let mut graph = Graph::new();
/// let a = graph.add_node("a");
/// let b = graph.add_node("b");
/// let c = graph.add_node("c");
/// graph.add_edge(a, b, "");
/// graph.add_edge(b, c, "");
/// graph.add_edge(c, a, "");
///
/// let cycle_edges = cycle_edges(&graph);
/// assert_eq!(cycle_edges.len(), 1);
/// ```
pub fn cycle_edges<N, E, Ix: IndexType>(
    graph: &Graph<N, E, Directed, Ix>,
) -> Vec<(NodeIndex<Ix>, NodeIndex<Ix>)> {
    let mut map = graph.visit_map();
    let mut ancestors = HashSet::new();
    let mut result = vec![];
    for u in graph.node_indices() {
        dfs_cycle(graph, &mut map, &mut ancestors, &mut result, u)
    }
    result
}

/// Removes all cycles from a directed graph.
///
/// This function identifies edges that form cycles and reverses their direction
/// to make the graph acyclic. The original edge weights are preserved.
///
/// # Arguments
///
/// * `graph` - The directed graph to modify
///
/// # Examples
///
/// ```
/// use petgraph::Graph;
/// use petgraph_algorithm_layering::cycle::remove_cycle;
///
/// let mut graph = Graph::new();
/// let a = graph.add_node("a");
/// let b = graph.add_node("b");
/// let c = graph.add_node("c");
/// graph.add_edge(a, b, "");
/// graph.add_edge(b, c, "");
/// graph.add_edge(c, a, "");
///
/// remove_cycle(&mut graph);
/// // Now the graph is acyclic
/// assert!(graph.find_edge(a, b).is_some());
/// assert!(graph.find_edge(b, c).is_some());
/// assert!(graph.find_edge(a, c).is_some() || graph.find_edge(c, a).is_some());
/// ```
pub fn remove_cycle<N, E, Ix: IndexType>(graph: &mut Graph<N, E, Directed, Ix>) {
    for (u, v) in cycle_edges(graph) {
        let index = graph.find_edge(u, v).unwrap();
        let weight = graph.remove_edge(index).unwrap();
        graph.add_edge(v, u, weight);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::Graph;

    #[test]
    fn test_cycle_edges_simple() {
        let mut graph = Graph::<&str, &str>::new();
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        graph.add_edge(a, b, "");
        graph.add_edge(b, c, "");
        graph.add_edge(c, a, "");
        assert_eq!(cycle_edges(&graph).len(), 1);
    }

    #[test]
    fn test_cycle_edges_multiple_cycles() {
        let mut graph = Graph::<&str, &str>::new();
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        let d = graph.add_node("d");

        // Create two cycles: a->b->c->a and b->d->b
        graph.add_edge(a, b, "");
        graph.add_edge(b, c, "");
        graph.add_edge(c, a, "");
        graph.add_edge(b, d, "");
        graph.add_edge(d, b, "");

        let edges = cycle_edges(&graph);
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn test_cycle_edges_no_cycles() {
        let mut graph = Graph::<&str, &str>::new();
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        graph.add_edge(a, b, "");
        graph.add_edge(b, c, "");

        let edges = cycle_edges(&graph);
        assert_eq!(edges.len(), 0);
    }

    #[test]
    fn test_remove_cycle_simple() {
        let mut graph = Graph::<&str, &str>::new();
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        graph.add_edge(a, b, "");
        graph.add_edge(b, c, "");
        graph.add_edge(c, a, "");

        remove_cycle(&mut graph);

        // After cycle removal, the graph should be acyclic
        assert_eq!(cycle_edges(&graph).len(), 0);

        // The original edges should still exist, though some may be reversed
        assert!(graph.find_edge(a, b).is_some());
        assert!(graph.find_edge(b, c).is_some());
        assert!(graph.find_edge(a, c).is_some() || graph.find_edge(c, a).is_some());
        assert_eq!(graph.edge_count(), 3);
    }

    #[test]
    fn test_remove_cycle_complex() {
        let mut graph = Graph::<&str, &str>::new();
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        let d = graph.add_node("d");
        let e = graph.add_node("e");

        // Create a complex graph with multiple cycles
        graph.add_edge(a, b, "");
        graph.add_edge(b, c, "");
        graph.add_edge(c, a, ""); // First cycle
        graph.add_edge(c, d, "");
        graph.add_edge(d, e, "");
        graph.add_edge(e, c, ""); // Second cycle

        remove_cycle(&mut graph);

        // After cycle removal, the graph should be acyclic
        assert_eq!(cycle_edges(&graph).len(), 0);

        // The edge count should remain the same
        assert_eq!(graph.edge_count(), 6);
    }
}
