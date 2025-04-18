use fixedbitset::FixedBitSet;
use petgraph::graph::{IndexType, NodeIndex};
use petgraph::visit::{VisitMap, Visitable};
use petgraph::{Directed, EdgeDirection, Graph};
use std::collections::{HashMap, HashSet};

use crate::Constraint;

// Cycle Removal Implementation
// Based on: https://github.com/likr/egraph-rs/blob/cbb2e93199c2c7c2b968bbc50b46dc6a60097a3a/crates/egraph-layered/src/cycle_removal.rs

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

pub fn cycle_edges<N, E, Ix: IndexType>(
    graph: &Graph<N, E, Directed, Ix>,
) -> Vec<(NodeIndex<Ix>, NodeIndex<Ix>)> {
    let mut map = graph.visit_map();
    let mut ancestors = HashSet::new();
    let mut result = vec![];
    for u in graph.node_indices() {
        dfs_cycle(&graph, &mut map, &mut ancestors, &mut result, u)
    }
    result
}

pub fn remove_cycle<N, E, Ix: IndexType>(graph: &mut Graph<N, E, Directed, Ix>) {
    for (u, v) in cycle_edges(graph) {
        let index = graph.find_edge(u, v).unwrap();
        let weight = graph.remove_edge(index).unwrap();
        graph.add_edge(v, u, weight);
    }
}

// Layer Assignment Implementation
// Based on: https://github.com/likr/egraph-rs/blob/eff56113818199eab6a1c9e50a9e1894b1b90e55/egraph-layered/src/ranking/longest_path.rs

fn dfs_layer<N, E, Ix: IndexType>(
    graph: &Graph<N, E, Directed, Ix>,
    layers: &mut HashMap<NodeIndex<Ix>, usize>,
    u: NodeIndex<Ix>,
    depth: usize,
) {
    for v in graph.neighbors(u) {
        if layers.contains_key(&v) {
            let layer = layers.get_mut(&v).unwrap();
            if *layer <= depth {
                *layer = depth + 1
            }
        } else {
            layers.insert(v, depth + 1);
        }
        dfs_layer(graph, layers, v, depth + 1);
    }
}

pub fn longest_path<N, E, Ix: IndexType>(
    graph: &Graph<N, E, Directed, Ix>,
) -> HashMap<NodeIndex<Ix>, usize> {
    let mut result = HashMap::new();
    for u in graph.externals(EdgeDirection::Incoming) {
        result.insert(u, 0);
        dfs_layer(graph, &mut result, u, 0);
    }
    result
}

/// Generates layered constraints for a directed graph based on the Sugiyama Framework.
///
/// This function performs cycle removal and layer assignment to create a hierarchical
/// layout, then generates separation constraints for edges that span multiple layers.
///
/// # Arguments
///
/// * `graph` - A reference to a directed graph.
/// * `min_layer_distance` - The minimum vertical distance between adjacent layers.
///
/// # Returns
///
/// A vector of `Constraint` objects representing the separation constraints.
///
/// # Type Parameters
///
/// * `N` - The node data type.
/// * `E` - The edge data type.
/// * `Ix` - The index type.
///
/// # Example
///
/// ```
/// use petgraph::Graph;
/// use petgraph_layout_separation_constraints::{generate_layered_constraints, project_1d};
/// use petgraph_drawing::DrawingEuclidean2d;
///
/// // Create a directed graph
/// let mut graph = Graph::new();
/// let n1 = graph.add_node(());
/// let n2 = graph.add_node(());
/// let n3 = graph.add_node(());
/// graph.add_edge(n1, n2, ());
/// graph.add_edge(n2, n3, ());
///
/// // Create a drawing with explicit type annotations
/// let mut drawing = DrawingEuclidean2d::initial_placement(&graph);
///
/// // Generate layered constraints
/// let constraints = generate_layered_constraints(&graph, 1.0);
///
/// // Apply constraints to the y-dimension (vertical)
/// project_1d(&mut drawing, 1, &constraints);
///
/// assert_eq!(drawing.y(n2).unwrap() - drawing.y(n1).unwrap(), 1.);
/// assert_eq!(drawing.y(n3).unwrap() - drawing.y(n2).unwrap(), 1.);
/// assert_eq!(drawing.y(n3).unwrap() - drawing.y(n1).unwrap(), 2.);
/// ```
pub fn generate_layered_constraints<N, E, Ix: IndexType>(
    graph: &Graph<N, E, Directed, Ix>,
    min_layer_distance: f32,
) -> Vec<Constraint> {
    // Create a clone of the graph to perform cycle removal
    let mut acyclic_graph = graph.map(|_, _| (), |_, _| ());

    // Remove cycles to make the graph acyclic
    remove_cycle(&mut acyclic_graph);

    // Assign layers to nodes using longest path algorithm
    let layers = longest_path(&acyclic_graph);

    // Generate constraints for edges that span multiple layers
    let mut constraints = Vec::new();

    // Create a mapping from node indices to their positions in the drawing
    let mut node_indices = HashMap::new();
    for (idx, node) in graph.node_indices().enumerate() {
        node_indices.insert(node, idx);
    }

    // For each edge in the acyclic graph
    for edge in acyclic_graph.edge_indices() {
        let (source, target) = acyclic_graph.edge_endpoints(edge).unwrap();

        // Get the layer of source and target nodes
        let source_layer = *layers.get(&source).unwrap_or(&0);
        let target_layer = *layers.get(&target).unwrap_or(&0);

        // If the edge spans multiple layers, create a constraint
        if source_layer < target_layer {
            // Get the indices of source and target nodes in the drawing
            let source_idx = node_indices[&source];
            let target_idx = node_indices[&target];

            // Create a constraint to ensure the target node is below the source node
            // The gap is proportional to the number of layers spanned
            constraints.push(Constraint::new(
                source_idx,
                target_idx,
                min_layer_distance * (target_layer - source_layer) as f32,
            ));
        }
    }

    constraints
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::Graph;

    #[test]
    fn test_cycle_edges() {
        let mut graph = Graph::<&str, &str>::new();
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        graph.add_edge(a, b, "");
        graph.add_edge(b, c, "");
        graph.add_edge(c, a, "");
        assert_eq!(cycle_edges(&graph), vec![(c, a)]);
    }

    #[test]
    fn test_remove_cycle() {
        let mut graph = Graph::<&str, &str>::new();
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        graph.add_edge(a, b, "");
        graph.add_edge(b, c, "");
        graph.add_edge(c, a, "");
        remove_cycle(&mut graph);
        assert!(graph.find_edge(a, b).is_some());
        assert!(graph.find_edge(a, c).is_some());
        assert!(graph.find_edge(b, c).is_some());
        assert_eq!(graph.edge_count(), 3);
    }

    #[test]
    fn test_longest_path() {
        let mut graph = Graph::<&str, &str>::new();
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        let d = graph.add_node("d");
        let e = graph.add_node("e");
        graph.add_edge(a, b, "");
        graph.add_edge(b, c, "");
        graph.add_edge(d, c, "");
        graph.add_edge(d, e, "");
        let layers = longest_path(&graph);
        assert_eq!(*layers.get(&a).unwrap(), 0);
        assert_eq!(*layers.get(&b).unwrap(), 1);
        assert_eq!(*layers.get(&c).unwrap(), 2);
        assert_eq!(*layers.get(&d).unwrap(), 0);
        assert_eq!(*layers.get(&e).unwrap(), 1);
    }

    #[test]
    fn test_generate_layered_constraints() {
        // Create a simple directed graph
        let mut graph = Graph::<(), ()>::new();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());

        // Add edges to form a path: a -> b -> c
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());

        // Generate layered constraints with a minimum distance of 1.0
        let constraints = generate_layered_constraints(&graph, 1.0);

        // We should have 2 constraints: a -> b and b -> c
        assert_eq!(constraints.len(), 2);

        // Check the first constraint (a -> b)
        assert_eq!(constraints[0].left, 0);
        assert_eq!(constraints[0].right, 1);
        assert_eq!(constraints[0].gap, 1.0);

        // Check the second constraint (b -> c)
        assert_eq!(constraints[1].left, 1);
        assert_eq!(constraints[1].right, 2);
        assert_eq!(constraints[1].gap, 1.0);
    }

    #[test]
    fn test_generate_layered_constraints_with_cycle() {
        // Create a directed graph with a cycle
        let mut graph = Graph::<(), ()>::new();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());

        // Add edges to form a cycle: a -> b -> c -> a
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());
        graph.add_edge(c, a, ());

        // Generate layered constraints with a minimum distance of 1.5
        let mut constraints = generate_layered_constraints(&graph, 1.5);

        // After cycle removal, we should have 3 constraints
        assert_eq!(constraints.len(), 3);

        // Check that all constraints have gaps that are multiples of the minimum layer distance
        constraints.sort_by(|a, b| a.gap.partial_cmp(&b.gap).unwrap());
        assert_eq!(constraints[0].gap, 1.5);
        assert_eq!(constraints[1].gap, 1.5);
        assert_eq!(constraints[2].gap, 3.0);
    }
}
