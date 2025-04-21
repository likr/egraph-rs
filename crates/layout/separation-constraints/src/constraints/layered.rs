use petgraph::graph::IndexType;
use petgraph::{Directed, Graph};
use petgraph_algorithm_layering::{cycle::remove_cycle, LongestPath};
use std::collections::HashMap;

use crate::Constraint;

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
    let longest_path = LongestPath::new();
    let layers = longest_path.assign_layers(&acyclic_graph);

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
