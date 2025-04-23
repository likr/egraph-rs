use petgraph::{
    graph::{IndexType, NodeIndex},
    visit::IntoNodeIdentifiers,
    EdgeType, Graph,
};
use petgraph_clustering::coarsen;
use petgraph_drawing::{Drawing, DrawingEuclidean2d};
use std::collections::HashMap;

use crate::{generate_rectangle_no_overlap_constraints_triangulated, project_1d};

/// Removes overlaps between rectangular regions that represent clusters of nodes.
///
/// This function takes a graph, a drawing, a function to get cluster IDs for nodes,
/// and a function to get node sizes. It then:
/// 1. Creates a cluster graph using `petgraph_clustering::coarsen`
/// 2. Determines the size of each cluster based on the nodes it contains
/// 3. Uses `generate_rectangle_no_overlap_constraints` and `project_1d` to remove overlaps
/// 4. Updates the original drawing with the new positions
///
/// # Arguments
///
/// * `graph` - The input graph
/// * `drawing` - The drawing to modify (must be a 2D Euclidean drawing)
/// * `d` - The dimension along which to apply the separation constraint (0 for x, 1 for y)
/// * `cluster_id` - A function that returns the cluster ID for a node
/// * `size` - A function that returns the size of a node in a given dimension
///
/// # Type Parameters
///
/// * `G` - The graph type
/// * `N` - The node ID type
/// * `S` - The scalar type for coordinates
/// * `F1` - The type of the cluster ID function
/// * `F2` - The type of the size function
pub fn project_clustered_rectangle_no_overlap_constraints<N, E, Ty, Ix, F1, F2>(
    graph: &Graph<N, E, Ty, Ix>,
    drawing: &mut DrawingEuclidean2d<NodeIndex<Ix>, f32>,
    d: usize,
    mut cluster_id: F1,
    mut size: F2,
) where
    Ty: EdgeType,
    Ix: IndexType,
    F1: FnMut(NodeIndex<Ix>) -> usize,
    F2: FnMut(NodeIndex<Ix>, usize) -> f32,
{
    // Cache cluster_id function results to minimize calls
    let mut node_cluster_map = HashMap::new();
    for node_id in graph.node_identifiers() {
        let cluster = cluster_id(node_id);
        node_cluster_map.insert(node_id, cluster);
    }

    // Create a cluster graph where each node represents a cluster
    let (cluster_graph, _) = coarsen(
        graph,
        &mut |_, node_id| node_cluster_map[&node_id], // Use cached cluster ID
        &mut |_, node_ids| {
            let mut min_x = f32::INFINITY;
            let mut min_y = f32::INFINITY;
            let mut max_x = f32::NEG_INFINITY;
            let mut max_y = f32::NEG_INFINITY;

            for &node_id in node_ids.iter() {
                if let Some(pos) = drawing.position(node_id) {
                    let x = pos.0;
                    let y = pos.1;
                    let half_width = size(node_id, 0) / 2.0;
                    let half_height = size(node_id, 1) / 2.0;

                    min_x = min_x.min(x - half_width);
                    min_y = min_y.min(y - half_height);
                    max_x = max_x.max(x + half_width);
                    max_y = max_y.max(y + half_height);
                }
            }
            (
                node_cluster_map[&node_ids[0]],
                node_ids.clone(),
                min_x,
                min_y,
                max_x,
                max_y,
            )
        },
        &mut |_, _| (), // Edge data is not needed for this algorithm
    );

    // Create a drawing for the cluster graph
    let mut cluster_drawing = DrawingEuclidean2d::initial_placement(&cluster_graph);

    // Set the initial position of each cluster to the center of its bounding box
    for cluster_id in cluster_graph.node_identifiers() {
        if let Some((_, _, min_x, min_y, max_x, max_y)) = cluster_graph.node_weight(cluster_id) {
            let center_x = (min_x + max_x) / 2.0;
            let center_y = (min_y + max_y) / 2.0;

            cluster_drawing.set_x(cluster_id, center_x);
            cluster_drawing.set_y(cluster_id, center_y);
        }
    }

    // Generate and apply constraints
    let constraints = generate_rectangle_no_overlap_constraints_triangulated(
        &cluster_drawing,
        &mut |cluster_id, dim| {
            let (_, _, min_x, min_y, max_x, max_y) = cluster_graph.node_weight(cluster_id).unwrap();
            if dim == 0 {
                max_x - min_x
            } else {
                max_y - min_y
            }
        },
        d,
    );
    project_1d(&mut cluster_drawing, d, &constraints);

    // Calculate the displacement for each cluster
    for cluster_id in cluster_graph.node_identifiers() {
        let (_, node_ids, min_x, min_y, max_x, max_y) =
            cluster_graph.node_weight(cluster_id).unwrap();
        let old_center_x = (min_x + max_x) / 2.0;
        let old_center_y = (min_y + max_y) / 2.0;

        let new_center_x = cluster_drawing.x(cluster_id).unwrap();
        let new_center_y = cluster_drawing.y(cluster_id).unwrap();

        let dx = new_center_x - old_center_x;
        let dy = new_center_y - old_center_y;

        for &node_id in node_ids.iter() {
            if let Some(pos) = drawing.position_mut(node_id) {
                pos.0 += dx;
                pos.1 += dy;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::{Graph, NodeIndex};

    #[test]
    fn test_project_clustered_rectangle_no_overlap_constraints() {
        // Create a graph with 4 nodes in 2 clusters
        let mut graph = Graph::<(), (), petgraph::Undirected>::new_undirected();
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());
        let n4 = graph.add_node(());

        // Create a drawing with the nodes positioned
        let mut drawing = DrawingEuclidean2d::new(&graph);

        // Cluster 1: nodes 0 and 1
        drawing.set_x(n1, 0.0);
        drawing.set_y(n1, 0.0);
        drawing.set_x(n2, 1.0);
        drawing.set_y(n2, 0.0);

        // Cluster 2: nodes 2 and 3
        drawing.set_x(n3, 2.0);
        drawing.set_y(n3, 1.0);
        drawing.set_x(n4, 3.0);
        drawing.set_y(n4, 1.0);

        // Define cluster ID function
        let cluster_id = |node: NodeIndex| {
            if node.index() < 2 {
                0 // Nodes 0 and 1 are in cluster 0
            } else {
                1 // Nodes 2 and 3 are in cluster 1
            }
        };

        // Define node size function
        let node_size = |_: NodeIndex, _: usize| 0.5;

        // Initial positions should have clusters close to each other
        let initial_distance_x = drawing.x(n3).unwrap() - drawing.x(n2).unwrap();
        assert!(
            initial_distance_x < 2.0,
            "Initial x-distance between clusters should be small"
        );

        // Apply the algorithm for x-dimension (d=0)
        project_clustered_rectangle_no_overlap_constraints(
            &graph,
            &mut drawing,
            0,
            cluster_id,
            node_size,
        );

        // After applying, clusters should be separated
        let final_distance_x = drawing.x(n3).unwrap() - drawing.x(n2).unwrap();
        assert!(
            final_distance_x >= 1.0,
            "Final x-distance between clusters should be at least the sum of half-widths"
        );

        // Nodes within the same cluster should maintain their relative positions
        let initial_distance_within_cluster1 = 1.0; // n2.x - n1.x
        let final_distance_within_cluster1 = drawing.x(n2).unwrap() - drawing.x(n1).unwrap();
        assert!(
            (final_distance_within_cluster1 - initial_distance_within_cluster1).abs() < 1e-6,
            "Relative positions within cluster should be preserved"
        );
    }
}
