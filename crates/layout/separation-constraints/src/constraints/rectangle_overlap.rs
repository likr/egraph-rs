use petgraph_algorithm_triangulation::triangulation;
use petgraph_drawing::{
    Delta, Drawing, DrawingEuclidean2d, DrawingIndex, DrawingValue, MetricCartesian,
};

use crate::Constraint;

fn overlap_1d<S>(x00: S, x01: S, x10: S, x11: S) -> bool
where
    S: DrawingValue,
{
    (x00 < x11 && x10 < x01) || (x10 < x01 && x00 < x11)
}

fn overlap<S>(a: &Vec<(S, S)>, b: &Vec<(S, S)>) -> bool
where
    S: DrawingValue,
{
    a.iter()
        .zip(b.iter())
        .all(|(&(x00, x01), &(x10, x11))| overlap_1d(x00, x01, x10, x11))
}

pub fn generate_rectangle_no_overlap_constraints<D, Diff, M, F>(
    drawing: &D,
    size: F,
    k: usize,
) -> Vec<Constraint>
where
    D: Drawing<Item = M>,
    D::Index: Clone,
    Diff: Delta<S = f32>,
    M: MetricCartesian<D = Diff>,
    F: FnMut(D::Index, usize) -> f32,
{
    let mut size = size;
    let n = drawing.len();
    let d = drawing.dimension();
    let mut constraints = vec![];
    let sizes = (0..drawing.len())
        .map(|i| {
            let u = drawing.node_id(i);
            let x = drawing.raw_entry(i);
            (0..d)
                .map(|j| {
                    let xj = x.nth(j);
                    let w = size(u.clone(), j) / 2.;
                    (xj - w, xj + w)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for j in 1..n {
        for i in 0..j {
            if overlap(&sizes[i], &sizes[j]) {
                constraints.push(
                    if drawing.raw_entry(i).nth(k) < drawing.raw_entry(j).nth(k) {
                        Constraint::new(
                            i,
                            j,
                            (sizes[i][k].1 - sizes[i][k].0 + sizes[j][k].1 - sizes[j][k].0) / 2.,
                        )
                    } else {
                        Constraint::new(
                            j,
                            i,
                            (sizes[i][k].1 - sizes[i][k].0 + sizes[j][k].1 - sizes[j][k].0) / 2.,
                        )
                    },
                )
            }
        }
    }
    constraints
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::Graph;

    #[test]
    fn test_rectangle_no_overlap_constraints_triangulated() {
        // Create a graph with 4 nodes in a square formation
        let mut graph = Graph::<(), (), petgraph::Undirected>::new_undirected();
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());
        let n4 = graph.add_node(());

        // Create a drawing with the nodes positioned in a square
        let mut drawing = DrawingEuclidean2d::new(&graph);
        drawing.set_x(n1, 0.0);
        drawing.set_y(n1, 0.0);
        drawing.set_x(n2, 1.0);
        drawing.set_y(n2, 0.0);
        drawing.set_x(n3, 0.0);
        drawing.set_y(n3, 1.0);
        drawing.set_x(n4, 1.0);
        drawing.set_y(n4, 1.0);

        // Set node size
        let node_size = 0.5;

        // Generate constraints for x-dimension (k=0)
        let constraints_x =
            generate_rectangle_no_overlap_constraints_triangulated(&drawing, |_, _| node_size, 0);

        // Generate constraints for y-dimension (k=1)
        let constraints_y =
            generate_rectangle_no_overlap_constraints_triangulated(&drawing, |_, _| node_size, 1);

        // The triangulated graph should have 5 edges (4 around the square and 1 diagonal)
        // So we should have 5 constraints for each dimension
        assert_eq!(constraints_x.len(), 5);
        assert_eq!(constraints_y.len(), 5);

        // Verify that the constraints have the correct gap values
        // The gap should be the sum of half-widths of the two nodes
        let expected_gap = node_size;
        for constraint in &constraints_x {
            assert!((constraint.gap - expected_gap).abs() < 1e-6);
        }
        for constraint in &constraints_y {
            assert!((constraint.gap - expected_gap).abs() < 1e-6);
        }
    }

    #[test]
    fn test_rectangle_no_overlap_constraints_triangulated_triangle() {
        // Create a graph with 3 nodes in a triangle formation
        let mut graph = Graph::<(), (), petgraph::Undirected>::new_undirected();
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());

        // Create a drawing with the nodes positioned in a triangle
        let mut drawing = DrawingEuclidean2d::new(&graph);
        drawing.set_x(n1, 0.0);
        drawing.set_y(n1, 0.0);
        drawing.set_x(n2, 1.0);
        drawing.set_y(n2, 0.0);
        drawing.set_x(n3, 0.5);
        drawing.set_y(n3, 0.866); // Approximately sqrt(3)/2

        // Set node size
        let node_size = 0.3;

        // Generate constraints for both dimensions
        let constraints_x =
            generate_rectangle_no_overlap_constraints_triangulated(&drawing, |_, _| node_size, 0);
        let constraints_y =
            generate_rectangle_no_overlap_constraints_triangulated(&drawing, |_, _| node_size, 1);

        // The triangulated graph should have 3 edges for a triangle
        // So we should have 3 constraints for each dimension
        assert_eq!(constraints_x.len(), 3);
        assert_eq!(constraints_y.len(), 3);

        // Verify that the constraints have the correct gap values
        let expected_gap = node_size;
        for constraint in &constraints_x {
            assert!((constraint.gap - expected_gap).abs() < 1e-6);
        }
        for constraint in &constraints_y {
            assert!((constraint.gap - expected_gap).abs() < 1e-6);
        }
    }

    #[test]
    fn test_rectangle_no_overlap_constraints_triangulated_collinear() {
        // Create a graph with 3 collinear nodes
        let mut graph = Graph::<(), (), petgraph::Undirected>::new_undirected();
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());

        // Create a drawing with the nodes positioned in a line
        let mut drawing = DrawingEuclidean2d::new(&graph);
        drawing.set_x(n1, 0.0);
        drawing.set_y(n1, 0.0);
        drawing.set_x(n2, 1.0);
        drawing.set_y(n2, 0.0);
        drawing.set_x(n3, 2.0);
        drawing.set_y(n3, 0.0);

        // Set node size
        let node_size = 0.4;

        // Generate constraints for both dimensions
        let constraints_x =
            generate_rectangle_no_overlap_constraints_triangulated(&drawing, |_, _| node_size, 0);
        let constraints_y =
            generate_rectangle_no_overlap_constraints_triangulated(&drawing, |_, _| node_size, 1);

        // The triangulated graph should have 2 edges for collinear points
        // So we should have 2 constraints for each dimension
        assert_eq!(constraints_x.len(), 2);
        assert_eq!(constraints_y.len(), 2);

        // Verify that the constraints have the correct gap values
        let expected_gap = node_size;
        for constraint in &constraints_x {
            assert!((constraint.gap - expected_gap).abs() < 1e-6);
        }
        for constraint in &constraints_y {
            assert!((constraint.gap - expected_gap).abs() < 1e-6);
        }
    }

    #[test]
    fn test_rectangle_no_overlap_constraints_triangulated_different_sizes() {
        // Create a graph with 4 nodes
        let mut graph = Graph::<u32, (), petgraph::Undirected>::new_undirected();
        let n1 = graph.add_node(1); // Node with ID 1
        let n2 = graph.add_node(2); // Node with ID 2
        let n3 = graph.add_node(3); // Node with ID 3
        let n4 = graph.add_node(4); // Node with ID 4

        // Create a drawing with the nodes positioned in a square
        let mut drawing = DrawingEuclidean2d::new(&graph);
        drawing.set_x(n1, 0.0);
        drawing.set_y(n1, 0.0);
        drawing.set_x(n2, 1.0);
        drawing.set_y(n2, 0.0);
        drawing.set_x(n3, 0.0);
        drawing.set_y(n3, 1.0);
        drawing.set_x(n4, 1.0);
        drawing.set_y(n4, 1.0);

        // Set node sizes based on node ID
        let constraints = generate_rectangle_no_overlap_constraints_triangulated(
            &drawing,
            |_, _| 5., // Size depends on node ID
            0,
        );

        // The triangulated graph should have 5 edges (4 around the square and 1 diagonal)
        assert_eq!(constraints.len(), 5);

        // Verify that the constraints have the correct gap values
        // For each constraint, check that the gap is the sum of half-widths of the two nodes
        for constraint in &constraints {
            let size_i = 5.;
            let size_j = 5.;
            let expected_gap = (size_i + size_j) / 2.0;
            assert!((constraint.gap - expected_gap).abs() < 1e-6);
        }
    }
}

/// Generates rectangle overlap constraints for adjacent vertex pairs in a triangulated graph.
///
/// This function takes a 2D Euclidean drawing as input, performs Delaunay triangulation
/// based on node positions, and then generates separation constraints for all
/// node pairs that are adjacent in the triangulated graph. This approach is more efficient than
/// checking all possible node pairs, especially for large graphs.
///
/// # Arguments
///
/// * `drawing` - A reference to a `DrawingEuclidean2d` that contains the positions of the nodes.
/// * `size` - A function that returns the size of a node in a given dimension.
/// * `k` - The dimension along which to apply the separation constraint (0 for x, 1 for y).
///
/// # Returns
///
/// A vector of `Constraint` objects representing the separation constraints.
///
/// # Type Parameters
///
/// * `N` - The node ID type.
/// * `F` - The type of the size function.
/// * `S` - The scalar type for coordinates (must implement `DrawingValue`).
///
/// # Example
///
/// ```
/// use petgraph::graph::Graph;
/// use petgraph_drawing::DrawingEuclidean2d;
/// use petgraph_layout_separation_constraints::{
///     generate_rectangle_no_overlap_constraints_triangulated, project_1d,
/// };
///
/// // Create a graph
/// let mut graph = Graph::<(), (), petgraph::Undirected>::new_undirected();
/// let n1 = graph.add_node(());
/// let n2 = graph.add_node(());
/// let n3 = graph.add_node(());
/// let n4 = graph.add_node(());
///
/// // Create a drawing
/// let mut drawing = DrawingEuclidean2d::new(&graph);
/// drawing.set_x(n1, 0.0);
/// drawing.set_y(n1, 0.0);
/// drawing.set_x(n2, 1.0);
/// drawing.set_y(n2, 0.0);
/// drawing.set_x(n3, 0.0);
/// drawing.set_y(n3, 1.0);
/// drawing.set_x(n4, 1.0);
/// drawing.set_y(n4, 1.0);
///
/// // Generate constraints for the y-dimension (k=1)
/// let node_size = 0.2; // Size of each node
/// let constraints = generate_rectangle_no_overlap_constraints_triangulated(
///     &drawing,
///     |_, _| node_size,
///     1,
/// );
///
/// // Apply the constraints
/// project_1d(&mut drawing, 1, &constraints);
/// ```
pub fn generate_rectangle_no_overlap_constraints_triangulated<N, F, S>(
    drawing: &DrawingEuclidean2d<N, S>,
    size: F,
    k: usize,
) -> Vec<Constraint>
where
    N: DrawingIndex + Copy,
    S: DrawingValue,
    F: FnMut(N, usize) -> S,
{
    let mut size = size;

    // Create a triangulated graph
    let triangulated_graph = triangulation(drawing);

    // Create a mapping from original node indices to drawing indices
    let mut node_map = std::collections::HashMap::new();
    for i in 0..drawing.len() {
        let node_id = drawing.node_id(i);
        node_map.insert(node_id, i);
    }

    // Calculate the sizes of all rectangles
    let d = drawing.dimension();
    let sizes = (0..drawing.len())
        .map(|i| {
            let u = drawing.node_id(i);
            let x = drawing.raw_entry(i);
            (0..d)
                .map(|j| {
                    let xj = *x.nth(j);
                    let w = size(*u, j) / S::from_f32(2.).unwrap();
                    (xj - w, xj + w)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // Generate constraints for adjacent nodes in the triangulated graph
    let mut constraints = Vec::new();
    let mut processed_pairs = std::collections::HashSet::new();

    // Create a mapping from NodeIndex in triangulated graph to original node IDs
    let mut tri_to_orig = std::collections::HashMap::new();
    for (idx, node) in triangulated_graph.node_indices().enumerate() {
        // The triangulation function preserves the order of nodes
        tri_to_orig.insert(node, drawing.node_id(idx));
    }

    for edge in triangulated_graph.edge_indices() {
        let (source, target) = triangulated_graph.edge_endpoints(edge).unwrap();

        // Get the original node IDs
        let source_id = tri_to_orig[&source];
        let target_id = tri_to_orig[&target];

        // Get the indices in the drawing
        let i = node_map[&source_id];
        let j = node_map[&target_id];

        // Skip if we've already processed this pair
        if processed_pairs.contains(&(i.min(j), i.max(j))) {
            continue;
        }
        processed_pairs.insert((i.min(j), i.max(j)));

        constraints.push(
            if drawing.raw_entry(i).nth(k) < drawing.raw_entry(j).nth(k) {
                Constraint::new(
                    i,
                    j,
                    ((sizes[i][k].1 - sizes[i][k].0 + sizes[j][k].1 - sizes[j][k].0)
                        / S::from_f32(2.).unwrap())
                    .to_f32()
                    .unwrap(),
                )
            } else {
                Constraint::new(
                    j,
                    i,
                    ((sizes[i][k].1 - sizes[i][k].0 + sizes[j][k].1 - sizes[j][k].0)
                        / S::from_f32(2.).unwrap())
                    .to_f32()
                    .unwrap(),
                )
            },
        )
    }

    constraints
}
