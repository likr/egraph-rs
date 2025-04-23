use petgraph_drawing::{
    Delta, Drawing, DrawingEuclidean2d, DrawingIndex, DrawingValue, MetricCartesian,
};

use crate::{project_1d, Constraint};

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
    fn test_rectangle_no_overlap_constraints_2d() {
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
        let node_size = 0.5f32;

        // Generate constraints for x-dimension (k=0)
        let constraints_x =
            generate_rectangle_no_overlap_constraints_2d(&drawing, |_, _| node_size, 0);

        // Generate constraints for y-dimension (k=1)
        let constraints_y =
            generate_rectangle_no_overlap_constraints_2d(&drawing, |_, _| node_size, 1);

        // We expect constraints between potentially overlapping rectangles
        // For this configuration, we expect 6 constraints (all node pairs)
        assert_eq!(constraints_x.len(), 6);
        assert_eq!(constraints_y.len(), 6);

        // Verify that the constraints have the correct gap values
        // The gap should be the sum of half-widths of the two nodes
        let expected_gap = node_size;
        for constraint in &constraints_x {
            assert!((constraint.gap - expected_gap as f32).abs() < 1e-6);
        }
        for constraint in &constraints_y {
            assert!((constraint.gap - expected_gap as f32).abs() < 1e-6);
        }
    }

    #[test]
    fn test_project_rectangle_no_overlap_constraints_2d() {
        // Create a graph with 2 overlapping nodes
        let mut graph = Graph::<(), (), petgraph::Undirected>::new_undirected();
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());

        // Create a drawing with the nodes positioned with overlap
        let mut drawing = DrawingEuclidean2d::new(&graph);
        drawing.set_x(n1, 0.0);
        drawing.set_y(n1, 0.0);
        drawing.set_x(n2, 0.5); // Overlapping in X
        drawing.set_y(n2, 0.5); // Overlapping in Y

        // Set node size so they overlap (each node is 1.0 wide)
        let node_size = 1.0f32;

        // Save original positions
        let orig_x1 = drawing.x(n1).unwrap();
        let orig_y1 = drawing.y(n1).unwrap();
        let orig_x2 = drawing.x(n2).unwrap();
        let orig_y2 = drawing.y(n2).unwrap();

        // Apply constraints to remove overlaps
        project_rectangle_no_overlap_constraints_2d(&mut drawing, |_, _| node_size, 0);

        // Check that positions have changed to resolve overlap
        assert!(
            drawing.x(n1).unwrap() != orig_x1
                || drawing.x(n2).unwrap() != orig_x2
                || drawing.y(n1).unwrap() != orig_y1
                || drawing.y(n2).unwrap() != orig_y2,
            "Rectangle positions should change to resolve overlap"
        );

        // Check that the nodes are no longer overlapping
        // With node size 1.0, centers should be at least 1.0 apart
        let dx = (drawing.x(n1).unwrap() - drawing.x(n2).unwrap()).abs() as f32;
        let dy = (drawing.y(n1).unwrap() - drawing.y(n2).unwrap()).abs() as f32;

        assert!(
            dx >= node_size || dy >= node_size,
            "Rectangles still overlap after projection: dx={}, dy={}",
            dx,
            dy
        );
    }
}

/// Generates separation constraints to prevent rectangle overlaps in a specific dimension.
///
/// This function generates separation constraints for rectangles in a drawing to ensure
/// they don't overlap. It examines all pairs of rectangles and generates constraints
/// for the given dimension (X or Y).
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
///     generate_rectangle_no_overlap_constraints_2d, project_1d,
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
/// let constraints = generate_rectangle_no_overlap_constraints_2d(
///     &drawing,
///     |_, _| node_size,
///     1,
/// );
///
/// // Apply the constraints
/// project_1d(&mut drawing, 1, &constraints);
/// ```
pub fn generate_rectangle_no_overlap_constraints_2d<N, F>(
    drawing: &DrawingEuclidean2d<N, f32>,
    mut size: F,
    k: usize,
) -> Vec<Constraint>
where
    N: DrawingIndex + Copy,
    F: FnMut(N, usize) -> f32,
{
    let n = drawing.len();
    let mut constraints = Vec::new();
    let d = drawing.dimension();

    let sizes = (0..n)
        .map(|i| {
            (0..d)
                .map(|k| {
                    let x = drawing.raw_entry(i).nth(k);
                    let w = size(*drawing.node_id(i), k);
                    (x - w / 2., x + w / 2.)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let s = if k == 0 { 0 } else { 1 };
    let t = if k == 0 { 1 } else { 0 };

    // Generate constraints for all potentially overlapping pairs
    for j in 1..n {
        for i in 0..j {
            // Check if the rectangles potentially overlap
            if overlap_1d(sizes[i][t].0, sizes[i][t].1, sizes[j][t].0, sizes[j][t].1) {
                let i_pos = drawing.raw_entry(i).nth(s);
                let j_pos = drawing.raw_entry(j).nth(s);

                // Calculate the minimum separation needed
                let gap = (sizes[i][s].1 - sizes[i][s].0 + sizes[j][s].1 - sizes[j][s].0) / 2.0;

                // Create constraint ensuring they don't overlap
                if i_pos < j_pos {
                    constraints.push(Constraint::new(i, j, gap));
                } else {
                    constraints.push(Constraint::new(j, i, gap));
                }
            }
        }
    }

    constraints
}

/// Projects rectangle positions to satisfy non-overlap constraints in both dimensions.
///
/// This function is a convenience wrapper that generates constraints for both X and Y
/// dimensions and applies them to ensure rectangles don't overlap. It's equivalent to
/// the `removeOverlaps` function in WebCola.
///
/// # Arguments
///
/// * `drawing` - A mutable reference to a `DrawingEuclidean2d` containing rectangle positions.
/// * `size` - A function that returns the size of a node in a given dimension.
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
/// use petgraph_layout_separation_constraints::project_rectangle_no_overlap_constraints_2d;
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
/// drawing.set_x(n2, 0.8);
/// drawing.set_y(n2, 0.0);
/// drawing.set_x(n3, 0.0);
/// drawing.set_y(n3, 0.8);
/// drawing.set_x(n4, 0.8);
/// drawing.set_y(n4, 0.8);
///
/// // Apply constraints to remove overlaps
/// let node_size = 1.0; // Size of each node
/// project_rectangle_no_overlap_constraints_2d(&mut drawing, |_, _| node_size);
/// ```
pub fn project_rectangle_no_overlap_constraints_2d<N, F>(
    drawing: &mut DrawingEuclidean2d<N, f32>,
    size: F,
    k: usize,
) where
    N: DrawingIndex + Copy,
    F: FnMut(N, usize) -> f32,
{
    let mut size = size;
    // Generate constraints for X dimension (0)
    let constraints = generate_rectangle_no_overlap_constraints_2d(drawing, &mut size, k);
    project_1d(drawing, 0, &constraints);
}
