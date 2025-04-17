use petgraph::{prelude::*, visit::IntoNodeIdentifiers};
use petgraph_drawing::{Drawing, DrawingEuclidean2d, DrawingIndex, DrawingValue};
use spade::{DelaunayTriangulation, HasPosition, Point2, SpadeNum, Triangulation};
use std::collections::HashMap;

/// A point type that wraps a node ID and its coordinates for use with spade.
#[derive(Clone, Copy, Debug)]
struct NodePoint<N, S> {
    node_id: N,
    x: S,
    y: S,
}

impl<N, S> NodePoint<N, S>
where
    N: Copy,
    S: DrawingValue,
{
    fn new(node_id: N, x: S, y: S) -> Self {
        Self { node_id, x, y }
    }
}

impl<N, S> HasPosition for NodePoint<N, S>
where
    S: DrawingValue + SpadeNum,
{
    type Scalar = S;

    fn position(&self) -> Point2<S> {
        Point2::new(self.x, self.y)
    }
}

impl<N, S> PartialEq for NodePoint<N, S>
where
    N: PartialEq,
    S: DrawingValue,
{
    fn eq(&self, other: &Self) -> bool {
        self.node_id == other.node_id
    }
}

impl<N, S> Eq for NodePoint<N, S>
where
    N: Eq,
    S: DrawingValue,
{
}

/// Performs Delaunay triangulation on a graph based on node positions in a 2D Euclidean drawing.
///
/// This function takes a graph and a drawing as input, extracts the node positions from the drawing,
/// computes the Delaunay triangulation of these points using the spade library, and returns a new
/// graph with the same nodes but with edges representing the triangulation.
///
/// # Arguments
///
/// * `graph` - A reference to any graph type that implements `IntoNodeIdentifiers`.
/// * `drawing` - A reference to a `DrawingEuclidean2d` that contains the positions of the nodes.
///
/// # Returns
///
/// A new undirected graph with the same nodes as the input graph, but with edges representing
/// the Delaunay triangulation.
///
/// # Examples
///
/// ```
/// use petgraph::graph::Graph;
/// use petgraph_drawing::DrawingEuclidean2d;
/// use petgraph_algorithm_triangulation::triangulation;
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
/// // Compute the Delaunay triangulation
/// let triangulated_graph = triangulation(&graph, &drawing);
///
/// // The triangulated graph should have 4 nodes and 5 edges
/// assert_eq!(triangulated_graph.node_count(), 4);
/// assert_eq!(triangulated_graph.edge_count(), 5);
/// ```
pub fn triangulation<G, S>(
    graph: G,
    drawing: &DrawingEuclidean2d<G::NodeId, S>,
) -> Graph<(), (), Undirected>
where
    G: IntoNodeIdentifiers,
    G::NodeId: DrawingIndex + Copy + Into<NodeIndex>,
    S: DrawingValue + SpadeNum,
{
    // Create a mapping from node IDs to their indices in the new graph
    let mut node_map = HashMap::new();
    let mut new_graph = Graph::new_undirected();

    // Add all nodes from the original graph to the new graph
    for node_id in graph.node_identifiers() {
        let new_node = new_graph.add_node(());
        node_map.insert(node_id, new_node);
    }

    // Create a Delaunay triangulation
    let mut triangulation = DelaunayTriangulation::<NodePoint<G::NodeId, S>>::new();

    // Insert points into the triangulation
    for node_id in graph.node_identifiers() {
        if let Some(pos) = drawing.position(node_id) {
            let _ = triangulation.insert(NodePoint::new(node_id, pos.0, pos.1));
        }
    }

    // Add edges from the triangulation to the new graph
    let mut added_edges = HashMap::new();
    for edge in triangulation.directed_edges() {
        let from = edge.from().data().node_id;
        let to = edge.to().data().node_id;

        // Skip if we've already added this edge
        if added_edges.contains_key(&(from, to)) || added_edges.contains_key(&(to, from)) {
            continue;
        }

        // Add the edge to the new graph
        new_graph.add_edge(node_map[&from], node_map[&to], ());
        added_edges.insert((from, to), true);
    }

    new_graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::Graph;

    #[test]
    fn test_triangulation_square() {
        // Create a graph with 4 nodes in a square formation
        let mut graph = Graph::new_undirected();
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());
        let n4 = graph.add_node(());
        graph.add_edge(n1, n2, ());
        graph.add_edge(n1, n3, ());
        graph.add_edge(n1, n4, ());

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

        // Compute the Delaunay triangulation
        let triangulated_graph = triangulation(&graph, &drawing);

        // The triangulated graph should have 4 nodes and 5 edges
        // (4 edges around the square and 1 diagonal)
        assert_eq!(triangulated_graph.node_count(), 4);
        assert_eq!(triangulated_graph.edge_count(), 5);
    }

    #[test]
    fn test_triangulation_triangle() {
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

        // Compute the Delaunay triangulation
        let triangulated_graph = triangulation(&graph, &drawing);

        // The triangulated graph should have 3 nodes and 3 edges
        assert_eq!(triangulated_graph.node_count(), 3);
        assert_eq!(triangulated_graph.edge_count(), 3);
    }

    #[test]
    fn test_triangulation_collinear_points() {
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

        // Compute the Delaunay triangulation
        let triangulated_graph = triangulation(&graph, &drawing);

        // The triangulated graph should have 3 nodes and 2 edges
        assert_eq!(triangulated_graph.node_count(), 3);
        assert_eq!(triangulated_graph.edge_count(), 2);
    }
}
