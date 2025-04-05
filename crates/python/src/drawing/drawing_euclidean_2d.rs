use crate::{
    drawing::PyDrawing,
    graph::{GraphType, NodeId, PyGraphAdapter},
};

/// A 2D point represented as (x, y) coordinates of floats
type Point2D = (f32, f32);
/// A line segment represented as two points (start, end)
type Segment2D = (Point2D, Point2D);
use petgraph::graph::node_index;
use petgraph_drawing::{Drawing, DrawingEuclidean2d};
use pyo3::prelude::*;

/// Python class for 2D Euclidean drawings
///
/// This class represents a drawing in 2D Euclidean space, where each node
/// is assigned (x, y) coordinates.
#[pyclass(extends=PyDrawing)]
#[pyo3(name = "DrawingEuclidean2d")]
pub struct PyDrawingEuclidean2d {
    drawing: DrawingEuclidean2d<NodeId, f32>,
}

impl PyDrawingEuclidean2d {
    /// Creates a new 2D Euclidean drawing
    ///
    /// # Parameters
    /// * `drawing` - The native Rust drawing object
    pub fn new(drawing: DrawingEuclidean2d<NodeId, f32>) -> Self {
        Self { drawing }
    }

    /// Returns a reference to the underlying drawing
    pub fn drawing(&self) -> &DrawingEuclidean2d<NodeId, f32> {
        &self.drawing
    }

    /// Returns a mutable reference to the underlying drawing
    pub fn drawing_mut(&mut self) -> &mut DrawingEuclidean2d<NodeId, f32> {
        &mut self.drawing
    }
}

#[pymethods]
impl PyDrawingEuclidean2d {
    /// Gets the x-coordinate of a node
    ///
    /// # Parameters
    /// * `u` - The node index
    ///
    /// # Returns
    /// The x-coordinate if the node exists, None otherwise
    pub fn x(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.drawing.x(u)
    }

    /// Gets the y-coordinate of a node
    ///
    /// # Parameters
    /// * `u` - The node index
    ///
    /// # Returns
    /// The y-coordinate if the node exists, None otherwise
    pub fn y(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.drawing.y(u)
    }

    /// Sets the x-coordinate of a node
    ///
    /// # Parameters
    /// * `u` - The node index
    /// * `x` - The new x-coordinate
    pub fn set_x(&mut self, u: usize, x: f32) {
        let u = node_index(u);
        self.drawing.set_x(u, x);
    }

    /// Sets the y-coordinate of a node
    ///
    /// # Parameters
    /// * `u` - The node index
    /// * `y` - The new y-coordinate
    pub fn set_y(&mut self, u: usize, y: f32) {
        let u = node_index(u);
        self.drawing.set_y(u, y);
    }

    /// Returns the number of nodes in the drawing
    ///
    /// # Returns
    /// The number of nodes
    pub fn len(&self) -> usize {
        self.drawing.len()
    }

    /// Centralizes the drawing by moving it to the center of the coordinate system
    ///
    /// This method calculates the center of all node positions and then
    /// translates all nodes so that this center is at the origin.
    pub fn centralize(&mut self) {
        self.drawing.centralize();
    }

    /// Clamps node positions to fit within a specified rectangular region
    ///
    /// # Parameters
    /// * `x0` - The minimum x-coordinate of the region
    /// * `y0` - The minimum y-coordinate of the region
    /// * `x1` - The maximum x-coordinate of the region
    /// * `y1` - The maximum y-coordinate of the region
    pub fn clamp_region(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) {
        self.drawing.clamp_region(x0, y0, x1, y1);
    }

    /// Gets the line segments representing an edge between two nodes
    ///
    /// # Parameters
    /// * `u` - The source node index
    /// * `v` - The target node index
    ///
    /// # Returns
    /// A vector of line segments if the edge exists, None otherwise
    pub fn edge_segments(&self, u: usize, v: usize) -> Option<Vec<Segment2D>> {
        self.drawing
            .edge_segments(node_index(u), node_index(v))
            .map(|segments| {
                segments
                    .iter()
                    .map(|&(p, q)| ((p.0, p.1), (q.0, q.1)))
                    .collect::<Vec<_>>()
            })
    }

    /// Creates a new drawing with an initial random placement of nodes
    ///
    /// # Parameters
    /// * `graph` - The graph to create a drawing for
    ///
    /// # Returns
    /// A new drawing with random node positions
    #[staticmethod]
    pub fn initial_placement(graph: &PyGraphAdapter) -> PyObject {
        PyDrawing::new_drawing_euclidean_2d(match graph.graph() {
            GraphType::Graph(native_graph) => DrawingEuclidean2d::initial_placement(native_graph),
            GraphType::DiGraph(native_graph) => DrawingEuclidean2d::initial_placement(native_graph),
        })
    }
}
