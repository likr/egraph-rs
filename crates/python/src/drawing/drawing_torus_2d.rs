/// 2D Torus drawing implementation for Python
///
/// This module provides a Python binding for 2D Toroidal space drawings.
/// Toroidal space is a "donut-shaped" space that wraps around in both
/// the x and y dimensions, allowing for continuous layouts without boundaries.
/// It's particularly useful for visualizing networks with periodic structures
/// or to eliminate boundary effects in layouts.
use crate::{
    drawing::PyDrawing,
    graph::{GraphType, NodeId, PyGraphAdapter},
};

type Point2D = (f32, f32);
type Segment2D = (Point2D, Point2D);
use petgraph::graph::node_index;
use petgraph_drawing::{Drawing, DrawingTorus2d};
use pyo3::prelude::*;

/// Python class for 2D Torus drawings
///
/// This class represents a drawing on a torus (donut shape), where each node
/// is assigned (x, y) coordinates in the unit square [0, 1] × [0, 1], with
/// the understanding that opposite edges of this square are "glued" together.
///
/// The toroidal topology allows for continuous layouts without boundaries,
/// which can be useful for visualizing periodic structures or eliminating
/// boundary effects in force-directed layouts.
#[pyclass(extends=PyDrawing)]
#[pyo3(name = "DrawingTorus2d")]
pub struct PyDrawingTorus2d {
    drawing: DrawingTorus2d<NodeId, f32>,
}

impl PyDrawingTorus2d {
    /// Creates a new 2D Torus drawing
    ///
    /// :param drawing: The native Rust drawing object
    /// :type drawing: DrawingTorus2d<NodeId, f32>
    /// :return: A new PyDrawingTorus2d instance
    /// :rtype: PyDrawingTorus2d
    pub fn new(drawing: DrawingTorus2d<NodeId, f32>) -> Self {
        Self { drawing }
    }

    /// Returns a reference to the underlying drawing
    ///
    /// :return: A reference to the underlying drawing
    /// :rtype: &DrawingTorus2d<NodeId, f32>
    pub fn drawing(&self) -> &DrawingTorus2d<NodeId, f32> {
        &self.drawing
    }

    /// Returns a mutable reference to the underlying drawing
    ///
    /// :return: A mutable reference to the underlying drawing
    /// :rtype: &mut DrawingTorus2d<NodeId, f32>
    pub fn drawing_mut(&mut self) -> &mut DrawingTorus2d<NodeId, f32> {
        &mut self.drawing
    }
}

#[pymethods]
impl PyDrawingTorus2d {
    /// Gets the x-coordinate of a node on the torus
    ///
    /// The x-coordinate is in the range [0, 1], representing the position
    /// along the horizontal dimension of the torus.
    ///
    /// :param u: The node index
    /// :type u: int
    /// :return: The x-coordinate if the node exists, None otherwise
    /// :rtype: float or None
    pub fn x(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.drawing.x(u)
    }

    /// Gets the y-coordinate of a node on the torus
    ///
    /// The y-coordinate is in the range [0, 1], representing the position
    /// along the vertical dimension of the torus.
    ///
    /// :param u: The node index
    /// :type u: int
    /// :return: The y-coordinate if the node exists, None otherwise
    /// :rtype: float or None
    pub fn y(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.drawing.y(u)
    }

    /// Sets the x-coordinate of a node on the torus
    ///
    /// Note that values outside the range [0, 1] will be wrapped around
    /// to maintain the toroidal topology.
    ///
    /// :param u: The node index
    /// :type u: int
    /// :param x: The new x-coordinate
    /// :type x: float
    /// :return: None
    /// :rtype: None
    pub fn set_x(&mut self, u: usize, x: f32) {
        let u = node_index(u);
        self.drawing.set_x(u, x);
    }

    /// Sets the y-coordinate of a node on the torus
    ///
    /// Note that values outside the range [0, 1] will be wrapped around
    /// to maintain the toroidal topology.
    ///
    /// :param u: The node index
    /// :type u: int
    /// :param y: The new y-coordinate
    /// :type y: float
    /// :return: None
    /// :rtype: None
    pub fn set_y(&mut self, u: usize, y: f32) {
        let u = node_index(u);
        self.drawing.set_y(u, y);
    }

    /// Returns the number of nodes in the drawing
    ///
    /// :return: The number of nodes
    /// :rtype: int
    pub fn len(&self) -> usize {
        self.drawing.len()
    }

    /// Checks if the drawing is empty
    ///
    /// :return: True if the drawing contains no nodes, false otherwise
    /// :rtype: bool
    pub fn is_empty(&self) -> bool {
        self.drawing.is_empty()
    }

    /// Computes the line segments needed to draw an edge on the torus
    ///
    /// In toroidal space, an edge may need to be drawn as multiple line segments
    /// when it crosses the boundary of the unit square. This method computes all
    /// segments needed to properly represent the edge.
    ///
    /// :param u: The source node index
    /// :type u: int
    /// :param v: The target node index
    /// :type v: int
    /// :return: A vector of line segments (pairs of points) if both nodes exist, None otherwise
    /// :rtype: list of tuple or None
    pub fn edge_segments(&self, u: usize, v: usize) -> Option<Vec<Segment2D>> {
        self.drawing
            .edge_segments(node_index(u), node_index(v))
            .map(|segments| {
                segments
                    .iter()
                    .map(|&(p, q)| ((p.0 .0, p.1 .0), (q.0 .0, q.1 .0)))
                    .collect::<Vec<_>>()
            })
    }

    /// Creates a new drawing with an initial random placement of nodes
    ///
    /// This method initializes a toroidal drawing with nodes placed randomly
    /// within the unit square [0, 1] × [0, 1].
    ///
    /// :param graph: The graph to create a drawing for
    /// :type graph: Graph or DiGraph
    /// :return: A new toroidal drawing with initial node positions
    /// :rtype: DrawingTorus2d
    #[staticmethod]
    pub fn initial_placement(graph: &PyGraphAdapter) -> PyObject {
        PyDrawing::new_drawing_torus_2d(match graph.graph() {
            GraphType::Graph(native_graph) => DrawingTorus2d::initial_placement(native_graph),
            GraphType::DiGraph(native_graph) => DrawingTorus2d::initial_placement(native_graph),
        })
    }
}
