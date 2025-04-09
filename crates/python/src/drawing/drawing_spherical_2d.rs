/// 2D Spherical drawing implementation for Python
///
/// This module provides a Python binding for 2D Spherical space drawings.
/// Spherical space represents coordinates on the surface of a sphere, using
/// longitude and latitude coordinates similar to those used in geographic
/// coordinate systems.
use crate::{
    drawing::PyDrawing,
    graph::{GraphType, NodeId, PyGraphAdapter},
};
use petgraph::graph::node_index;
use petgraph_drawing::{Drawing, DrawingSpherical2d};
use pyo3::prelude::*;

/// Python class for 2D Spherical drawings
///
/// This class represents a drawing on the surface of a sphere, where each node
/// is assigned longitude and latitude coordinates. Spherical space is useful for
/// representing global networks, wrapping layouts around a sphere, or visualizing
/// data with a naturally spherical distribution.
///
/// The coordinates are specified in radians, with longitude ranging from -π to π
/// and latitude ranging from -π/2 to π/2.
#[pyclass(extends=PyDrawing)]
#[pyo3(name = "DrawingSpherical2d")]
pub struct PyDrawingSpherical2d {
    drawing: DrawingSpherical2d<NodeId, f32>,
}

impl PyDrawingSpherical2d {
    /// Creates a new 2D Spherical drawing
    ///
    /// :param drawing: The native Rust drawing object
    /// :type drawing: DrawingSpherical2d<NodeId, f32>
    /// :return: A new PyDrawingSpherical2d instance
    /// :rtype: PyDrawingSpherical2d
    pub fn new(drawing: DrawingSpherical2d<NodeId, f32>) -> Self {
        Self { drawing }
    }

    /// Returns a reference to the underlying drawing
    ///
    /// :return: A reference to the underlying drawing
    /// :rtype: &DrawingSpherical2d<NodeId, f32>
    pub fn drawing(&self) -> &DrawingSpherical2d<NodeId, f32> {
        &self.drawing
    }

    /// Returns a mutable reference to the underlying drawing
    ///
    /// :return: A mutable reference to the underlying drawing
    /// :rtype: &mut DrawingSpherical2d<NodeId, f32>
    pub fn drawing_mut(&mut self) -> &mut DrawingSpherical2d<NodeId, f32> {
        &mut self.drawing
    }
}

#[pymethods]
impl PyDrawingSpherical2d {
    /// Gets the longitude coordinate of a node on the sphere
    ///
    /// Longitude represents the angular distance east or west on the sphere,
    /// analogous to longitude on Earth.
    ///
    /// :param u: The node index
    /// :type u: int
    /// :return: The longitude (in radians, from -π to π) if the node exists, None otherwise
    /// :rtype: float or None
    pub fn lon(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.drawing.lon(u)
    }

    /// Gets the latitude coordinate of a node on the sphere
    ///
    /// Latitude represents the angular distance north or south on the sphere,
    /// analogous to latitude on Earth.
    ///
    /// :param u: The node index
    /// :type u: int
    /// :return: The latitude (in radians, from -π/2 to π/2) if the node exists, None otherwise
    /// :rtype: float or None
    pub fn lat(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.drawing.lat(u)
    }

    /// Sets the longitude coordinate of a node on the sphere
    ///
    /// :param u: The node index
    /// :type u: int
    /// :param value: The new longitude value (in radians)
    /// :type value: float
    /// :return: None
    /// :rtype: None
    pub fn set_lon(&mut self, u: usize, value: f32) {
        let u = node_index(u);
        self.drawing.set_lon(u, value);
    }

    /// Sets the latitude coordinate of a node on the sphere
    ///
    /// :param u: The node index
    /// :type u: int
    /// :param value: The new latitude value (in radians)
    /// :type value: float
    /// :return: None
    /// :rtype: None
    pub fn set_lat(&mut self, u: usize, value: f32) {
        let u = node_index(u);
        self.drawing.set_lat(u, value);
    }

    /// Returns the number of nodes in the drawing
    ///
    /// :return: The number of nodes
    /// :rtype: int
    pub fn len(&self) -> usize {
        self.drawing.len()
    }

    /// Creates a new drawing with an initial random placement of nodes
    ///
    /// This method initializes a spherical drawing with nodes placed
    /// randomly on the surface of the sphere.
    ///
    /// :param graph: The graph to create a drawing for
    /// :type graph: Graph or DiGraph
    /// :return: A new spherical drawing with initial node positions
    /// :rtype: DrawingSpherical2d
    #[staticmethod]
    pub fn initial_placement(graph: &PyGraphAdapter) -> PyObject {
        PyDrawing::new_drawing_spherical_2d(match graph.graph() {
            GraphType::Graph(native_graph) => DrawingSpherical2d::initial_placement(native_graph),
            GraphType::DiGraph(native_graph) => DrawingSpherical2d::initial_placement(native_graph),
        })
    }
}
