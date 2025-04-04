/// 2D Hyperbolic drawing implementation for Python
///
/// This module provides a Python binding for 2D Hyperbolic space drawings.
/// Hyperbolic space is a non-Euclidean geometry with negative curvature,
/// which can be useful for visualizing hierarchical data or graphs with
/// exponential growth patterns.
use crate::{
    drawing::PyDrawing,
    graph::{GraphType, NodeId, PyGraphAdapter},
};
use petgraph::graph::node_index;
use petgraph_drawing::{Drawing, DrawingHyperbolic2d};
use pyo3::prelude::*;

/// Python class for 2D Hyperbolic drawings
///
/// This class represents a drawing in 2D Hyperbolic space, where each node
/// is assigned (x, y) coordinates. Hyperbolic space has the property that
/// distances grow exponentially as you move away from the origin, allowing
/// more space to represent large, complex graphs with hierarchical structure.
///
/// In the Poincaré disk model used here, the entire hyperbolic plane is
/// represented within a unit disk.
#[pyclass(extends=PyDrawing)]
#[pyo3(name = "DrawingHyperbolic2d")]
pub struct PyDrawingHyperbolic2d {
    drawing: DrawingHyperbolic2d<NodeId, f32>,
}

impl PyDrawingHyperbolic2d {
    pub fn new(drawing: DrawingHyperbolic2d<NodeId, f32>) -> Self {
        Self { drawing }
    }

    pub fn drawing(&self) -> &DrawingHyperbolic2d<NodeId, f32> {
        &self.drawing
    }

    pub fn drawing_mut(&mut self) -> &mut DrawingHyperbolic2d<NodeId, f32> {
        &mut self.drawing
    }
}

#[pymethods]
impl PyDrawingHyperbolic2d {
    /// Gets the x-coordinate of a node in the hyperbolic plane
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

    /// Gets the y-coordinate of a node in the hyperbolic plane
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

    /// Sets the x-coordinate of a node in the hyperbolic plane
    ///
    /// When setting coordinates in hyperbolic space, you should ensure that
    /// points remain within the unit disk (x² + y² < 1) for the Poincaré model.
    ///
    /// # Parameters
    /// * `u` - The node index
    /// * `x` - The new x-coordinate
    pub fn set_x(&mut self, u: usize, x: f32) {
        let u = node_index(u);
        self.drawing.set_x(u, x);
    }

    /// Sets the y-coordinate of a node in the hyperbolic plane
    ///
    /// When setting coordinates in hyperbolic space, you should ensure that
    /// points remain within the unit disk (x² + y² < 1) for the Poincaré model.
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

    /// Creates a new drawing with an initial random placement of nodes
    ///
    /// This method initializes a hyperbolic drawing with nodes placed
    /// randomly within the unit disk, with a bias toward the center to
    /// avoid extreme distortion at the boundary of the disk.
    ///
    /// # Parameters
    /// * `graph` - The graph to create a drawing for
    ///
    /// # Returns
    /// A new hyperbolic drawing with initial node positions
    #[staticmethod]
    pub fn initial_placement(graph: &PyGraphAdapter) -> PyObject {
        PyDrawing::new_drawing_hyperbolic_2d(match graph.graph() {
            GraphType::Graph(native_graph) => DrawingHyperbolic2d::initial_placement(native_graph),
            GraphType::DiGraph(native_graph) => {
                DrawingHyperbolic2d::initial_placement(native_graph)
            }
        })
    }
}
