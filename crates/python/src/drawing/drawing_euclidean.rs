/// N-dimensional Euclidean drawing implementation for Python
///
/// This module provides a Python binding for N-dimensional Euclidean drawings.
/// These drawings place nodes in an N-dimensional Euclidean space, where each node
/// has coordinates in each dimension.
use crate::{drawing::PyDrawing, graph::NodeId};
use petgraph::graph::node_index;
use petgraph_drawing::{Drawing, DrawingEuclidean};
use pyo3::prelude::*;

/// Python class for N-dimensional Euclidean drawings
///
/// This class represents a drawing in N-dimensional Euclidean space, where each node
/// is assigned a position with coordinates in each of the N dimensions. This allows
/// for representing nodes in spaces with more than 2 or 3 dimensions, which can be
/// useful for certain visualization or analysis techniques.
#[pyclass(extends=PyDrawing)]
#[pyo3(name = "DrawingEuclidean")]
pub struct PyDrawingEuclidean {
    drawing: DrawingEuclidean<NodeId, f32>,
}

impl PyDrawingEuclidean {
    pub fn new(drawing: DrawingEuclidean<NodeId, f32>) -> Self {
        Self { drawing }
    }

    pub fn drawing(&self) -> &DrawingEuclidean<NodeId, f32> {
        &self.drawing
    }

    pub fn drawing_mut(&mut self) -> &mut DrawingEuclidean<NodeId, f32> {
        &mut self.drawing
    }
}

#[pymethods]
impl PyDrawingEuclidean {
    /// Gets the coordinate of a node in a specific dimension
    ///
    /// This method retrieves the position coordinate of a node in the specified dimension.
    ///
    /// # Parameters
    /// * `u` - The node index
    /// * `d` - The dimension index (0, 1, 2, etc.)
    ///
    /// # Returns
    /// The coordinate value if the node exists, None otherwise
    pub fn get(&self, u: usize, d: usize) -> Option<f32> {
        let u = node_index(u);
        self.drawing.get(u, d)
    }

    /// Sets the coordinate of a node in a specific dimension
    ///
    /// This method updates the position coordinate of a node in the specified dimension.
    ///
    /// # Parameters
    /// * `u` - The node index
    /// * `d` - The dimension index (0, 1, 2, etc.)
    /// * `value` - The new coordinate value
    pub fn set(&mut self, u: usize, d: usize, value: f32) {
        let u = node_index(u);
        self.drawing.set(u, d, value);
    }

    /// Returns the number of nodes in the drawing
    ///
    /// # Returns
    /// The number of nodes
    pub fn len(&self) -> usize {
        self.drawing.len()
    }
}
