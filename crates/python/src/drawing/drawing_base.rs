use crate::{
    drawing::{PyDrawingEuclidean, PyDrawingEuclidean2d, PyDrawingSpherical2d, PyDrawingTorus2d},
    graph::NodeId,
};
use petgraph_drawing::{
    DrawingEuclidean, DrawingEuclidean2d, DrawingHyperbolic2d, DrawingSpherical2d, DrawingTorus2d,
};
use pyo3::prelude::*;

use super::PyDrawingHyperbolic2d;

/// Enum representing the different types of drawing spaces supported
///
/// This enum is used to track which type of geometric space a drawing exists in.
#[derive(Clone, Copy)]
pub enum DrawingType {
    /// Two-dimensional Euclidean space
    Euclidean2d,
    /// N-dimensional Euclidean space
    Euclidean,
    /// Two-dimensional Hyperbolic space
    Hyperbolic2d,
    /// Two-dimensional Spherical space
    Spherical2d,
    /// Two-dimensional Torus space
    Torus2d,
}

/// Base class for all drawing types
///
/// A drawing maps graph nodes to positions in a geometric space.
/// This class serves as the base for specific drawing implementations
/// like Euclidean, Hyperbolic, Spherical, and Torus spaces.
#[pyclass(subclass)]
#[pyo3(name = "Drawing")]
pub struct PyDrawing {
    drawing_type: DrawingType,
}

impl PyDrawing {
    /// Creates a new Euclidean 2D drawing object
    ///
    /// # Parameters
    /// * `drawing` - The native Rust drawing object
    ///
    /// # Returns
    /// A Python drawing object
    pub fn new_drawing_euclidean_2d(drawing: DrawingEuclidean2d<NodeId, f32>) -> PyObject {
        let base = PyClassInitializer::from(Self {
            drawing_type: DrawingType::Euclidean2d,
        });
        let py_drawing = base.add_subclass(PyDrawingEuclidean2d::new(drawing));
        Python::with_gil(|py| Py::new(py, py_drawing).unwrap().to_object(py))
    }

    /// Creates a new N-dimensional Euclidean drawing object
    ///
    /// # Parameters
    /// * `drawing` - The native Rust drawing object
    ///
    /// # Returns
    /// A Python drawing object
    pub fn new_drawing_euclidean(drawing: DrawingEuclidean<NodeId, f32>) -> PyObject {
        let base = PyClassInitializer::from(Self {
            drawing_type: DrawingType::Euclidean,
        });
        let py_drawing = base.add_subclass(PyDrawingEuclidean::new(drawing));
        Python::with_gil(|py| Py::new(py, py_drawing).unwrap().to_object(py))
    }

    /// Creates a new Hyperbolic 2D drawing object
    ///
    /// # Parameters
    /// * `drawing` - The native Rust drawing object
    ///
    /// # Returns
    /// A Python drawing object
    pub fn new_drawing_hyperbolic_2d(drawing: DrawingHyperbolic2d<NodeId, f32>) -> PyObject {
        let base = PyClassInitializer::from(Self {
            drawing_type: DrawingType::Hyperbolic2d,
        });
        let py_drawing = base.add_subclass(PyDrawingHyperbolic2d::new(drawing));
        Python::with_gil(|py| Py::new(py, py_drawing).unwrap().to_object(py))
    }

    /// Creates a new Spherical 2D drawing object
    ///
    /// # Parameters
    /// * `drawing` - The native Rust drawing object
    ///
    /// # Returns
    /// A Python drawing object
    pub fn new_drawing_spherical_2d(drawing: DrawingSpherical2d<NodeId, f32>) -> PyObject {
        let base = PyClassInitializer::from(Self {
            drawing_type: DrawingType::Spherical2d,
        });
        let py_drawing = base.add_subclass(PyDrawingSpherical2d::new(drawing));
        Python::with_gil(|py| Py::new(py, py_drawing).unwrap().to_object(py))
    }

    /// Creates a new Torus 2D drawing object
    ///
    /// # Parameters
    /// * `drawing` - The native Rust drawing object
    ///
    /// # Returns
    /// A Python drawing object
    pub fn new_drawing_torus_2d(drawing: DrawingTorus2d<NodeId, f32>) -> PyObject {
        let base = PyClassInitializer::from(Self {
            drawing_type: DrawingType::Torus2d,
        });
        let py_drawing = base.add_subclass(PyDrawingTorus2d::new(drawing));
        Python::with_gil(|py| Py::new(py, py_drawing).unwrap().to_object(py))
    }

    /// Returns the type of this drawing
    ///
    /// # Returns
    /// The drawing type (Euclidean2d, Euclidean, Hyperbolic2d, etc.)
    pub fn drawing_type(&self) -> DrawingType {
        self.drawing_type
    }
}
