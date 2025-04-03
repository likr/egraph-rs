use crate::{
    drawing::{PyDrawingEuclidean, PyDrawingEuclidean2d, PyDrawingSpherical2d, PyDrawingTorus2d},
    graph::NodeId,
};
use petgraph_drawing::{
    DrawingEuclidean, DrawingEuclidean2d, DrawingHyperbolic2d, DrawingSpherical2d, DrawingTorus2d,
};
use pyo3::prelude::*;

use super::PyDrawingHyperbolic2d;

#[derive(Clone, Copy)]
pub enum DrawingType {
    Euclidean2d,
    Euclidean,
    Hyperbolic2d,
    Spherical2d,
    Torus2d,
}

#[pyclass(subclass)]
#[pyo3(name = "Drawing")]
pub struct PyDrawing {
    drawing_type: DrawingType,
}

impl PyDrawing {
    pub fn new_drawing_euclidean_2d(drawing: DrawingEuclidean2d<NodeId, f32>) -> PyObject {
        let base = PyClassInitializer::from(Self {
            drawing_type: DrawingType::Euclidean2d,
        });
        let py_drawing = base.add_subclass(PyDrawingEuclidean2d::new(drawing));
        Python::with_gil(|py| Py::new(py, py_drawing).unwrap().to_object(py))
    }

    pub fn new_drawing_euclidean(drawing: DrawingEuclidean<NodeId, f32>) -> PyObject {
        let base = PyClassInitializer::from(Self {
            drawing_type: DrawingType::Euclidean,
        });
        let py_drawing = base.add_subclass(PyDrawingEuclidean::new(drawing));
        Python::with_gil(|py| Py::new(py, py_drawing).unwrap().to_object(py))
    }

    pub fn new_drawing_hyperbolic_2d(drawing: DrawingHyperbolic2d<NodeId, f32>) -> PyObject {
        let base = PyClassInitializer::from(Self {
            drawing_type: DrawingType::Hyperbolic2d,
        });
        let py_drawing = base.add_subclass(PyDrawingHyperbolic2d::new(drawing));
        Python::with_gil(|py| Py::new(py, py_drawing).unwrap().to_object(py))
    }

    pub fn new_drawing_spherical_2d(drawing: DrawingSpherical2d<NodeId, f32>) -> PyObject {
        let base = PyClassInitializer::from(Self {
            drawing_type: DrawingType::Spherical2d,
        });
        let py_drawing = base.add_subclass(PyDrawingSpherical2d::new(drawing));
        Python::with_gil(|py| Py::new(py, py_drawing).unwrap().to_object(py))
    }

    pub fn new_drawing_torus_2d(drawing: DrawingTorus2d<NodeId, f32>) -> PyObject {
        let base = PyClassInitializer::from(Self {
            drawing_type: DrawingType::Torus2d,
        });
        let py_drawing = base.add_subclass(PyDrawingTorus2d::new(drawing));
        Python::with_gil(|py| Py::new(py, py_drawing).unwrap().to_object(py))
    }

    pub fn drawing_type(&self) -> DrawingType {
        self.drawing_type
    }
}
