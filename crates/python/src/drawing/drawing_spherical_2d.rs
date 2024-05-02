use crate::{
    drawing::PyDrawing,
    graph::{GraphType, NodeId, PyGraphAdapter},
};
use petgraph::graph::node_index;
use petgraph_drawing::{Drawing, DrawingSpherical2d};
use pyo3::prelude::*;

#[pyclass(extends=PyDrawing)]
#[pyo3(name = "DrawingSpherical2d")]
pub struct PyDrawingSpherical2d {
    drawing: DrawingSpherical2d<NodeId, f32>,
}

impl PyDrawingSpherical2d {
    pub fn new(drawing: DrawingSpherical2d<NodeId, f32>) -> Self {
        Self { drawing }
    }

    pub fn drawing(&self) -> &DrawingSpherical2d<NodeId, f32> {
        &self.drawing
    }

    pub fn drawing_mut(&mut self) -> &mut DrawingSpherical2d<NodeId, f32> {
        &mut self.drawing
    }
}

#[pymethods]
impl PyDrawingSpherical2d {
    pub fn lon(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.drawing.lon(u)
    }

    pub fn lat(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.drawing.lat(u)
    }

    pub fn set_lon(&mut self, u: usize, value: f32) {
        let u = node_index(u);
        self.drawing.set_lon(u, value);
    }

    pub fn set_lat(&mut self, u: usize, value: f32) {
        let u = node_index(u);
        self.drawing.set_lat(u, value);
    }

    pub fn len(&self) -> usize {
        self.drawing.len()
    }

    #[staticmethod]
    pub fn initial_placement(graph: &PyGraphAdapter) -> PyObject {
        PyDrawing::new_drawing_spherical_2d(match graph.graph() {
            GraphType::Graph(native_graph) => DrawingSpherical2d::initial_placement(native_graph),
            GraphType::DiGraph(native_graph) => DrawingSpherical2d::initial_placement(native_graph),
        })
    }
}
