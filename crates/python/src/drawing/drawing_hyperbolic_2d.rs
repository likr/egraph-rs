use crate::{
    drawing::PyDrawing,
    graph::{GraphType, NodeId, PyGraphAdapter},
};
use petgraph::graph::node_index;
use petgraph_drawing::{Drawing, DrawingHyperbolic2d};
use pyo3::prelude::*;

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
    pub fn x(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.drawing.x(u)
    }

    pub fn y(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.drawing.y(u)
    }

    pub fn set_x(&mut self, u: usize, x: f32) {
        let u = node_index(u);
        self.drawing.set_x(u, x);
    }

    pub fn set_y(&mut self, u: usize, y: f32) {
        let u = node_index(u);
        self.drawing.set_y(u, y);
    }

    pub fn len(&self) -> usize {
        self.drawing.len()
    }

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
