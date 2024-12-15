use crate::{drawing::PyDrawing, graph::NodeId};
use petgraph::graph::node_index;
use petgraph_drawing::DrawingEuclidean;
use pyo3::prelude::*;

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
    pub fn get(&self, u: usize, d: usize) -> Option<f32> {
        let u = node_index(u);
        self.drawing.get(u, d)
    }

    pub fn set_x(&mut self, u: usize, d: usize, value: f32) {
        let u = node_index(u);
        self.drawing.set(u, d, value);
    }
}
