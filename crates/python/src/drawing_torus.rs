use crate::graph::IndexType;
use petgraph::graph::{node_index, NodeIndex};
use petgraph_drawing::{DrawingTorus, Metric2DTorus};
use pyo3::prelude::*;

type NodeId = NodeIndex<IndexType>;
type DrawingImpl = DrawingTorus<NodeId, f32>;

#[pyclass]
#[pyo3(name = "DrawingTorus")]
pub struct PyDrawingTorus {
    drawing: DrawingImpl,
}

impl PyDrawingTorus {
    pub fn new(drawing: DrawingImpl) -> Self {
        Self { drawing }
    }
    pub fn indices(&self) -> &[NodeId] {
        &self.drawing.indices
    }

    pub fn indices_mut(&mut self) -> &mut [NodeId] {
        &mut self.drawing.indices
    }

    pub fn coordinates(&self) -> &[Metric2DTorus<f32>] {
        &self.drawing.coordinates
    }

    pub fn coordinates_mut(&mut self) -> &mut [Metric2DTorus<f32>] {
        &mut self.drawing.coordinates
    }

    pub fn drawing(&self) -> &DrawingImpl {
        &self.drawing
    }

    pub fn drawing_mut(&mut self) -> &mut DrawingImpl {
        &mut self.drawing
    }

    pub fn position(&self, u: usize) -> Option<&Metric2DTorus<f32>> {
        let u = node_index(u);
        self.drawing.position(u)
    }

    pub fn set_position(&mut self, u: usize, p: Metric2DTorus<f32>) {
        let u = node_index(u);
        self.drawing.position_mut(u).map(|q| *q = p);
    }
}

#[pymethods]
impl PyDrawingTorus {
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
}

pub fn register(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyDrawingTorus>()?;
    Ok(())
}
