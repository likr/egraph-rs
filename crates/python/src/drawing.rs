use crate::graph::{GraphType, IndexType, PyGraphAdapter};
use ndarray::Array2;
use petgraph::graph::{node_index, NodeIndex};
use petgraph_drawing::Drawing;
use pyo3::prelude::*;

type NodeId = NodeIndex<IndexType>;
type DrawingImpl = Drawing<NodeId, f32>;

#[pyclass]
#[pyo3(name = "Drawing")]
pub struct PyDrawing {
    drawing: DrawingImpl,
}

impl PyDrawing {
    pub fn new(drawing: DrawingImpl) -> Self {
        Self { drawing }
    }
    pub fn indices(&self) -> &[NodeId] {
        &self.drawing.indices
    }

    pub fn indices_mut(&mut self) -> &mut [NodeId] {
        &mut self.drawing.indices
    }

    pub fn coordinates(&self) -> &Array2<f32> {
        &self.drawing.coordinates
    }

    pub fn coordinates_mut(&mut self) -> &mut Array2<f32> {
        &mut self.drawing.coordinates
    }

    pub fn drawing(&self) -> &DrawingImpl {
        &self.drawing
    }

    pub fn drawing_mut(&mut self) -> &mut DrawingImpl {
        &mut self.drawing
    }

    pub fn position(&self, u: usize) -> Option<(f32, f32)> {
        let u = node_index(u);
        self.drawing.position(u)
    }

    pub fn set_position(&mut self, u: usize, p: (f32, f32)) {
        let u = node_index(u);
        self.drawing.set_position(u, p);
    }
}

#[pymethods]
impl PyDrawing {
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

    pub fn centralize(&mut self) {
        self.drawing.centralize();
    }

    pub fn clamp_region(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) {
        self.drawing.clamp_region(x0, y0, x1, y1);
    }

    #[staticmethod]
    pub fn initial_placement(graph: &PyGraphAdapter) -> Self {
        Self::new(match graph.graph() {
            GraphType::Graph(native_graph) => Drawing::initial_placement(native_graph),
            GraphType::DiGraph(native_graph) => Drawing::initial_placement(native_graph),
        })
    }
}

pub fn register(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyDrawing>()?;
    Ok(())
}
