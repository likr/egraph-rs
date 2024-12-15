use crate::{
    drawing::PyDrawing,
    graph::{GraphType, NodeId, PyGraphAdapter},
};
use petgraph::graph::node_index;
use petgraph_drawing::{Drawing, DrawingEuclidean2d};
use pyo3::prelude::*;

#[pyclass(extends=PyDrawing)]
#[pyo3(name = "DrawingEuclidean2d")]
pub struct PyDrawingEuclidean2d {
    drawing: DrawingEuclidean2d<NodeId, f32>,
}

impl PyDrawingEuclidean2d {
    pub fn new(drawing: DrawingEuclidean2d<NodeId, f32>) -> Self {
        Self { drawing }
    }

    pub fn drawing(&self) -> &DrawingEuclidean2d<NodeId, f32> {
        &self.drawing
    }

    pub fn drawing_mut(&mut self) -> &mut DrawingEuclidean2d<NodeId, f32> {
        &mut self.drawing
    }
}

#[pymethods]
impl PyDrawingEuclidean2d {
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

    pub fn edge_segments(&self, u: usize, v: usize) -> Option<Vec<((f32, f32), (f32, f32))>> {
        self.drawing
            .edge_segments(node_index(u), node_index(v))
            .map(|segments| {
                segments
                    .iter()
                    .map(|&(p, q)| ((p.0, p.1), (q.0, q.1)))
                    .collect::<Vec<_>>()
            })
    }

    #[staticmethod]
    pub fn initial_placement(graph: &PyGraphAdapter) -> PyObject {
        PyDrawing::new_drawing_euclidean_2d(match graph.graph() {
            GraphType::Graph(native_graph) => DrawingEuclidean2d::initial_placement(native_graph),
            GraphType::DiGraph(native_graph) => DrawingEuclidean2d::initial_placement(native_graph),
        })
    }
}
