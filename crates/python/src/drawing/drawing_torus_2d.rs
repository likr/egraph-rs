use crate::{
    drawing::PyDrawing,
    graph::{GraphType, NodeId, PyGraphAdapter},
};

type Point2D = (f32, f32);
type Segment2D = (Point2D, Point2D);
use petgraph::graph::node_index;
use petgraph_drawing::{Drawing, DrawingTorus2d};
use pyo3::prelude::*;

#[pyclass(extends=PyDrawing)]
#[pyo3(name = "DrawingTorus2d")]
pub struct PyDrawingTorus2d {
    drawing: DrawingTorus2d<NodeId, f32>,
}

impl PyDrawingTorus2d {
    pub fn new(drawing: DrawingTorus2d<NodeId, f32>) -> Self {
        Self { drawing }
    }

    pub fn drawing(&self) -> &DrawingTorus2d<NodeId, f32> {
        &self.drawing
    }

    pub fn drawing_mut(&mut self) -> &mut DrawingTorus2d<NodeId, f32> {
        &mut self.drawing
    }
}

#[pymethods]
impl PyDrawingTorus2d {
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

    pub fn is_empty(&self) -> bool {
        self.drawing.is_empty()
    }

    pub fn edge_segments(&self, u: usize, v: usize) -> Option<Vec<Segment2D>> {
        self.drawing
            .edge_segments(node_index(u), node_index(v))
            .map(|segments| {
                segments
                    .iter()
                    .map(|&(p, q)| ((p.0 .0, p.1 .0), (q.0 .0, q.1 .0)))
                    .collect::<Vec<_>>()
            })
    }

    #[staticmethod]
    pub fn initial_placement(graph: &PyGraphAdapter) -> PyObject {
        PyDrawing::new_drawing_torus_2d(match graph.graph() {
            GraphType::Graph(native_graph) => DrawingTorus2d::initial_placement(native_graph),
            GraphType::DiGraph(native_graph) => DrawingTorus2d::initial_placement(native_graph),
        })
    }
}
