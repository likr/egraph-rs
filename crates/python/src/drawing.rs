use crate::graph::{GraphType, IndexType, PyGraphAdapter};
use petgraph::graph::{node_index, NodeIndex};
use petgraph_drawing::{Drawing, DrawingEuclidean, DrawingEuclidean2d, DrawingTorus2d};
use pyo3::prelude::*;

type NodeId = NodeIndex<IndexType>;
pub enum DrawingType {
    Euclidean2d(DrawingEuclidean2d<NodeId, f32>),
    Euclidean(DrawingEuclidean<NodeId, f32>),
    Torus2d(DrawingTorus2d<NodeId, f32>),
}

#[pyclass]
#[pyo3(name = "Drawing")]
pub struct PyDrawing {
    drawing: DrawingType,
}

impl PyDrawing {
    pub fn new_drawing_2d(drawing: DrawingEuclidean2d<NodeId, f32>) -> Self {
        Self {
            drawing: DrawingType::Euclidean2d(drawing),
        }
    }

    pub fn new_drawing_torus(drawing: DrawingTorus2d<NodeId, f32>) -> Self {
        Self {
            drawing: DrawingType::Torus2d(drawing),
        }
    }

    pub fn drawing(&self) -> &DrawingType {
        &self.drawing
    }

    pub fn drawing_mut(&mut self) -> &mut DrawingType {
        &mut self.drawing
    }
}

#[pymethods]
impl PyDrawing {
    pub fn x(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        match self.drawing() {
            DrawingType::Euclidean2d(drawing) => drawing.x(u),
            DrawingType::Torus2d(drawing) => drawing.x(u),
            _ => unimplemented!(),
        }
    }

    pub fn y(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        match self.drawing() {
            DrawingType::Euclidean2d(drawing) => drawing.y(u),
            DrawingType::Torus2d(drawing) => drawing.y(u),
            _ => unimplemented!(),
        }
    }

    pub fn set_x(&mut self, u: usize, x: f32) {
        let u = node_index(u);
        match self.drawing_mut() {
            DrawingType::Euclidean2d(drawing) => drawing.set_x(u, x),
            DrawingType::Torus2d(drawing) => drawing.set_x(u, x),
            _ => unimplemented!(),
        };
    }

    pub fn set_y(&mut self, u: usize, y: f32) {
        let u = node_index(u);
        match self.drawing_mut() {
            DrawingType::Euclidean2d(drawing) => drawing.set_y(u, y),
            DrawingType::Torus2d(drawing) => drawing.set_y(u, y),
            _ => unimplemented!(),
        };
    }

    pub fn len(&self) -> usize {
        match self.drawing() {
            DrawingType::Euclidean2d(drawing) => drawing.len(),
            DrawingType::Torus2d(drawing) => drawing.len(),
            _ => unimplemented!(),
        }
    }

    pub fn centralize(&mut self) {
        match self.drawing_mut() {
            DrawingType::Euclidean2d(drawing) => drawing.centralize(),
            _ => unimplemented!(),
        };
    }

    pub fn clamp_region(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) {
        match self.drawing_mut() {
            DrawingType::Euclidean2d(drawing) => drawing.clamp_region(x0, y0, x1, y1),
            _ => unimplemented!(),
        };
    }

    pub fn edge_segments(&self, u: usize, v: usize) -> Option<Vec<((f32, f32), (f32, f32))>> {
        match self.drawing() {
            DrawingType::Euclidean2d(drawing) => drawing
                .edge_segments(node_index(u), node_index(v))
                .map(|segments| {
                    segments
                        .iter()
                        .map(|&(p, q)| ((p.0, p.1), (q.0, q.1)))
                        .collect::<Vec<_>>()
                }),
            DrawingType::Torus2d(drawing) => drawing
                .edge_segments(node_index(u), node_index(v))
                .map(|segments| {
                    segments
                        .iter()
                        .map(|&(p, q)| ((p.0 .0, p.1 .0), (q.0 .0, q.1 .0)))
                        .collect::<Vec<_>>()
                }),
            _ => unimplemented!(),
        }
    }
}

#[pyclass]
#[pyo3(name = "DrawingEuclidean")]
pub struct PyDrawingEuclidean;

#[pyclass]
#[pyo3(name = "DrawingEuclidean2d")]
pub struct PyDrawingEuclidean2d;

#[pymethods]
impl PyDrawingEuclidean2d {
    #[staticmethod]
    pub fn initial_placement(graph: &PyGraphAdapter) -> PyDrawing {
        PyDrawing::new_drawing_2d(match graph.graph() {
            GraphType::Graph(native_graph) => DrawingEuclidean2d::initial_placement(native_graph),
            GraphType::DiGraph(native_graph) => DrawingEuclidean2d::initial_placement(native_graph),
        })
    }
}

#[pyclass]
#[pyo3(name = "DrawingTorus2d")]
pub struct PyDrawingTorus2d;

#[pymethods]
impl PyDrawingTorus2d {
    #[staticmethod]
    pub fn initial_placement(graph: &PyGraphAdapter) -> PyDrawing {
        PyDrawing::new_drawing_torus(match graph.graph() {
            GraphType::Graph(native_graph) => DrawingTorus2d::initial_placement(native_graph),
            GraphType::DiGraph(native_graph) => DrawingTorus2d::initial_placement(native_graph),
        })
    }
}

pub fn register(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyDrawing>()?;
    m.add_class::<PyDrawingEuclidean2d>()?;
    m.add_class::<PyDrawingEuclidean>()?;
    m.add_class::<PyDrawingTorus2d>()?;
    Ok(())
}
