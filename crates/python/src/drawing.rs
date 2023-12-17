use crate::graph::{GraphType, IndexType, PyGraphAdapter};
use petgraph::graph::{node_index, NodeIndex};
use petgraph_drawing::{Drawing2D, DrawingTorus};
use pyo3::prelude::*;

type NodeId = NodeIndex<IndexType>;
pub enum DrawingType {
    Drawing2D(Drawing2D<NodeId, f32>),
    DrawingTorus(DrawingTorus<NodeId, f32>),
}

#[pyclass]
#[pyo3(name = "Drawing")]
pub struct PyDrawing {
    drawing: DrawingType,
}

impl PyDrawing {
    pub fn new_drawing_2d(drawing: Drawing2D<NodeId, f32>) -> Self {
        Self {
            drawing: DrawingType::Drawing2D(drawing),
        }
    }

    pub fn new_drawing_torus(drawing: DrawingTorus<NodeId, f32>) -> Self {
        Self {
            drawing: DrawingType::DrawingTorus(drawing),
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
            DrawingType::Drawing2D(drawing) => drawing.x(u),
            DrawingType::DrawingTorus(drawing) => drawing.x(u),
        }
    }

    pub fn y(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        match self.drawing() {
            DrawingType::Drawing2D(drawing) => drawing.y(u),
            DrawingType::DrawingTorus(drawing) => drawing.y(u),
        }
    }

    pub fn set_x(&mut self, u: usize, x: f32) {
        let u = node_index(u);
        match self.drawing_mut() {
            DrawingType::Drawing2D(drawing) => drawing.set_x(u, x),
            DrawingType::DrawingTorus(drawing) => drawing.set_x(u, x),
        };
    }

    pub fn set_y(&mut self, u: usize, y: f32) {
        let u = node_index(u);
        match self.drawing_mut() {
            DrawingType::Drawing2D(drawing) => drawing.set_y(u, y),
            DrawingType::DrawingTorus(drawing) => drawing.set_y(u, y),
        };
    }

    pub fn len(&self) -> usize {
        match self.drawing() {
            DrawingType::Drawing2D(drawing) => drawing.len(),
            DrawingType::DrawingTorus(drawing) => drawing.len(),
        }
    }

    pub fn centralize(&mut self) {
        match self.drawing_mut() {
            DrawingType::Drawing2D(drawing) => drawing.centralize(),
            _ => unimplemented!(),
        };
    }

    pub fn clamp_region(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) {
        match self.drawing_mut() {
            DrawingType::Drawing2D(drawing) => drawing.clamp_region(x0, y0, x1, y1),
            _ => unimplemented!(),
        };
    }
}

#[pyclass]
#[pyo3(name = "Drawing2D")]
pub struct PyDrawing2D;

#[pymethods]
impl PyDrawing2D {
    #[staticmethod]
    pub fn initial_placement(graph: &PyGraphAdapter) -> PyDrawing {
        PyDrawing::new_drawing_2d(match graph.graph() {
            GraphType::Graph(native_graph) => Drawing2D::initial_placement(native_graph),
            GraphType::DiGraph(native_graph) => Drawing2D::initial_placement(native_graph),
        })
    }
}

#[pyclass]
#[pyo3(name = "DrawingTorus")]
pub struct PyDrawingTorus;

#[pymethods]
impl PyDrawingTorus {
    #[staticmethod]
    pub fn initial_placement(graph: &PyGraphAdapter) -> PyDrawing {
        PyDrawing::new_drawing_torus(match graph.graph() {
            GraphType::Graph(native_graph) => DrawingTorus::initial_placement(native_graph),
            GraphType::DiGraph(native_graph) => DrawingTorus::initial_placement(native_graph),
        })
    }
}

pub fn register(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyDrawing>()?;
    m.add_class::<PyDrawing2D>()?;
    m.add_class::<PyDrawingTorus>()?;
    Ok(())
}
