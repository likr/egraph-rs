use petgraph_layout_overwrap_removal::OverwrapRemoval;
use pyo3::prelude::*;

use crate::{
    drawing::{
        PyDrawingEuclidean, PyDrawingEuclidean2d, PyDrawingHyperbolic2d, PyDrawingSpherical2d,
        PyDrawingTorus2d,
    },
    graph::{GraphType, PyGraphAdapter},
};

#[pyclass]
#[pyo3(name = "OverwrapRemoval")]
struct PyOverwrapRemoval {
    overwrap_removal: OverwrapRemoval<f32>,
}

#[pymethods]
impl PyOverwrapRemoval {
    #[new]
    fn new(graph: &PyGraphAdapter, f: &Bound<PyAny>) -> PyOverwrapRemoval {
        match graph.graph() {
            GraphType::Graph(native_graph) => PyOverwrapRemoval {
                overwrap_removal: OverwrapRemoval::new(native_graph, |u| {
                    f.call1((u.index(),)).unwrap().extract().unwrap()
                }),
            },
            _ => panic!("unsupported graph type"),
        }
    }

    fn apply_with_drawing_euclidean_2d(&self, drawing: &mut PyDrawingEuclidean2d) {
        self.overwrap_removal.apply(drawing.drawing_mut());
    }

    fn apply_with_drawing_euclidean(&self, drawing: &mut PyDrawingEuclidean) {
        self.overwrap_removal.apply(drawing.drawing_mut());
    }

    fn apply_with_drawing_hyperbolic_2d(&self, drawing: &mut PyDrawingHyperbolic2d) {
        self.overwrap_removal.apply(drawing.drawing_mut());
    }

    fn apply_with_drawing_spherical_2d(&self, drawing: &mut PyDrawingSpherical2d) {
        self.overwrap_removal.apply(drawing.drawing_mut());
    }

    fn apply_with_drawing_torus_2d(&self, drawing: &mut PyDrawingTorus2d) {
        self.overwrap_removal.apply(drawing.drawing_mut());
    }

    #[getter]
    fn get_strength(&self) -> f32 {
        self.overwrap_removal.strength
    }

    #[setter]
    fn set_strength(&mut self, value: f32) {
        self.overwrap_removal.strength = value;
    }

    #[getter]
    fn get_iterations(&self) -> usize {
        self.overwrap_removal.iterations
    }

    #[setter]
    fn set_iterations(&mut self, value: usize) {
        self.overwrap_removal.iterations = value;
    }

    #[getter]
    fn get_min_distance(&self) -> f32 {
        self.overwrap_removal.min_distance
    }

    #[setter]
    fn set_min_distance(&mut self, value: f32) {
        self.overwrap_removal.min_distance = value;
    }
}

pub fn register(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyOverwrapRemoval>()?;
    Ok(())
}
