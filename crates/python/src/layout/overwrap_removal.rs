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
}

pub fn register(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyOverwrapRemoval>()?;
    Ok(())
}
