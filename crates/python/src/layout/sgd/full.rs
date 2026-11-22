use crate::{
    distance_matrix::with_full_distance,
    graph::{GraphType, PyGraphAdapter},
    layout::sgd::PySgd,
};
use petgraph::visit::EdgeRef;
use petgraph_layout_sgd::FullSgd;
use pyo3::prelude::*;

#[pyclass]
#[pyo3(name = "FullSgd")]
pub struct PyFullSgd {
    builder: FullSgd,
}

#[pymethods]
impl PyFullSgd {
    #[new]
    fn new() -> Self {
        Self {
            builder: FullSgd::new(),
        }
    }

    #[pyo3(signature = (graph, f=None))]
    fn build(&self, graph: &PyGraphAdapter, f: Option<&Bound<PyAny>>) -> PyResult<PySgd> {
        let edge_len = crate::graph::EdgeLength::from_py(f)?;
        let sgd = match graph.graph() {
            GraphType::Graph(native_graph) => {
                let resolved_len = edge_len.resolve_for_graph(native_graph, true)?;
                self.builder
                    .build(native_graph, |e| resolved_len.get(e.id().index()))
            }
            _ => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "unsupported graph type for FullSgd",
                ));
            }
        };
        Ok(PySgd::new_with_sgd(sgd))
    }

    #[pyo3(signature = (n, d, w=None))]
    fn build_with_numpy(
        &self,
        n: usize,
        d: numpy::PyReadonlyArray2<'_, crate::FloatType>,
        w: Option<numpy::PyReadonlyArray2<'_, crate::FloatType>>,
    ) -> PyResult<PySgd> {
        let d_view = d.as_array();
        if d_view.dim() != (n, n) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "distance matrix shape {:?} does not match expected ({}, {})",
                d_view.dim(),
                n,
                n
            )));
        }
        let w_view = match &w {
            Some(w_arr) => {
                let view = w_arr.as_array();
                if view.dim() != (n, n) {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "weight matrix shape {:?} does not match expected ({}, {})",
                        view.dim(),
                        n,
                        n
                    )));
                }
                Some(view)
            }
            None => None,
        };
        Ok(PySgd::new_with_sgd(
            self.builder.build_with_arrays(n, d_view, w_view),
        ))
    }

    fn build_with_distance_matrix(&self, d: &Bound<PyAny>) -> PyResult<PySgd> {
        let sgd = with_full_distance(d, |distance| {
            self.builder
                .try_build_with_distance_matrix(distance)
                .map_err(pyo3::exceptions::PyValueError::new_err)
        })??;
        Ok(PySgd::new_with_sgd(sgd))
    }
}
