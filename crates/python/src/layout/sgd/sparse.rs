use crate::{
    distance_matrix::{
        with_distance, DistanceMatrixType, EdgeLengthFn, PyDistanceMatrix, PyPivotedNegLogDistance,
        PyPivotedNegLogSimDistance,
    },
    graph::{GraphType, PyGraphAdapter},
    layout::sgd::PySgd,
    rng::PyRng,
    FloatType,
};
use petgraph::visit::{EdgeRef, IntoNodeIdentifiers};
use petgraph_layout_sgd::SparseSgd;
use pyo3::prelude::*;
use rand::SeedableRng;

fn merge_edge_length_arg<'py>(
    f: Option<&Bound<'py, PyAny>>,
    length_fn: Option<&Bound<'py, PyAny>>,
) -> PyResult<EdgeLengthFn> {
    if f.is_some() && length_fn.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot specify both f and length_fn",
        ));
    }
    EdgeLengthFn::from_bound(length_fn.or(f))
}

fn collect_pivot_nodes(
    native_graph: &petgraph::graph::UnGraph<Py<PyAny>, Py<PyAny>, crate::graph::IndexType>,
    pivot: &[usize],
) -> PyResult<Vec<petgraph::graph::NodeIndex<crate::graph::IndexType>>> {
    let nodes = native_graph.node_identifiers().collect::<Vec<_>>();
    pivot
        .iter()
        .map(|&i| {
            nodes.get(i).copied().ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err(format!(
                    "Pivot index {i} out of bounds for graph with {} nodes",
                    nodes.len()
                ))
            })
        })
        .collect()
}

/// Sparse Stochastic Gradient Descent (SGD) layout builder.
#[pyclass]
#[pyo3(name = "SparseSgd")]
pub struct PySparseSgd {
    builder: SparseSgd,
}

#[pymethods]
impl PySparseSgd {
    /// Creates a new `SparseSgd` builder.
    #[new]
    fn new() -> Self {
        Self {
            builder: SparseSgd::new(),
        }
    }

    /// Sets the number of pivot nodes `h`.
    fn h(mut slf: PyRefMut<Self>, h: usize) -> Py<Self> {
        slf.builder.h(h);
        slf.into()
    }

    /// Builds an `Sgd` schedule using random max-min pivots.
    ///
    /// Edge length is optional: passing `None` (or omitting `f`/`length_fn`) uses uniform `1.0`
    /// edge lengths without acquiring the GIL.
    #[pyo3(signature = (graph, f = None, rng = None, *, length_fn = None))]
    fn build(
        &self,
        graph: &PyGraphAdapter,
        f: Option<&Bound<PyAny>>,
        rng: Option<&Bound<PyRng>>,
        length_fn: Option<&Bound<PyAny>>,
    ) -> PyResult<PySgd> {
        let (edge_length, rng_opt) = if let Some(rng_bound) = rng {
            (
                merge_edge_length_arg(f, length_fn)?,
                Some(rng_bound.borrow_mut()),
            )
        } else if let (Some(first), None) = (f, length_fn) {
            if let Ok(rng_bound) = first.cast::<PyRng>() {
                (EdgeLengthFn::Constant(1.0), Some(rng_bound.borrow_mut()))
            } else {
                (EdgeLengthFn::from_bound(Some(first))?, None)
            }
        } else {
            (merge_edge_length_arg(f, length_fn)?, None)
        };

        let sgd = match graph.graph() {
            GraphType::Graph(native_graph) => {
                let len_closure =
                    |e: petgraph::graph::EdgeReference<Py<PyAny>>| edge_length.eval(e.id().index());
                match rng_opt {
                    Some(mut r) => self.builder.build(native_graph, len_closure, r.get_mut()),
                    None => {
                        let mut local = rand::rngs::StdRng::from_entropy();
                        self.builder.build(native_graph, len_closure, &mut local)
                    }
                }
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "unsupported graph type",
                ))
            }
        };
        Ok(PySgd::new_with_sgd(sgd))
    }

    /// Builds an `Sgd` schedule with user-specified pivot node indices.
    #[pyo3(signature = (graph, f = None, pivot = None, *, length_fn = None))]
    pub fn new_with_pivot(
        &self,
        graph: &PyGraphAdapter,
        f: Option<&Bound<PyAny>>,
        pivot: Option<Vec<usize>>,
        length_fn: Option<&Bound<PyAny>>,
    ) -> PyResult<PySgd> {
        let (edge_length, pivot_vec) = match (pivot, f, length_fn) {
            (Some(p), f_arg, lf_arg) => (merge_edge_length_arg(f_arg, lf_arg)?, p),
            (None, Some(first), lf_arg) => {
                let p = first.extract::<Vec<usize>>().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err("pivot argument is required")
                })?;
                (EdgeLengthFn::from_bound(lf_arg)?, p)
            }
            (None, None, _) => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "pivot argument is required",
                ))
            }
        };

        let sgd = match graph.graph() {
            GraphType::Graph(native_graph) => {
                let pivot_nodes = collect_pivot_nodes(native_graph, &pivot_vec)?;
                self.builder.build_with_pivot(
                    native_graph,
                    |e| edge_length.eval(e.id().index()),
                    &pivot_nodes,
                )
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "unsupported graph type",
                ))
            }
        };
        Ok(PySgd::new_with_sgd(sgd))
    }

    /// Builds an `Sgd` schedule using given pivot indices and a precomputed distance matrix.
    #[pyo3(signature = (graph, f = None, pivot = None, d = None, *, length_fn = None))]
    pub fn build_with_pivot_and_distance_matrix(
        &self,
        graph: &PyGraphAdapter,
        f: Option<&Bound<PyAny>>,
        pivot: Option<&Bound<PyAny>>,
        d: Option<&Bound<PyAny>>,
        length_fn: Option<&Bound<PyAny>>,
    ) -> PyResult<PySgd> {
        let (edge_length, pivot_vec, d_bound) = if let Some(d_arg) = d {
            let p = pivot
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("pivot is required"))?
                .extract::<Vec<usize>>()?;
            (merge_edge_length_arg(f, length_fn)?, p, d_arg)
        } else if let (Some(p_arg), Some(d_arg)) = (f, pivot) {
            if length_fn.is_some() {
                return Err(pyo3::exceptions::PyValueError::new_err("d is required"));
            }
            let p = p_arg.extract::<Vec<usize>>()?;
            (EdgeLengthFn::Constant(1.0), p, d_arg)
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pivot and distance matrix d are required",
            ));
        };

        match graph.graph() {
            GraphType::Graph(native_graph) => {
                let pivot_nodes = collect_pivot_nodes(native_graph, &pivot_vec)?;
                let sgd = with_distance(d_bound, |distance| {
                    self.builder.build_with_pivot_and_distance_matrix(
                        native_graph,
                        |e| edge_length.eval(e.id().index()),
                        &pivot_nodes,
                        distance,
                    )
                })?;
                Ok(PySgd::new_with_sgd(sgd))
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "unsupported graph type",
            )),
        }
    }

    /// Builds an `Sgd` schedule using a pivoted distance matrix (`PivotedNegLogSimDistance`,
    /// `PivotedNegLogDistance`, or pivoted `DistanceMatrix`).
    #[pyo3(signature = (graph, f = None, d = None, *, length_fn = None))]
    pub fn build_with_pivoted_distance(
        &self,
        graph: &PyGraphAdapter,
        f: Option<&Bound<PyAny>>,
        d: Option<&Bound<PyAny>>,
        length_fn: Option<&Bound<PyAny>>,
    ) -> PyResult<PySgd> {
        let (edge_length, d_bound) = if let Some(d_arg) = d {
            (merge_edge_length_arg(f, length_fn)?, d_arg)
        } else if let Some(d_arg) = f {
            (EdgeLengthFn::from_bound(length_fn)?, d_arg)
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pivoted distance matrix d is required",
            ));
        };

        let GraphType::Graph(native_graph) = graph.graph() else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "unsupported graph type",
            ));
        };
        let len_closure = |e: petgraph::graph::EdgeReference<Py<PyAny>>| -> FloatType {
            edge_length.eval(e.id().index())
        };

        if let Ok(dm) = d_bound.extract::<PyRef<PyPivotedNegLogSimDistance>>() {
            let sgd = self.builder.build_with_pivoted_neg_log_sim_distance(
                native_graph,
                len_closure,
                &dm.matrix,
            );
            Ok(PySgd::new_with_sgd(sgd))
        } else if let Ok(dm) = d_bound.extract::<PyRef<PyPivotedNegLogDistance>>() {
            let sgd = self.builder.build_with_pivoted_neg_log_distance(
                native_graph,
                len_closure,
                &dm.matrix,
            );
            Ok(PySgd::new_with_sgd(sgd))
        } else if let Ok(dm) = d_bound.extract::<PyRef<PyDistanceMatrix>>() {
            match dm.distance_matrix() {
                DistanceMatrixType::Pivoted(pivoted_dm) => {
                    let sgd = self.builder.build_with_pivoted_distance_matrix(
                        native_graph,
                        len_closure,
                        pivoted_dm,
                    );
                    Ok(PySgd::new_with_sgd(sgd))
                }
                _ => Err(pyo3::exceptions::PyTypeError::new_err(
                    "Expected a pivoted distance matrix",
                )),
            }
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Expected PivotedNegLogSimDistance, PivotedNegLogDistance, or PivotedDistanceMatrix",
            ))
        }
    }
}
