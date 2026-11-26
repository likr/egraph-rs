//! Graph Laplacian and edge-length bindings.

use crate::{
    FloatType,
    graph::{GraphType, IndexType, PyGraphAdapter},
};
use petgraph::visit::EdgeRef;
use petgraph_linalg_kernel::SparseSymmetricMatrix;
use pyo3::prelude::*;

/// Edge length abstraction allowing optional/constant edge weights without acquiring the GIL.
pub(crate) enum EdgeLengthFn {
    Constant(FloatType),
    Callable(Py<PyAny>),
}

impl EdgeLengthFn {
    pub(crate) fn from_bound(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let Some(bound) = obj else {
            return Ok(Self::Constant(1.0));
        };
        if bound.is_none() {
            return Ok(Self::Constant(1.0));
        }
        if bound.is_instance_of::<pyo3::types::PyBool>() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "edge length must be callable, float, or None, not bool",
            ));
        }
        if !bound.is_callable()
            && let Ok(val) = bound.extract::<FloatType>()
        {
            if !val.is_finite() || val < 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "edge length must be finite and non-negative",
                ));
            }
            return Ok(Self::Constant(val));
        }
        if bound.is_callable() {
            return Ok(Self::Callable(bound.clone().unbind()));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "edge length must be callable, float, or None",
        ))
    }

    pub(crate) fn resolve_for_graph(
        &self,
        graph: &petgraph::graph::UnGraph<Py<PyAny>, Py<PyAny>, IndexType>,
    ) -> PyResult<crate::graph::ResolvedEdgeLengths> {
        use petgraph::visit::EdgeIndexable;
        match self {
            Self::Constant(v) => {
                if !v.is_finite() || *v < 0.0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "edge length must be finite and non-negative",
                    ));
                }
                Ok(crate::graph::ResolvedEdgeLengths::Constant(*v))
            }
            Self::Callable(func) => Python::attach(|py| {
                let mut lengths = vec![0.0 as FloatType; graph.edge_bound()];
                for e in graph.edge_references() {
                    let val: FloatType = func.call1(py, (e.id().index(),))?.extract(py)?;
                    if !val.is_finite() || val < 0.0 {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "edge length must be finite and non-negative",
                        ));
                    }
                    lengths[e.id().index()] = val;
                }
                Ok(crate::graph::ResolvedEdgeLengths::PerEdge(lengths))
            }),
        }
    }
}

/// Sparse symmetric graph Laplacian matrix.
#[pyclass(from_py_object)]
#[pyo3(name = "Laplacian")]
#[derive(Clone)]
pub struct PyLaplacian {
    pub(crate) matrix: SparseSymmetricMatrix<FloatType>,
}

fn build_graph_laplacian(
    graph: &PyGraphAdapter,
    length: Option<&Bound<'_, PyAny>>,
    normalized: bool,
) -> PyResult<PyLaplacian> {
    let edge_length = EdgeLengthFn::from_bound(length)?;
    let matrix = match graph.graph() {
        GraphType::Graph(native_graph) => {
            let resolved_len = edge_length.resolve_for_graph(native_graph)?;
            let length_fn = |edge: petgraph::graph::EdgeReference<'_, Py<PyAny>>| {
                resolved_len.get(edge.id().index())
            };
            if normalized {
                SparseSymmetricMatrix::symmetric_normalized_laplacian(native_graph, length_fn)
            } else {
                SparseSymmetricMatrix::standard_laplacian(native_graph, length_fn)
            }
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Unsupported graph type",
            ));
        }
    };
    Ok(PyLaplacian { matrix })
}

#[pymethods]
impl PyLaplacian {
    /// Builds an unnormalized combinatorial Laplacian matrix (`L = D - W`).
    #[staticmethod]
    #[pyo3(signature = (graph, length = None))]
    pub fn standard(
        graph: &PyGraphAdapter,
        length: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyLaplacian> {
        build_graph_laplacian(graph, length, false)
    }

    /// Builds a symmetric normalized Laplacian matrix (`L_sym = I - D^{-1/2} W D^{-1/2}`).
    #[staticmethod]
    #[pyo3(signature = (graph, length = None))]
    pub fn symmetric_normalized(
        graph: &PyGraphAdapter,
        length: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyLaplacian> {
        build_graph_laplacian(graph, length, true)
    }

    /// Returns the number of nodes in the Laplacian matrix.
    pub fn n(&self) -> usize {
        self.matrix.n
    }
}

/// Builder namespace for constructing unnormalized combinatorial Laplacians.
#[pyclass]
#[pyo3(name = "StandardLaplacian")]
pub struct PyStandardLaplacian;

#[pymethods]
impl PyStandardLaplacian {
    /// Builds an unnormalized combinatorial Laplacian matrix (`L = D - W`).
    #[staticmethod]
    #[pyo3(signature = (graph, length = None))]
    pub fn build(
        graph: &PyGraphAdapter,
        length: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyLaplacian> {
        build_graph_laplacian(graph, length, false)
    }
}

/// Builder namespace for constructing symmetric normalized graph Laplacians.
#[pyclass]
#[pyo3(name = "SymmetricNormalizedLaplacian")]
pub struct PySymmetricNormalizedLaplacian;

#[pymethods]
impl PySymmetricNormalizedLaplacian {
    /// Builds a symmetric normalized Laplacian matrix (`L_sym = I - D^{-1/2} W D^{-1/2}`).
    #[staticmethod]
    #[pyo3(signature = (graph, length = None))]
    pub fn build(
        graph: &PyGraphAdapter,
        length: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyLaplacian> {
        build_graph_laplacian(graph, length, true)
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLaplacian>()?;
    m.add_class::<PyStandardLaplacian>()?;
    m.add_class::<PySymmetricNormalizedLaplacian>()?;
    Ok(())
}
