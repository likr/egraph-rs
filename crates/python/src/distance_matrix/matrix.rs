//! Core distance matrix wrappers, kernel distance, and dynamic dispatch helpers.

use super::kernels::{InnerKernel, extract_inner_kernel};
use super::neg_log::{
    PyNegLogDistance, PyNegLogSimDistance, PyPivotedNegLogDistance, PyPivotedNegLogSimDistance,
};
use crate::{
    FloatType,
    graph::{GraphType, IndexType, PyGraphAdapter},
};
use petgraph::{graph::NodeIndex, stable_graph::node_index};
use petgraph_algorithm_shortest_path::{DistanceMatrix, FullDistanceMatrix, PivotedDistanceMatrix};
use petgraph_linalg_kernel::{Distance, KernelDistance, KernelDistanceBuilder};
use pyo3::prelude::*;

pub enum DistanceMatrixType {
    Full(FullDistanceMatrix<NodeIndex<IndexType>, FloatType>),
    Pivoted(PivotedDistanceMatrix<NodeIndex<IndexType>, FloatType>),
}

#[pyclass]
#[pyo3(name = "DistanceMatrix")]
pub struct PyDistanceMatrix {
    distance_matrix: DistanceMatrixType,
}

impl PyDistanceMatrix {
    pub fn new_with_full_distance_matrix(
        distance_matrix: FullDistanceMatrix<NodeIndex<IndexType>, FloatType>,
    ) -> Self {
        PyDistanceMatrix {
            distance_matrix: DistanceMatrixType::Full(distance_matrix),
        }
    }
    pub fn new_with_pivoted_distance_matrix(
        distance_matrix: PivotedDistanceMatrix<NodeIndex<IndexType>, FloatType>,
    ) -> Self {
        PyDistanceMatrix {
            distance_matrix: DistanceMatrixType::Pivoted(distance_matrix),
        }
    }
    pub fn distance_matrix(&self) -> &DistanceMatrixType {
        &self.distance_matrix
    }

    pub fn distance_matrix_mut(&mut self) -> &mut DistanceMatrixType {
        &mut self.distance_matrix
    }
}

#[pymethods]
impl PyDistanceMatrix {
    pub fn shape(&self) -> (usize, usize) {
        match &self.distance_matrix {
            DistanceMatrixType::Full(dm) => DistanceMatrix::shape(dm),
            DistanceMatrixType::Pivoted(dm) => DistanceMatrix::shape(dm),
        }
    }

    pub fn pivots(&self) -> Option<Vec<usize>> {
        match &self.distance_matrix {
            DistanceMatrixType::Pivoted(dm) => {
                Some(dm.pivots().iter().map(|u| u.index()).collect())
            }
            DistanceMatrixType::Full(_) => None,
        }
    }

    pub fn get(&self, u: usize, v: usize) -> Option<FloatType> {
        match &self.distance_matrix {
            DistanceMatrixType::Full(dm) => {
                DistanceMatrix::get(dm, node_index::<IndexType>(u), node_index::<IndexType>(v))
            }
            DistanceMatrixType::Pivoted(dm) => {
                DistanceMatrix::get(dm, node_index::<IndexType>(u), node_index::<IndexType>(v))
            }
        }
    }

    pub fn get_by_index(&self, i: usize, j: usize) -> PyResult<FloatType> {
        let (rows, cols) = self.shape();
        if i >= rows || j >= cols {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Index ({i}, {j}) out of bounds for distance matrix shape ({rows}, {cols})"
            )));
        }
        Ok(match &self.distance_matrix {
            DistanceMatrixType::Full(dm) => DistanceMatrix::get_by_index(dm, i, j),
            DistanceMatrixType::Pivoted(dm) => DistanceMatrix::get_by_index(dm, i, j),
        })
    }

    pub fn set(&mut self, u: usize, v: usize, d: FloatType) -> Option<()> {
        match self.distance_matrix_mut() {
            DistanceMatrixType::Full(dm) => {
                dm.set(node_index::<IndexType>(u), node_index::<IndexType>(v), d)
            }
            DistanceMatrixType::Pivoted(dm) => {
                dm.set(node_index::<IndexType>(u), node_index::<IndexType>(v), d)
            }
        }
    }

    /// Create a full distance matrix from an `(N, N)` float64 NumPy array.
    #[staticmethod]
    pub fn from_numpy(
        graph: &PyGraphAdapter,
        array: numpy::PyReadonlyArray2<'_, FloatType>,
    ) -> PyResult<Self> {
        let view = array.as_array();
        let n = match graph.graph() {
            GraphType::Graph(g) => g.node_count(),
            GraphType::DiGraph(g) => g.node_count(),
        };
        if view.dim() != (n, n) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "array shape {:?} does not match expected ({}, {})",
                view.dim(),
                n,
                n
            )));
        }
        let owned = view.to_owned();
        let distance_matrix = match graph.graph() {
            GraphType::Graph(g) => DistanceMatrixType::Full(
                petgraph_algorithm_shortest_path::FullDistanceMatrix::from_array(g, owned),
            ),
            GraphType::DiGraph(g) => DistanceMatrixType::Full(
                petgraph_algorithm_shortest_path::FullDistanceMatrix::from_array(g, owned),
            ),
        };
        Ok(Self { distance_matrix })
    }

    /// Create a pivoted distance matrix from pivot node indices and a `(P, N)` float64 NumPy array.
    #[staticmethod]
    pub fn from_pivoted_numpy(
        graph: &PyGraphAdapter,
        pivots: Vec<usize>,
        array: numpy::PyReadonlyArray2<'_, FloatType>,
    ) -> PyResult<Self> {
        let view = array.as_array();
        let n = match graph.graph() {
            GraphType::Graph(g) => g.node_count(),
            GraphType::DiGraph(g) => g.node_count(),
        };
        if view.dim() != (pivots.len(), n) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "array shape {:?} does not match expected ({}, {})",
                view.dim(),
                pivots.len(),
                n
            )));
        }
        let mut seen = std::collections::HashSet::with_capacity(pivots.len());
        let mut pivot_ids = Vec::with_capacity(pivots.len());
        for &p in &pivots {
            if p > IndexType::MAX as usize {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "pivot index {} exceeds IndexType capacity",
                    p
                )));
            }
            let pid = node_index::<IndexType>(p);
            let exists = match graph.graph() {
                GraphType::Graph(g) => g.node_weight(pid).is_some(),
                GraphType::DiGraph(g) => g.node_weight(pid).is_some(),
            };
            if !exists {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "pivot index {} does not exist in graph",
                    p
                )));
            }
            if !seen.insert(p) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "duplicate pivot index {}",
                    p
                )));
            }
            pivot_ids.push(pid);
        }
        let owned = view.to_owned();
        let distance_matrix = match graph.graph() {
            GraphType::Graph(g) => DistanceMatrixType::Pivoted(
                petgraph_algorithm_shortest_path::PivotedDistanceMatrix::from_array(
                    g, pivot_ids, owned,
                ),
            ),
            GraphType::DiGraph(g) => DistanceMatrixType::Pivoted(
                petgraph_algorithm_shortest_path::PivotedDistanceMatrix::from_array(
                    g, pivot_ids, owned,
                ),
            ),
        };
        Ok(Self { distance_matrix })
    }

    /// Export the distance matrix as a 2D float64 NumPy array matching `shape()`.
    pub fn to_numpy<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<FloatType>>> {
        use numpy::ToPyArray;
        match &self.distance_matrix {
            DistanceMatrixType::Full(dm) => Ok(dm.as_array().to_pyarray(py)),
            DistanceMatrixType::Pivoted(dm) => Ok(dm.as_array().to_pyarray(py)),
        }
    }
}

pub fn distance_node_count(distance_matrix: &Bound<PyAny>) -> PyResult<usize> {
    if let Ok(dm) = distance_matrix.extract::<PyRef<PyDistanceMatrix>>() {
        match dm.distance_matrix() {
            DistanceMatrixType::Full(d) => Ok(Distance::shape(d).0),
            DistanceMatrixType::Pivoted(d) => Ok(Distance::shape(d).1),
        }
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyNegLogSimDistance>>() {
        Ok(dm.matrix.shape().0)
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyNegLogDistance>>() {
        Ok(dm.matrix.shape().0)
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyPivotedNegLogDistance>>() {
        Ok(dm.matrix.shape().0)
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyPivotedNegLogSimDistance>>() {
        Ok(dm.matrix.shape().0)
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyKernelDistance>>() {
        Ok(Distance::<NodeIndex<IndexType>, FloatType>::shape(&dm.matrix).0)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Unsupported distance matrix type",
        ))
    }
}

pub fn with_distance<R>(
    distance_matrix: &Bound<PyAny>,
    f: impl FnOnce(&dyn Distance<NodeIndex<IndexType>, FloatType>) -> R,
) -> PyResult<R> {
    if let Ok(dm) = distance_matrix.extract::<PyRef<PyDistanceMatrix>>() {
        match dm.distance_matrix() {
            DistanceMatrixType::Full(d) => Ok(f(d)),
            DistanceMatrixType::Pivoted(d) => Ok(f(d)),
        }
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyNegLogSimDistance>>() {
        Ok(f(&dm.matrix))
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyNegLogDistance>>() {
        Ok(f(&dm.matrix))
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyPivotedNegLogDistance>>() {
        Ok(f(&dm.matrix))
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyPivotedNegLogSimDistance>>() {
        Ok(f(&dm.matrix))
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyKernelDistance>>() {
        Ok(f(&dm.matrix))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Unsupported distance matrix type",
        ))
    }
}

pub fn with_full_distance<R>(
    distance_matrix: &Bound<PyAny>,
    f: impl FnOnce(&dyn Distance<NodeIndex<IndexType>, FloatType>) -> R,
) -> PyResult<R> {
    if let Ok(dm) = distance_matrix.extract::<PyRef<PyDistanceMatrix>>() {
        match dm.distance_matrix() {
            DistanceMatrixType::Full(d) => {
                let (rows, cols) = Distance::shape(d);
                if rows != cols {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Expected square (N, N) distance matrix, got shape ({rows}, {cols})"
                    )));
                }
                Ok(f(d))
            }
            DistanceMatrixType::Pivoted(_) => Err(pyo3::exceptions::PyTypeError::new_err(
                "Expected a full (N, N) distance matrix, not a pivoted DistanceMatrix",
            )),
        }
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyNegLogSimDistance>>() {
        Ok(f(&dm.matrix))
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyNegLogDistance>>() {
        Ok(f(&dm.matrix))
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyKernelDistance>>() {
        Ok(f(&dm.matrix))
    } else if distance_matrix
        .extract::<PyRef<PyPivotedNegLogDistance>>()
        .is_ok()
        || distance_matrix
            .extract::<PyRef<PyPivotedNegLogSimDistance>>()
            .is_ok()
    {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Expected a full (N, N) distance matrix, not a pivoted distance matrix",
        ))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Unsupported distance matrix type",
        ))
    }
}

pub fn with_pivot_mds_distance<R>(
    distance_matrix: &Bound<PyAny>,
    f: impl FnOnce(&dyn Distance<NodeIndex<IndexType>, FloatType>) -> R,
) -> PyResult<R> {
    if let Ok(dm) = distance_matrix.extract::<PyRef<PyDistanceMatrix>>() {
        match dm.distance_matrix() {
            DistanceMatrixType::Full(d) => Ok(f(d)),
            DistanceMatrixType::Pivoted(d) => Ok(f(d)),
        }
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyNegLogSimDistance>>() {
        Ok(f(&dm.matrix))
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyNegLogDistance>>() {
        Ok(f(&dm.matrix))
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyKernelDistance>>() {
        Ok(f(&dm.matrix))
    } else if distance_matrix
        .extract::<PyRef<PyPivotedNegLogDistance>>()
        .is_ok()
        || distance_matrix
            .extract::<PyRef<PyPivotedNegLogSimDistance>>()
            .is_ok()
    {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "PivotMds requires a full (N, N) or (h, N) PivotedDistanceMatrix, not PivotedNegLogSimDistance/PivotedNegLogDistance",
        ))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Unsupported distance matrix type",
        ))
    }
}

#[pyclass]
#[pyo3(name = "KernelDistance")]
pub struct PyKernelDistance {
    pub(crate) matrix: KernelDistance<InnerKernel, FloatType>,
}

#[pymethods]
impl PyKernelDistance {
    #[new]
    #[pyo3(signature = (kernel, min_dist = 0.0))]
    pub fn new(kernel: &Bound<PyAny>, min_dist: FloatType) -> PyResult<Self> {
        let inner = extract_inner_kernel(kernel)?;
        let matrix = KernelDistanceBuilder::new()
            .min_dist(min_dist)
            .build(inner)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { matrix })
    }
}

#[pyclass]
#[pyo3(name = "KernelDistanceBuilder")]
pub struct PyKernelDistanceBuilder {
    pub(crate) builder: KernelDistanceBuilder<FloatType>,
}

#[pymethods]
impl PyKernelDistanceBuilder {
    #[new]
    #[pyo3(signature = (min_dist = 0.0))]
    pub fn new(min_dist: FloatType) -> Self {
        Self {
            builder: KernelDistanceBuilder::new().min_dist(min_dist),
        }
    }

    pub fn min_dist(mut slf: PyRefMut<'_, Self>, min_dist: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().min_dist(min_dist);
        slf
    }

    pub fn build(&self, kernel: &Bound<PyAny>) -> PyResult<PyKernelDistance> {
        let inner = extract_inner_kernel(kernel)?;
        let matrix = self
            .builder
            .clone()
            .build(inner)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyKernelDistance { matrix })
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDistanceMatrix>()?;
    m.add_class::<PyKernelDistance>()?;
    m.add_class::<PyKernelDistanceBuilder>()?;
    Ok(())
}
