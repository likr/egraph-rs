use crate::{
    graph::{GraphType, IndexType, PyGraphAdapter},
    FloatType,
};
use petgraph::{graph::NodeIndex, stable_graph::node_index, visit::EdgeRef};
use petgraph_algorithm_shortest_path::{DistanceMatrix, FullDistanceMatrix, PivotedDistanceMatrix};
use petgraph_linalg_kernel::{
    BandPassDiffusionKernel, DiffusionKernel, Distance, EmbeddingKernel, EmbeddingKernelBuilder,
    ExponentialKernel, ExponentialKernelBuilder, GaussianKernel, GaussianKernelBuilder, Kernel,
    KernelDistance, KernelDistanceBuilder, LowRankDiffusionKernel,
    LowRankMultiscaleDiffusionKernel, MultiscaleDiffusionKernel, NegLogDistance,
    NegLogDistanceBuilder, NegLogSimDistance, NegLogSimDistanceBuilder,
    PivotedBandPassDiffusionKernel, PivotedDiffusionKernel, PivotedKernel,
    PivotedMultiscaleDiffusionKernel, PivotedNegLogDistance, PivotedNegLogDistanceBuilder,
    PivotedNegLogSimDistance, PivotedNegLogSimDistanceBuilder, TKernel, TKernelBuilder,
};
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
        if !bound.is_callable() {
            if let Ok(val) = bound.extract::<FloatType>() {
                if !val.is_finite() || val < 0.0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "edge length must be finite and non-negative",
                    ));
                }
                return Ok(Self::Constant(val));
            }
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

#[derive(Clone)]
pub enum InnerKernel {
    Diffusion(DiffusionKernel<FloatType>),
    BandPass(BandPassDiffusionKernel<FloatType>),
    LowRank(LowRankDiffusionKernel<FloatType>),
    LowRankMultiscale(LowRankMultiscaleDiffusionKernel<FloatType>),
    Multiscale(MultiscaleDiffusionKernel<FloatType>),
    Embedding(EmbeddingKernel<NodeIndex<IndexType>, FloatType>),
    Gaussian(GaussianKernel<Box<InnerKernel>, FloatType>),
    Exponential(ExponentialKernel<Box<InnerKernel>, FloatType>),
    T(TKernel<Box<InnerKernel>, FloatType>),
}

impl InnerKernel {
    fn get_by_index_internal(&self, i: usize, j: usize) -> FloatType {
        match self {
            Self::Diffusion(k) => Kernel::<usize, FloatType>::get_by_index(k, i, j),
            Self::BandPass(k) => Kernel::<usize, FloatType>::get_by_index(k, i, j),
            Self::LowRank(k) => Kernel::<usize, FloatType>::get_by_index(k, i, j),
            Self::LowRankMultiscale(k) => Kernel::<usize, FloatType>::get_by_index(k, i, j),
            Self::Multiscale(k) => Kernel::<usize, FloatType>::get_by_index(k, i, j),
            Self::Embedding(k) => Kernel::<NodeIndex<IndexType>, FloatType>::get_by_index(k, i, j),
            Self::Gaussian(k) => Kernel::<usize, FloatType>::get_by_index(k, i, j),
            Self::Exponential(k) => Kernel::<usize, FloatType>::get_by_index(k, i, j),
            Self::T(k) => Kernel::<usize, FloatType>::get_by_index(k, i, j),
        }
    }

    fn shape_internal(&self) -> (usize, usize) {
        match self {
            Self::Diffusion(k) => Kernel::<usize, FloatType>::shape(k),
            Self::BandPass(k) => Kernel::<usize, FloatType>::shape(k),
            Self::LowRank(k) => Kernel::<usize, FloatType>::shape(k),
            Self::LowRankMultiscale(k) => Kernel::<usize, FloatType>::shape(k),
            Self::Multiscale(k) => Kernel::<usize, FloatType>::shape(k),
            Self::Embedding(k) => Kernel::<NodeIndex<IndexType>, FloatType>::shape(k),
            Self::Gaussian(k) => Kernel::<usize, FloatType>::shape(k),
            Self::Exponential(k) => Kernel::<usize, FloatType>::shape(k),
            Self::T(k) => Kernel::<usize, FloatType>::shape(k),
        }
    }
}

impl Kernel<usize, FloatType> for InnerKernel {
    fn get(&self, u: usize, v: usize) -> Option<FloatType> {
        let (r, c) = self.shape_internal();
        if u < r && v < c {
            Some(self.get_by_index_internal(u, v))
        } else {
            None
        }
    }

    fn get_by_index(&self, i: usize, j: usize) -> FloatType {
        self.get_by_index_internal(i, j)
    }

    fn shape(&self) -> (usize, usize) {
        self.shape_internal()
    }

    fn row_index(&self, u: usize) -> Option<usize> {
        Some(u).filter(|&i| i < self.shape_internal().0)
    }

    fn col_index(&self, v: usize) -> Option<usize> {
        Some(v).filter(|&j| j < self.shape_internal().1)
    }
}

impl Kernel<NodeIndex<IndexType>, FloatType> for InnerKernel {
    fn get(&self, u: NodeIndex<IndexType>, v: NodeIndex<IndexType>) -> Option<FloatType> {
        let (r, c) = self.shape_internal();
        if u.index() < r && v.index() < c {
            Some(self.get_by_index_internal(u.index(), v.index()))
        } else {
            None
        }
    }

    fn get_by_index(&self, i: usize, j: usize) -> FloatType {
        self.get_by_index_internal(i, j)
    }

    fn shape(&self) -> (usize, usize) {
        self.shape_internal()
    }

    fn row_index(&self, u: NodeIndex<IndexType>) -> Option<usize> {
        Some(u.index()).filter(|&i| i < self.shape_internal().0)
    }

    fn col_index(&self, v: NodeIndex<IndexType>) -> Option<usize> {
        Some(v.index()).filter(|&j| j < self.shape_internal().1)
    }
}

pub fn extract_inner_kernel(kernel: &Bound<PyAny>) -> PyResult<InnerKernel> {
    if let Ok(k) = kernel.extract::<PyRef<crate::layout::sgd::PyDiffusionKernel>>() {
        Ok(InnerKernel::Diffusion(k.kernel.clone()))
    } else if let Ok(k) = kernel.extract::<PyRef<crate::layout::sgd::PyBandPassDiffusionKernel>>() {
        Ok(InnerKernel::BandPass(k.kernel.clone()))
    } else if let Ok(k) = kernel.extract::<PyRef<crate::layout::sgd::PyLowRankDiffusionKernel>>() {
        Ok(InnerKernel::LowRank(k.kernel.clone()))
    } else if let Ok(k) =
        kernel.extract::<PyRef<crate::layout::sgd::PyLowRankMultiscaleDiffusionKernel>>()
    {
        Ok(InnerKernel::LowRankMultiscale(k.kernel.clone()))
    } else if let Ok(k) = kernel.extract::<PyRef<crate::layout::sgd::PyMultiscaleDiffusionKernel>>()
    {
        Ok(InnerKernel::Multiscale(k.kernel.clone()))
    } else if let Ok(k) = kernel.extract::<PyRef<PyEmbeddingKernel>>() {
        Ok(InnerKernel::Embedding(k.kernel.clone()))
    } else if let Ok(k) = kernel.extract::<PyRef<PyGaussianKernel>>() {
        Ok(InnerKernel::Gaussian(k.kernel.clone()))
    } else if let Ok(k) = kernel.extract::<PyRef<PyExponentialKernel>>() {
        Ok(InnerKernel::Exponential(k.kernel.clone()))
    } else if let Ok(k) = kernel.extract::<PyRef<PyTKernel>>() {
        Ok(InnerKernel::T(k.kernel.clone()))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Unsupported kernel type",
        ))
    }
}

#[derive(Clone)]
pub enum InnerPivotedKernel {
    Diffusion(PivotedDiffusionKernel<FloatType>),
    BandPass(PivotedBandPassDiffusionKernel<FloatType>),
    Multiscale(PivotedMultiscaleDiffusionKernel<FloatType>),
}

impl PivotedKernel<FloatType> for InnerPivotedKernel {
    fn pivots(&self) -> &[usize] {
        match self {
            Self::Diffusion(k) => k.pivots(),
            Self::BandPass(k) => k.pivots(),
            Self::Multiscale(k) => k.pivots(),
        }
    }

    fn get_from_pivot(&self, pivot_idx: usize, j: usize) -> FloatType {
        match self {
            Self::Diffusion(k) => k.get_from_pivot(pivot_idx, j),
            Self::BandPass(k) => k.get_from_pivot(pivot_idx, j),
            Self::Multiscale(k) => k.get_from_pivot(pivot_idx, j),
        }
    }

    fn diagonal(&self, j: usize) -> Option<FloatType> {
        match self {
            Self::Diffusion(k) => k.diagonal(j),
            Self::BandPass(k) => k.diagonal(j),
            Self::Multiscale(k) => k.diagonal(j),
        }
    }

    fn node_count(&self) -> Option<usize> {
        match self {
            Self::Diffusion(k) => k.node_count(),
            Self::BandPass(k) => k.node_count(),
            Self::Multiscale(k) => k.node_count(),
        }
    }
}

pub fn extract_inner_pivoted_kernel(kernel: &Bound<PyAny>) -> PyResult<InnerPivotedKernel> {
    if let Ok(k) = kernel.extract::<PyRef<crate::layout::sgd::PyPivotedDiffusionKernel>>() {
        Ok(InnerPivotedKernel::Diffusion(k.kernel.clone()))
    } else if let Ok(k) =
        kernel.extract::<PyRef<crate::layout::sgd::PyPivotedBandPassDiffusionKernel>>()
    {
        Ok(InnerPivotedKernel::BandPass(k.kernel.clone()))
    } else if let Ok(k) =
        kernel.extract::<PyRef<crate::layout::sgd::PyPivotedMultiscaleDiffusionKernel>>()
    {
        Ok(InnerPivotedKernel::Multiscale(k.kernel.clone()))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Unsupported pivoted kernel type",
        ))
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
#[pyo3(name = "NegLogSimDistance")]
pub struct PyNegLogSimDistance {
    pub(crate) matrix: NegLogSimDistance<NodeIndex<IndexType>, FloatType, InnerKernel>,
}

#[pymethods]
impl PyNegLogSimDistance {
    pub fn get(&self, u: usize, v: usize) -> Option<FloatType> {
        self.matrix
            .get(node_index::<IndexType>(u), node_index::<IndexType>(v))
    }
}

#[pyclass]
#[pyo3(name = "NegLogSimDistanceBuilder")]
pub struct PyNegLogSimDistanceBuilder {
    pub(crate) builder: NegLogSimDistanceBuilder<FloatType>,
}

#[pymethods]
impl PyNegLogSimDistanceBuilder {
    #[new]
    fn new() -> Self {
        Self {
            builder: NegLogSimDistanceBuilder::new(),
        }
    }

    fn alpha(mut slf: PyRefMut<'_, Self>, alpha: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().alpha(alpha);
        slf
    }

    fn beta(mut slf: PyRefMut<'_, Self>, beta: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().beta(beta);
        slf
    }

    fn p(mut slf: PyRefMut<'_, Self>, p: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().p(p);
        slf
    }

    fn min_dist(mut slf: PyRefMut<'_, Self>, min_dist: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().min_dist(min_dist);
        slf
    }

    fn build(
        &self,
        graph: &PyGraphAdapter,
        kernel: &Bound<PyAny>,
    ) -> PyResult<PyNegLogSimDistance> {
        let inner_kernel = extract_inner_kernel(kernel)?;
        let matrix = match graph.graph() {
            GraphType::Graph(native_graph) => self
                .builder
                .clone()
                .build(native_graph, inner_kernel)
                .map_err(pyo3::exceptions::PyValueError::new_err)?,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Unsupported graph type",
                ))
            }
        };
        Ok(PyNegLogSimDistance { matrix })
    }
}

#[pyclass]
#[pyo3(name = "NegLogDistance")]
pub struct PyNegLogDistance {
    pub(crate) matrix: NegLogDistance<NodeIndex<IndexType>, FloatType, InnerKernel>,
}

#[pymethods]
impl PyNegLogDistance {
    pub fn get(&self, u: usize, v: usize) -> Option<FloatType> {
        self.matrix
            .get(node_index::<IndexType>(u), node_index::<IndexType>(v))
    }
}

#[pyclass]
#[pyo3(name = "NegLogDistanceBuilder")]
pub struct PyNegLogDistanceBuilder {
    pub(crate) builder: NegLogDistanceBuilder<FloatType>,
}

#[pymethods]
impl PyNegLogDistanceBuilder {
    #[new]
    fn new() -> Self {
        Self {
            builder: NegLogDistanceBuilder::new(),
        }
    }

    fn alpha(mut slf: PyRefMut<'_, Self>, alpha: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().alpha(alpha);
        slf
    }

    fn beta(mut slf: PyRefMut<'_, Self>, beta: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().beta(beta);
        slf
    }

    fn p(mut slf: PyRefMut<'_, Self>, p: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().p(p);
        slf
    }

    fn min_dist(mut slf: PyRefMut<'_, Self>, min_dist: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().min_dist(min_dist);
        slf
    }

    fn build(&self, graph: &PyGraphAdapter, kernel: &Bound<PyAny>) -> PyResult<PyNegLogDistance> {
        let inner_kernel = extract_inner_kernel(kernel)?;
        let matrix = match graph.graph() {
            GraphType::Graph(native_graph) => self
                .builder
                .clone()
                .build(native_graph, inner_kernel)
                .map_err(pyo3::exceptions::PyValueError::new_err)?,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Unsupported graph type",
                ))
            }
        };
        Ok(PyNegLogDistance { matrix })
    }
}

#[pyclass]
#[pyo3(name = "PivotedNegLogDistance")]
pub struct PyPivotedNegLogDistance {
    pub(crate) matrix: PivotedNegLogDistance<NodeIndex<IndexType>, FloatType, InnerPivotedKernel>,
}

#[pymethods]
impl PyPivotedNegLogDistance {
    pub fn get(&self, u: usize, v: usize) -> Option<FloatType> {
        self.matrix
            .get(node_index::<IndexType>(u), node_index::<IndexType>(v))
    }
}

#[pyclass]
#[pyo3(name = "PivotedNegLogDistanceBuilder")]
pub struct PyPivotedNegLogDistanceBuilder {
    pub(crate) builder: PivotedNegLogDistanceBuilder<FloatType>,
}

#[pymethods]
impl PyPivotedNegLogDistanceBuilder {
    #[new]
    fn new() -> Self {
        Self {
            builder: PivotedNegLogDistanceBuilder::new(),
        }
    }

    fn alpha(mut slf: PyRefMut<'_, Self>, alpha: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().alpha(alpha);
        slf
    }

    fn beta(mut slf: PyRefMut<'_, Self>, beta: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().beta(beta);
        slf
    }

    fn p(mut slf: PyRefMut<'_, Self>, p: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().p(p);
        slf
    }

    fn min_dist(mut slf: PyRefMut<'_, Self>, min_dist: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().min_dist(min_dist);
        slf
    }

    fn build(
        &self,
        graph: &PyGraphAdapter,
        kernel: &Bound<PyAny>,
    ) -> PyResult<PyPivotedNegLogDistance> {
        let inner_kernel = extract_inner_pivoted_kernel(kernel)?;
        let matrix = match graph.graph() {
            GraphType::Graph(native_graph) => self
                .builder
                .clone()
                .build(native_graph, inner_kernel)
                .map_err(pyo3::exceptions::PyValueError::new_err)?,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Unsupported graph type",
                ))
            }
        };
        Ok(PyPivotedNegLogDistance { matrix })
    }
}

fn build_pivoted_neg_log_sim_distance(
    builder: PivotedNegLogSimDistanceBuilder<FloatType>,
    graph: &PyGraphAdapter,
    kernel: &Bound<PyAny>,
) -> PyResult<PyPivotedNegLogSimDistance> {
    let inner_kernel = extract_inner_pivoted_kernel(kernel)?;
    let matrix = match graph.graph() {
        GraphType::Graph(native_graph) => builder
            .build(native_graph, inner_kernel)
            .map_err(pyo3::exceptions::PyValueError::new_err)?,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Unsupported graph type",
            ))
        }
    };
    Ok(PyPivotedNegLogSimDistance { matrix })
}

/// Pivoted distance matrix of shape `(N, h)` computed from normalized negative log similarity.
#[pyclass]
#[pyo3(name = "PivotedNegLogSimDistance")]
pub struct PyPivotedNegLogSimDistance {
    pub(crate) matrix:
        PivotedNegLogSimDistance<NodeIndex<IndexType>, FloatType, InnerPivotedKernel>,
}

#[pymethods]
impl PyPivotedNegLogSimDistance {
    /// Creates a new `PivotedNegLogSimDistance` from a graph and pivoted kernel.
    #[new]
    #[pyo3(signature = (graph, kernel, epsilon = 1e-12, scale = 1.0, power = 0.5))]
    pub fn new(
        graph: &PyGraphAdapter,
        kernel: &Bound<PyAny>,
        epsilon: FloatType,
        scale: FloatType,
        power: FloatType,
    ) -> PyResult<Self> {
        let builder = PivotedNegLogSimDistanceBuilder::new()
            .epsilon(epsilon)
            .scale(scale)
            .power(power);
        build_pivoted_neg_log_sim_distance(builder, graph, kernel)
    }

    /// Returns the distance between node `u` and node `v` when one of them is a pivot, else `None`.
    pub fn get(&self, u: usize, v: usize) -> Option<FloatType> {
        self.matrix
            .get(node_index::<IndexType>(u), node_index::<IndexType>(v))
    }

    /// Returns the distance for row node index `j` and pivot column index `pivot_idx`.
    pub fn get_by_index(&self, j: usize, pivot_idx: usize) -> PyResult<FloatType> {
        let (n, h) = self.matrix.shape();
        if j >= n || pivot_idx >= h {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Index ({j}, {pivot_idx}) out of bounds for shape ({n}, {h})"
            )));
        }
        Ok(self.matrix.get_by_index(j, pivot_idx))
    }

    /// Returns the `(N, h)` matrix shape.
    pub fn shape(&self) -> (usize, usize) {
        self.matrix.shape()
    }

    /// Returns the list of pivot node indices.
    pub fn pivots(&self) -> Vec<usize> {
        self.matrix.pivots().to_vec()
    }
}

/// Builder for [`PivotedNegLogSimDistance`].
#[pyclass]
#[pyo3(name = "PivotedNegLogSimDistanceBuilder")]
pub struct PyPivotedNegLogSimDistanceBuilder {
    pub(crate) builder: PivotedNegLogSimDistanceBuilder<FloatType>,
}

#[pymethods]
impl PyPivotedNegLogSimDistanceBuilder {
    /// Creates a new `PivotedNegLogSimDistanceBuilder` with default parameters.
    #[new]
    fn new() -> Self {
        Self {
            builder: PivotedNegLogSimDistanceBuilder::new(),
        }
    }

    /// Sets the positivity clamping floor `epsilon`.
    fn epsilon(mut slf: PyRefMut<'_, Self>, epsilon: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().epsilon(epsilon);
        slf
    }

    /// Sets the distance multiplier `scale`.
    fn scale(mut slf: PyRefMut<'_, Self>, scale: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().scale(scale);
        slf
    }

    /// Sets the power exponent `power`.
    fn power(mut slf: PyRefMut<'_, Self>, power: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().power(power);
        slf
    }

    /// Builds a `PivotedNegLogSimDistance` for `graph` and `kernel`.
    fn build(
        &self,
        graph: &PyGraphAdapter,
        kernel: &Bound<PyAny>,
    ) -> PyResult<PyPivotedNegLogSimDistance> {
        build_pivoted_neg_log_sim_distance(self.builder.clone(), graph, kernel)
    }
}

#[pyclass]
#[pyo3(name = "EmbeddingKernel")]
pub struct PyEmbeddingKernel {
    pub(crate) kernel: EmbeddingKernel<NodeIndex<IndexType>, FloatType>,
}

#[pymethods]
impl PyEmbeddingKernel {
    #[new]
    pub fn new(graph: &PyGraphAdapter, embedding: &crate::array::PyArray2) -> PyResult<Self> {
        let kernel = match graph.graph() {
            GraphType::Graph(native_graph) => {
                EmbeddingKernel::try_new(native_graph, embedding.as_array().clone())
                    .map_err(pyo3::exceptions::PyValueError::new_err)?
            }
            GraphType::DiGraph(native_graph) => {
                EmbeddingKernel::try_new(native_graph, embedding.as_array().clone())
                    .map_err(pyo3::exceptions::PyValueError::new_err)?
            }
        };
        Ok(Self { kernel })
    }

    pub fn get(&self, u: usize, v: usize) -> Option<FloatType> {
        self.kernel.get(NodeIndex::new(u), NodeIndex::new(v))
    }
}

#[pyclass]
#[pyo3(name = "EmbeddingKernelBuilder")]
pub struct PyEmbeddingKernelBuilder {
    pub(crate) builder: EmbeddingKernelBuilder,
}

#[pymethods]
impl PyEmbeddingKernelBuilder {
    #[new]
    pub fn new() -> Self {
        Self {
            builder: EmbeddingKernelBuilder::new(),
        }
    }

    pub fn build(
        &self,
        graph: &PyGraphAdapter,
        embedding: &crate::array::PyArray2,
    ) -> PyResult<PyEmbeddingKernel> {
        let kernel = match graph.graph() {
            GraphType::Graph(native_graph) => self
                .builder
                .clone()
                .build(native_graph, embedding.as_array().clone())
                .map_err(pyo3::exceptions::PyValueError::new_err)?,
            GraphType::DiGraph(native_graph) => self
                .builder
                .clone()
                .build(native_graph, embedding.as_array().clone())
                .map_err(pyo3::exceptions::PyValueError::new_err)?,
        };
        Ok(PyEmbeddingKernel { kernel })
    }
}

#[pyclass]
#[pyo3(name = "GaussianKernel")]
pub struct PyGaussianKernel {
    pub(crate) kernel: GaussianKernel<Box<InnerKernel>, FloatType>,
}

#[pymethods]
impl PyGaussianKernel {
    #[new]
    pub fn new(kernel: &Bound<PyAny>, gamma: FloatType) -> PyResult<Self> {
        let inner = extract_inner_kernel(kernel)?;
        Ok(Self {
            kernel: GaussianKernel::new(Box::new(inner), gamma),
        })
    }

    pub fn get(&self, u: usize, v: usize) -> Option<FloatType> {
        self.kernel.get(u, v)
    }
}

#[pyclass]
#[pyo3(name = "GaussianKernelBuilder")]
pub struct PyGaussianKernelBuilder {
    pub(crate) builder: GaussianKernelBuilder<FloatType>,
}

#[pymethods]
impl PyGaussianKernelBuilder {
    #[new]
    #[pyo3(signature = (gamma = 1.0))]
    pub fn new(gamma: FloatType) -> Self {
        Self {
            builder: GaussianKernelBuilder::new(gamma),
        }
    }

    pub fn gamma(mut slf: PyRefMut<'_, Self>, gamma: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().gamma(gamma);
        slf
    }

    pub fn build(&self, kernel: &Bound<PyAny>) -> PyResult<PyGaussianKernel> {
        let inner = extract_inner_kernel(kernel)?;
        let kernel = self
            .builder
            .clone()
            .build(Box::new(inner))
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyGaussianKernel { kernel })
    }
}

#[pyclass]
#[pyo3(name = "ExponentialKernel")]
pub struct PyExponentialKernel {
    pub(crate) kernel: ExponentialKernel<Box<InnerKernel>, FloatType>,
}

#[pymethods]
impl PyExponentialKernel {
    #[new]
    pub fn new(kernel: &Bound<PyAny>, gamma: FloatType) -> PyResult<Self> {
        let inner = extract_inner_kernel(kernel)?;
        Ok(Self {
            kernel: ExponentialKernel::new(Box::new(inner), gamma),
        })
    }

    pub fn get(&self, u: usize, v: usize) -> Option<FloatType> {
        self.kernel.get(u, v)
    }
}

#[pyclass]
#[pyo3(name = "ExponentialKernelBuilder")]
pub struct PyExponentialKernelBuilder {
    pub(crate) builder: ExponentialKernelBuilder<FloatType>,
}

#[pymethods]
impl PyExponentialKernelBuilder {
    #[new]
    #[pyo3(signature = (gamma = 1.0))]
    pub fn new(gamma: FloatType) -> Self {
        Self {
            builder: ExponentialKernelBuilder::new(gamma),
        }
    }

    pub fn gamma(mut slf: PyRefMut<'_, Self>, gamma: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().gamma(gamma);
        slf
    }

    pub fn build(&self, kernel: &Bound<PyAny>) -> PyResult<PyExponentialKernel> {
        let inner = extract_inner_kernel(kernel)?;
        let kernel = self
            .builder
            .clone()
            .build(Box::new(inner))
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyExponentialKernel { kernel })
    }
}

#[pyclass]
#[pyo3(name = "TKernel")]
pub struct PyTKernel {
    pub(crate) kernel: TKernel<Box<InnerKernel>, FloatType>,
}

#[pymethods]
impl PyTKernel {
    #[new]
    pub fn new(kernel: &Bound<PyAny>, dof: FloatType) -> PyResult<Self> {
        let inner = extract_inner_kernel(kernel)?;
        Ok(Self {
            kernel: TKernel::new(Box::new(inner), dof),
        })
    }

    pub fn get(&self, u: usize, v: usize) -> Option<FloatType> {
        self.kernel.get(u, v)
    }
}

#[pyclass]
#[pyo3(name = "TKernelBuilder")]
pub struct PyTKernelBuilder {
    pub(crate) builder: TKernelBuilder<FloatType>,
}

#[pymethods]
impl PyTKernelBuilder {
    #[new]
    #[pyo3(signature = (alpha = 1.0))]
    pub fn new(alpha: FloatType) -> Self {
        Self {
            builder: TKernelBuilder::new(alpha),
        }
    }

    pub fn alpha(mut slf: PyRefMut<'_, Self>, alpha: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().alpha(alpha);
        slf
    }

    pub fn build(&self, kernel: &Bound<PyAny>) -> PyResult<PyTKernel> {
        let inner = extract_inner_kernel(kernel)?;
        let kernel = self
            .builder
            .clone()
            .build(Box::new(inner))
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyTKernel { kernel })
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
        let mut matrix = KernelDistance::new(inner);
        matrix.min_dist = min_dist;
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

use petgraph_linalg_kernel::SparseSymmetricMatrix;

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
            ))
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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDistanceMatrix>()?;
    m.add_class::<PyNegLogSimDistance>()?;
    m.add_class::<PyNegLogSimDistanceBuilder>()?;
    m.add_class::<PyNegLogDistance>()?;
    m.add_class::<PyNegLogDistanceBuilder>()?;
    m.add_class::<PyPivotedNegLogDistance>()?;
    m.add_class::<PyPivotedNegLogDistanceBuilder>()?;
    m.add_class::<PyPivotedNegLogSimDistance>()?;
    m.add_class::<PyPivotedNegLogSimDistanceBuilder>()?;
    m.add_class::<PyEmbeddingKernel>()?;
    m.add_class::<PyEmbeddingKernelBuilder>()?;
    m.add_class::<PyGaussianKernel>()?;
    m.add_class::<PyGaussianKernelBuilder>()?;
    m.add_class::<PyExponentialKernel>()?;
    m.add_class::<PyExponentialKernelBuilder>()?;
    m.add_class::<PyTKernel>()?;
    m.add_class::<PyTKernelBuilder>()?;
    m.add_class::<PyKernelDistance>()?;
    m.add_class::<PyKernelDistanceBuilder>()?;
    m.add_class::<PyLaplacian>()?;
    m.add_class::<PyStandardLaplacian>()?;
    m.add_class::<PySymmetricNormalizedLaplacian>()?;
    Ok(())
}
