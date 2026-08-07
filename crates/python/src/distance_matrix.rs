use crate::{
    graph::{GraphType, IndexType, PyGraphAdapter},
    FloatType,
};
use petgraph::{graph::NodeIndex, stable_graph::node_index, visit::EdgeRef};
use petgraph_algorithm_shortest_path::{DistanceMatrix, FullDistanceMatrix, SubDistanceMatrix};
use petgraph_distance::{Distance, GaussianKernel, Kernel, KernelDistance};
use petgraph_linalg_diffusion_kernel::{
    DiffusionKernel, LowRankDiffusionKernel, LowRankMultiscaleDiffusionKernel,
    MultiscaleDiffusionKernel, NegLogDistance, NegLogDistanceBuilder, NegLogSimDistance,
    NegLogSimDistanceBuilder, PivotedDiffusionKernel, PivotedKernel,
    PivotedMultiscaleDiffusionKernel, PivotedNegLogDistance, PivotedNegLogDistanceBuilder,
};
use petgraph_linalg_embedding_distance::EmbeddingDistanceMatrix;
use pyo3::prelude::*;

pub enum DistanceMatrixType {
    Full(FullDistanceMatrix<NodeIndex<IndexType>, FloatType>),
    Sub(SubDistanceMatrix<NodeIndex<IndexType>, FloatType>),
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
    pub fn new_with_sub_distance_matrix(
        distance_matrix: SubDistanceMatrix<NodeIndex<IndexType>, FloatType>,
    ) -> Self {
        PyDistanceMatrix {
            distance_matrix: DistanceMatrixType::Sub(distance_matrix),
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
            DistanceMatrixType::Sub(dm) => DistanceMatrix::shape(dm),
        }
    }

    pub fn get(&self, u: usize, v: usize) -> Option<FloatType> {
        match &self.distance_matrix {
            DistanceMatrixType::Full(dm) => {
                DistanceMatrix::get(dm, node_index::<IndexType>(u), node_index::<IndexType>(v))
            }
            DistanceMatrixType::Sub(dm) => {
                DistanceMatrix::get(dm, node_index::<IndexType>(u), node_index::<IndexType>(v))
            }
        }
    }

    pub fn get_by_index(&self, i: usize, j: usize) -> FloatType {
        match &self.distance_matrix {
            DistanceMatrixType::Full(dm) => DistanceMatrix::get_by_index(dm, i, j),
            DistanceMatrixType::Sub(dm) => DistanceMatrix::get_by_index(dm, i, j),
        }
    }

    pub fn set(&mut self, u: usize, v: usize, d: FloatType) -> Option<()> {
        match self.distance_matrix_mut() {
            DistanceMatrixType::Full(dm) => {
                dm.set(node_index::<IndexType>(u), node_index::<IndexType>(v), d)
            }
            DistanceMatrixType::Sub(dm) => {
                dm.set(node_index::<IndexType>(u), node_index::<IndexType>(v), d)
            }
        }
    }
}

#[derive(Clone)]
pub enum InnerKernel {
    Diffusion(DiffusionKernel<FloatType>),
    LowRank(LowRankDiffusionKernel<FloatType>),
    LowRankMultiscale(LowRankMultiscaleDiffusionKernel<FloatType>),
    Multiscale(MultiscaleDiffusionKernel<FloatType>),
}

impl Kernel<FloatType> for InnerKernel {
    fn get(&self, i: usize, j: usize) -> FloatType {
        match self {
            Self::Diffusion(k) => k.get(i, j),
            Self::LowRank(k) => k.get(i, j),
            Self::LowRankMultiscale(k) => k.get(i, j),
            Self::Multiscale(k) => k.get(i, j),
        }
    }
    fn n(&self) -> usize {
        match self {
            Self::Diffusion(k) => k.n(),
            Self::LowRank(k) => k.n(),
            Self::LowRankMultiscale(k) => k.n(),
            Self::Multiscale(k) => k.n(),
        }
    }
}

pub fn extract_inner_kernel(kernel: &Bound<PyAny>) -> PyResult<InnerKernel> {
    if let Ok(k) = kernel.extract::<PyRef<crate::layout::sgd::PyDiffusionKernel>>() {
        Ok(InnerKernel::Diffusion(k.kernel.clone()))
    } else if let Ok(k) = kernel.extract::<PyRef<crate::layout::sgd::PyLowRankDiffusionKernel>>() {
        Ok(InnerKernel::LowRank(k.kernel.clone()))
    } else if let Ok(k) =
        kernel.extract::<PyRef<crate::layout::sgd::PyLowRankMultiscaleDiffusionKernel>>()
    {
        Ok(InnerKernel::LowRankMultiscale(k.kernel.clone()))
    } else if let Ok(k) = kernel.extract::<PyRef<crate::layout::sgd::PyMultiscaleDiffusionKernel>>()
    {
        Ok(InnerKernel::Multiscale(k.kernel.clone()))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Unsupported kernel type",
        ))
    }
}

#[derive(Clone)]
pub enum InnerPivotedKernel {
    Diffusion(PivotedDiffusionKernel<FloatType>),
    Multiscale(PivotedMultiscaleDiffusionKernel<FloatType>),
}

impl PivotedKernel<FloatType> for InnerPivotedKernel {
    fn pivots(&self) -> &[usize] {
        match self {
            Self::Diffusion(k) => k.pivots(),
            Self::Multiscale(k) => k.pivots(),
        }
    }

    fn get_from_pivot(&self, pivot_idx: usize, j: usize) -> FloatType {
        match self {
            Self::Diffusion(k) => k.get_from_pivot(pivot_idx, j),
            Self::Multiscale(k) => k.get_from_pivot(pivot_idx, j),
        }
    }
}

pub fn extract_inner_pivoted_kernel(kernel: &Bound<PyAny>) -> PyResult<InnerPivotedKernel> {
    if let Ok(k) = kernel.extract::<PyRef<crate::layout::sgd::PyPivotedDiffusionKernel>>() {
        Ok(InnerPivotedKernel::Diffusion(k.kernel.clone()))
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

#[derive(Clone)]
pub enum InnerDistanceMatrix {
    Full(FullDistanceMatrix<NodeIndex<IndexType>, FloatType>),
    Sub(SubDistanceMatrix<NodeIndex<IndexType>, FloatType>),
    NegLogSim(NegLogSimDistance<NodeIndex<IndexType>, FloatType, InnerKernel>),
    NegLog(NegLogDistance<NodeIndex<IndexType>, FloatType, InnerKernel>),
    PivotedNegLog(PivotedNegLogDistance<NodeIndex<IndexType>, FloatType, InnerPivotedKernel>),
    Embedding(EmbeddingDistanceMatrix<NodeIndex<IndexType>, FloatType>),
    KernelDistance(
        Box<KernelDistance<GaussianKernel<NodeIndex<IndexType>, InnerDistanceMatrix, FloatType>>>,
    ),
}

impl Distance<NodeIndex<IndexType>, FloatType> for InnerDistanceMatrix {
    fn get(&self, u: NodeIndex<IndexType>, v: NodeIndex<IndexType>) -> Option<FloatType> {
        match self {
            Self::Full(d) => Distance::get(d, u, v),
            Self::Sub(d) => Distance::get(d, u, v),
            Self::NegLogSim(d) => Distance::get(d, u, v),
            Self::NegLog(d) => Distance::get(d, u, v),
            Self::PivotedNegLog(d) => Distance::get(d, u, v),
            Self::Embedding(d) => Distance::get(d, u, v),
            Self::KernelDistance(d) => Distance::get(d.as_ref(), u, v),
        }
    }

    fn get_by_index(&self, i: usize, j: usize) -> FloatType {
        match self {
            Self::Full(d) => Distance::get_by_index(d, i, j),
            Self::Sub(d) => Distance::get_by_index(d, i, j),
            Self::NegLogSim(d) => Distance::get_by_index(d, i, j),
            Self::NegLog(d) => Distance::get_by_index(d, i, j),
            Self::PivotedNegLog(d) => Distance::get_by_index(d, i, j),
            Self::Embedding(d) => Distance::get_by_index(d, i, j),
            Self::KernelDistance(d) => {
                Distance::<NodeIndex<IndexType>, FloatType>::get_by_index(d.as_ref(), i, j)
            }
        }
    }

    fn shape(&self) -> (usize, usize) {
        match self {
            Self::Full(d) => Distance::shape(d),
            Self::Sub(d) => Distance::shape(d),
            Self::NegLogSim(d) => Distance::shape(d),
            Self::NegLog(d) => Distance::shape(d),
            Self::PivotedNegLog(d) => Distance::shape(d),
            Self::Embedding(d) => Distance::shape(d),
            Self::KernelDistance(d) => {
                Distance::<NodeIndex<IndexType>, FloatType>::shape(d.as_ref())
            }
        }
    }

    fn row_index(&self, u: NodeIndex<IndexType>) -> Option<usize> {
        match self {
            Self::Full(d) => Distance::row_index(d, u),
            Self::Sub(d) => Distance::row_index(d, u),
            Self::NegLogSim(d) => Distance::row_index(d, u),
            Self::NegLog(d) => Distance::row_index(d, u),
            Self::PivotedNegLog(d) => Distance::row_index(d, u),
            Self::Embedding(d) => Distance::row_index(d, u),
            Self::KernelDistance(d) => Distance::row_index(d.as_ref(), u),
        }
    }

    fn col_index(&self, u: NodeIndex<IndexType>) -> Option<usize> {
        match self {
            Self::Full(d) => Distance::col_index(d, u),
            Self::Sub(d) => Distance::col_index(d, u),
            Self::NegLogSim(d) => Distance::col_index(d, u),
            Self::NegLog(d) => Distance::col_index(d, u),
            Self::PivotedNegLog(d) => Distance::col_index(d, u),
            Self::Embedding(d) => Distance::col_index(d, u),
            Self::KernelDistance(d) => Distance::col_index(d.as_ref(), u),
        }
    }
}

pub fn extract_inner_distance(distance_matrix: &Bound<PyAny>) -> PyResult<InnerDistanceMatrix> {
    if let Ok(dm) = distance_matrix.extract::<PyRef<PyDistanceMatrix>>() {
        match dm.distance_matrix() {
            DistanceMatrixType::Full(d) => Ok(InnerDistanceMatrix::Full(d.clone())),
            DistanceMatrixType::Sub(d) => Ok(InnerDistanceMatrix::Sub(d.clone())),
        }
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyNegLogSimDistance>>() {
        Ok(InnerDistanceMatrix::NegLogSim(dm.matrix.clone()))
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyNegLogDistance>>() {
        Ok(InnerDistanceMatrix::NegLog(dm.matrix.clone()))
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyPivotedNegLogDistance>>() {
        Ok(InnerDistanceMatrix::PivotedNegLog(dm.matrix.clone()))
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyEmbeddingDistanceMatrix>>() {
        Ok(InnerDistanceMatrix::Embedding(dm.matrix.clone()))
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyKernelDistance>>() {
        Ok(InnerDistanceMatrix::KernelDistance(Box::new(
            dm.matrix.clone(),
        )))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Unsupported distance matrix type for wrapping",
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
            DistanceMatrixType::Sub(d) => Ok(f(d)),
        }
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyNegLogSimDistance>>() {
        Ok(f(&dm.matrix))
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyNegLogDistance>>() {
        Ok(f(&dm.matrix))
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyPivotedNegLogDistance>>() {
        Ok(f(&dm.matrix))
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyEmbeddingDistanceMatrix>>() {
        Ok(f(&dm.matrix))
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyKernelDistance>>() {
        Ok(f(&dm.matrix))
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
    #[new]
    #[pyo3(signature = (graph, kernel, alpha = 1.0, beta = 0.0))]
    pub fn new(
        graph: &PyGraphAdapter,
        kernel: &Bound<PyAny>,
        alpha: FloatType,
        beta: FloatType,
    ) -> PyResult<Self> {
        let inner_kernel = extract_inner_kernel(kernel)?;
        let matrix = match graph.graph() {
            GraphType::Graph(native_graph) => NegLogSimDistanceBuilder::new(inner_kernel)
                .alpha(alpha)
                .beta(beta)
                .build(native_graph)
                .map_err(pyo3::exceptions::PyValueError::new_err)?,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Unsupported graph type",
                ))
            }
        };
        Ok(Self { matrix })
    }

    pub fn get(&self, u: usize, v: usize) -> Option<FloatType> {
        self.matrix
            .get(node_index::<IndexType>(u), node_index::<IndexType>(v))
    }
}

#[pyclass]
#[pyo3(name = "NegLogDistance")]
pub struct PyNegLogDistance {
    pub(crate) matrix: NegLogDistance<NodeIndex<IndexType>, FloatType, InnerKernel>,
}

#[pymethods]
impl PyNegLogDistance {
    #[new]
    #[pyo3(signature = (graph, kernel, alpha = 1.0, beta = 0.0))]
    pub fn new(
        graph: &PyGraphAdapter,
        kernel: &Bound<PyAny>,
        alpha: FloatType,
        beta: FloatType,
    ) -> PyResult<Self> {
        let inner_kernel = extract_inner_kernel(kernel)?;
        let matrix = match graph.graph() {
            GraphType::Graph(native_graph) => NegLogDistanceBuilder::new(inner_kernel)
                .alpha(alpha)
                .beta(beta)
                .build(native_graph)
                .map_err(pyo3::exceptions::PyValueError::new_err)?,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Unsupported graph type",
                ))
            }
        };
        Ok(Self { matrix })
    }

    pub fn get(&self, u: usize, v: usize) -> Option<FloatType> {
        self.matrix
            .get(node_index::<IndexType>(u), node_index::<IndexType>(v))
    }
}

#[pyclass]
#[pyo3(name = "PivotedNegLogDistance")]
pub struct PyPivotedNegLogDistance {
    pub(crate) matrix: PivotedNegLogDistance<NodeIndex<IndexType>, FloatType, InnerPivotedKernel>,
}

#[pymethods]
impl PyPivotedNegLogDistance {
    #[new]
    #[pyo3(signature = (graph, kernel, alpha = 1.0, beta = 0.0))]
    pub fn new(
        graph: &PyGraphAdapter,
        kernel: &Bound<PyAny>,
        alpha: FloatType,
        beta: FloatType,
    ) -> PyResult<Self> {
        let inner_kernel = extract_inner_pivoted_kernel(kernel)?;
        let matrix = match graph.graph() {
            GraphType::Graph(native_graph) => PivotedNegLogDistanceBuilder::new(inner_kernel)
                .alpha(alpha)
                .beta(beta)
                .build(native_graph)
                .map_err(pyo3::exceptions::PyValueError::new_err)?,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Unsupported graph type",
                ))
            }
        };
        Ok(Self { matrix })
    }

    pub fn get(&self, u: usize, v: usize) -> Option<FloatType> {
        self.matrix
            .get(node_index::<IndexType>(u), node_index::<IndexType>(v))
    }
}

#[pyclass]
#[pyo3(name = "EmbeddingDistanceMatrix")]
pub struct PyEmbeddingDistanceMatrix {
    pub(crate) matrix: EmbeddingDistanceMatrix<NodeIndex<IndexType>, FloatType>,
}

#[pymethods]
impl PyEmbeddingDistanceMatrix {
    #[new]
    pub fn new(
        graph: &PyGraphAdapter,
        embedding: &crate::array::PyArray2,
        min_dist: FloatType,
    ) -> PyResult<Self> {
        let matrix = match graph.graph() {
            GraphType::Graph(native_graph) => {
                EmbeddingDistanceMatrix::new(native_graph, embedding.as_array().clone(), min_dist)
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Unsupported graph type",
                ))
            }
        };
        Ok(Self { matrix })
    }

    pub fn get(&self, u: usize, v: usize) -> Option<FloatType> {
        self.matrix
            .get(node_index::<IndexType>(u), node_index::<IndexType>(v))
    }
}

#[pyclass]
#[pyo3(name = "KernelDistance")]
pub struct PyKernelDistance {
    pub(crate) matrix:
        KernelDistance<GaussianKernel<NodeIndex<IndexType>, InnerDistanceMatrix, FloatType>>,
}

#[pymethods]
impl PyKernelDistance {
    #[new]
    pub fn new(distance_matrix: &Bound<PyAny>, gamma: FloatType) -> PyResult<Self> {
        let inner = extract_inner_distance(distance_matrix)?;
        let kernel = GaussianKernel::new(inner, gamma);
        let matrix = KernelDistance::new(kernel);
        Ok(Self { matrix })
    }

    pub fn get(&self, u: usize, v: usize) -> Option<FloatType> {
        self.matrix
            .get(node_index::<IndexType>(u), node_index::<IndexType>(v))
    }
}

use petgraph_distance::SparseSymmetricMatrix;

#[pyclass(from_py_object)]
#[pyo3(name = "Laplacian")]
#[derive(Clone)]
pub struct PyLaplacian {
    pub(crate) matrix: SparseSymmetricMatrix<FloatType>,
}

#[pyclass]
#[pyo3(name = "StandardLaplacian")]
pub struct PyStandardLaplacian;

#[pymethods]
impl PyStandardLaplacian {
    #[staticmethod]
    pub fn build(graph: &PyGraphAdapter, length: Py<PyAny>) -> PyResult<PyLaplacian> {
        let matrix = match graph.graph() {
            GraphType::Graph(native_graph) => {
                let length_fn = |edge: petgraph::graph::EdgeReference<Py<PyAny>>| -> FloatType {
                    Python::attach(|py| {
                        let result = length.call1(py, (edge.id().index(),));
                        match result {
                            Ok(value) => value.extract::<FloatType>(py).unwrap_or(1.0),
                            Err(_) => 1.0,
                        }
                    })
                };
                SparseSymmetricMatrix::standard_laplacian(native_graph, length_fn)
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Unsupported graph type",
                ))
            }
        };
        Ok(PyLaplacian { matrix })
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDistanceMatrix>()?;
    m.add_class::<PyNegLogSimDistance>()?;
    m.add_class::<PyNegLogDistance>()?;
    m.add_class::<PyPivotedNegLogDistance>()?;
    m.add_class::<PyEmbeddingDistanceMatrix>()?;
    m.add_class::<PyKernelDistance>()?;
    m.add_class::<PyLaplacian>()?;
    m.add_class::<PyStandardLaplacian>()?;
    Ok(())
}
