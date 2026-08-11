use pyo3::prelude::*;
use crate::{
    graph::{GraphType, IndexType, PyGraphAdapter},
    FloatType,
};
use petgraph::{graph::NodeIndex, stable_graph::node_index, visit::EdgeRef};
use petgraph_algorithm_shortest_path::{DistanceMatrix, FullDistanceMatrix, PivotedDistanceMatrix};
use petgraph_distance::{Distance, Kernel, KernelDistance, GaussianKernel, ExponentialKernel, TKernel};
use petgraph_linalg_diffusion_kernel::{
    DiffusionKernel, LowRankDiffusionKernel, LowRankMultiscaleDiffusionKernel,
    MultiscaleDiffusionKernel, NegLogDistance, NegLogDistanceBuilder, NegLogSimDistance,
    NegLogSimDistanceBuilder, PivotedDiffusionKernel, PivotedKernel,
    PivotedMultiscaleDiffusionKernel, PivotedNegLogDistance, PivotedNegLogDistanceBuilder,
};
use petgraph_linalg_embedding_kernel::EmbeddingKernel;


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

    pub fn get_by_index(&self, i: usize, j: usize) -> FloatType {
        match &self.distance_matrix {
            DistanceMatrixType::Full(dm) => DistanceMatrix::get_by_index(dm, i, j),
            DistanceMatrixType::Pivoted(dm) => DistanceMatrix::get_by_index(dm, i, j),
        }
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
}

#[derive(Clone)]
pub enum InnerKernel {
    Diffusion(DiffusionKernel<FloatType>),
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

    fn build(
        &self,
        graph: &PyGraphAdapter,
        kernel: &Bound<PyAny>,
    ) -> PyResult<PyNegLogDistance> {
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

#[pyclass]
#[pyo3(name = "EmbeddingKernel")]
pub struct PyEmbeddingKernel {
    pub(crate) kernel: EmbeddingKernel<NodeIndex<IndexType>, FloatType>,
}

#[pymethods]
impl PyEmbeddingKernel {
    #[new]
    pub fn new(
        graph: &PyGraphAdapter,
        embedding: &crate::array::PyArray2,
    ) -> PyResult<Self> {
        let kernel = match graph.graph() {
            GraphType::Graph(native_graph) => {
                EmbeddingKernel::new(native_graph, embedding.as_array().clone())
            }
            GraphType::DiGraph(native_graph) => {
                EmbeddingKernel::new(native_graph, embedding.as_array().clone())
            }
        };
        Ok(Self { kernel })
    }

    pub fn get(&self, u: usize, v: usize) -> Option<FloatType> {
        self.kernel.get(NodeIndex::new(u), NodeIndex::new(v))
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
    m.add_class::<PyNegLogSimDistanceBuilder>()?;
    m.add_class::<PyNegLogDistance>()?;
    m.add_class::<PyNegLogDistanceBuilder>()?;
    m.add_class::<PyPivotedNegLogDistance>()?;
    m.add_class::<PyPivotedNegLogDistanceBuilder>()?;
    m.add_class::<PyEmbeddingKernel>()?;
    m.add_class::<PyGaussianKernel>()?;
    m.add_class::<PyExponentialKernel>()?;
    m.add_class::<PyTKernel>()?;
    m.add_class::<PyKernelDistance>()?;
    m.add_class::<PyLaplacian>()?;
    m.add_class::<PyStandardLaplacian>()?;
    Ok(())
}
