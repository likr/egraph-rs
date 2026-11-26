//! Kernel wrappers and dispatch enum for full graph similarity kernels.

use crate::{
    FloatType,
    graph::{GraphType, IndexType, PyGraphAdapter},
};
use petgraph::graph::NodeIndex;
use petgraph_linalg_kernel::{
    BandPassDiffusionKernel, DiffusionKernel, EmbeddingKernel, EmbeddingKernelBuilder,
    ExponentialKernel, ExponentialKernelBuilder, GaussianKernel, GaussianKernelBuilder, Kernel,
    LowRankDiffusionKernel, LowRankMultiscaleDiffusionKernel, MultiscaleDiffusionKernel, TKernel,
    TKernelBuilder,
};
use pyo3::prelude::*;

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
        let kernel = GaussianKernelBuilder::new(gamma)
            .build(Box::new(inner))
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { kernel })
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
        let kernel = ExponentialKernelBuilder::new(gamma)
            .build(Box::new(inner))
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { kernel })
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
        let kernel = TKernelBuilder::new(dof)
            .build(Box::new(inner))
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { kernel })
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

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEmbeddingKernel>()?;
    m.add_class::<PyEmbeddingKernelBuilder>()?;
    m.add_class::<PyGaussianKernel>()?;
    m.add_class::<PyGaussianKernelBuilder>()?;
    m.add_class::<PyExponentialKernel>()?;
    m.add_class::<PyExponentialKernelBuilder>()?;
    m.add_class::<PyTKernel>()?;
    m.add_class::<PyTKernelBuilder>()?;
    Ok(())
}
