//! Negative-log and pivoted negative-log similarity distance bindings.

use super::kernels::{InnerKernel, extract_inner_kernel};
use crate::{
    FloatType,
    graph::{GraphType, IndexType, PyGraphAdapter},
};
use petgraph::{graph::NodeIndex, stable_graph::node_index};
use petgraph_linalg_kernel::{
    Distance, NegLogDistance, NegLogDistanceBuilder, NegLogSimDistance, NegLogSimDistanceBuilder,
    PivotedBandPassDiffusionKernel, PivotedDiffusionKernel, PivotedKernel,
    PivotedMultiscaleDiffusionKernel, PivotedNegLogDistance, PivotedNegLogDistanceBuilder,
    PivotedNegLogSimDistance, PivotedNegLogSimDistanceBuilder,
};
use pyo3::prelude::*;

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
                ));
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
                ));
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
                ));
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
            ));
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

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNegLogSimDistance>()?;
    m.add_class::<PyNegLogSimDistanceBuilder>()?;
    m.add_class::<PyNegLogDistance>()?;
    m.add_class::<PyNegLogDistanceBuilder>()?;
    m.add_class::<PyPivotedNegLogDistance>()?;
    m.add_class::<PyPivotedNegLogDistanceBuilder>()?;
    m.add_class::<PyPivotedNegLogSimDistance>()?;
    m.add_class::<PyPivotedNegLogSimDistanceBuilder>()?;
    Ok(())
}
