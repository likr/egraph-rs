//! Typed Python result classes owning NumPy buffers for community detection and clustering.

use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;

use crate::errors::invalid_input_err;

/// Result returned by `louvain_communities`.
#[pyclass(name = "LouvainResult", module = "egraph._core")]
pub struct PyLouvainResult {
    labels: Py<PyArray1<u32>>,
    num_communities: usize,
    modularity: f64,
    passes: usize,
}

impl PyLouvainResult {
    pub(super) fn from_core(
        py: Python<'_>,
        res: egraph::clustering::LouvainResult,
    ) -> PyResult<Self> {
        Ok(Self {
            labels: PyArray1::from_vec(py, res.labels).unbind(),
            num_communities: res.num_communities,
            modularity: res.modularity,
            passes: res.passes,
        })
    }
}

#[pymethods]
impl PyLouvainResult {
    /// Canonical community assignment in `0..num_communities` for each vertex (`uint32` array).
    #[getter]
    fn labels(&self, py: Python<'_>) -> Py<PyArray1<u32>> {
        self.labels.clone_ref(py)
    }

    /// Number of distinct non-empty communities detected.
    #[getter]
    fn num_communities(&self) -> usize {
        self.num_communities
    }

    /// Modularity of the returned partition.
    #[getter]
    fn modularity(&self) -> f64 {
        self.modularity
    }

    /// Number of multi-level passes committed.
    #[getter]
    fn passes(&self) -> usize {
        self.passes
    }

    fn __repr__(&self) -> String {
        format!(
            "LouvainResult(num_communities={}, modularity={:.6}, passes={})",
            self.num_communities, self.modularity, self.passes
        )
    }
}

/// Result returned by `leiden_communities`.
#[pyclass(name = "LeidenResult", module = "egraph._core")]
pub struct PyLeidenResult {
    labels: Py<PyArray1<u32>>,
    num_communities: usize,
    quality: f64,
    modularity: f64,
    passes: usize,
}

impl PyLeidenResult {
    pub(super) fn from_core(
        py: Python<'_>,
        res: egraph::clustering::LeidenResult,
    ) -> PyResult<Self> {
        Ok(Self {
            labels: PyArray1::from_vec(py, res.labels).unbind(),
            num_communities: res.num_communities,
            quality: res.quality,
            modularity: res.modularity,
            passes: res.passes,
        })
    }
}

#[pymethods]
impl PyLeidenResult {
    /// Canonical community assignment in `0..num_communities` for each vertex (`uint32` array).
    #[getter]
    fn labels(&self, py: Python<'_>) -> Py<PyArray1<u32>> {
        self.labels.clone_ref(py)
    }

    /// Number of distinct non-empty communities detected.
    #[getter]
    fn num_communities(&self) -> usize {
        self.num_communities
    }

    /// Objective quality value for the configured Leiden objective.
    #[getter]
    fn quality(&self) -> f64 {
        self.quality
    }

    /// Standard normalized modularity evaluated at `resolution=1.0`.
    #[getter]
    fn modularity(&self) -> f64 {
        self.modularity
    }

    /// Number of multi-level passes committed.
    #[getter]
    fn passes(&self) -> usize {
        self.passes
    }

    fn __repr__(&self) -> String {
        format!(
            "LeidenResult(num_communities={}, quality={:.6}, modularity={:.6}, passes={})",
            self.num_communities, self.quality, self.modularity, self.passes
        )
    }
}

/// Result returned by `kmeans_plus_plus`.
#[pyclass(name = "KMeansResult", module = "egraph._core")]
pub struct PyKMeansResult {
    labels: Py<PyArray1<u32>>,
    centroids: Py<PyArray2<f64>>,
    k: usize,
    dims: usize,
    inertia: f64,
    iterations: usize,
    converged: bool,
}

impl PyKMeansResult {
    pub(super) fn from_core(
        py: Python<'_>,
        res: egraph::clustering::KMeansResult,
    ) -> PyResult<Self> {
        let centroids_arr = Array2::from_shape_vec((res.k, res.dims), res.centroids)
            .map_err(|e| invalid_input_err(e.to_string()))?;
        Ok(Self {
            labels: PyArray1::from_vec(py, res.labels).unbind(),
            centroids: PyArray2::from_owned_array(py, centroids_arr).unbind(),
            k: res.k,
            dims: res.dims,
            inertia: res.inertia,
            iterations: res.iterations,
            converged: res.converged,
        })
    }
}

#[pymethods]
impl PyKMeansResult {
    /// Canonical cluster assignment in `0..k` for each input point (`uint32` array).
    #[getter]
    fn labels(&self, py: Python<'_>) -> Py<PyArray1<u32>> {
        self.labels.clone_ref(py)
    }

    /// Centroid coordinates of shape `(k, dims)` (`float64` array).
    #[getter]
    fn centroids(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        self.centroids.clone_ref(py)
    }

    /// Number of target clusters `k`.
    #[getter]
    fn k(&self) -> usize {
        self.k
    }

    /// Feature dimensionality `dims`.
    #[getter]
    fn dims(&self) -> usize {
        self.dims
    }

    /// Total within-cluster sum of squared Euclidean distances to assigned centroids.
    #[getter]
    fn inertia(&self) -> f64 {
        self.inertia
    }

    /// Number of Lloyd iterations executed in the best initialization.
    #[getter]
    fn iterations(&self) -> usize {
        self.iterations
    }

    /// Whether the centroid shift converged within `tol`.
    #[getter]
    fn converged(&self) -> bool {
        self.converged
    }

    fn __repr__(&self) -> String {
        format!(
            "KMeansResult(k={}, dims={}, inertia={:.6}, iterations={}, converged={})",
            self.k, self.dims, self.inertia, self.iterations, self.converged
        )
    }
}

/// Result returned by `label_propagation`.
#[pyclass(name = "LabelPropagationResult", module = "egraph._core")]
pub struct PyLabelPropagationResult {
    labels: Py<PyArray1<u32>>,
    num_communities: usize,
    iterations: usize,
    converged: bool,
}

impl PyLabelPropagationResult {
    pub(super) fn from_core(
        py: Python<'_>,
        res: egraph::clustering::LabelPropagationResult,
    ) -> PyResult<Self> {
        Ok(Self {
            labels: PyArray1::from_vec(py, res.labels).unbind(),
            num_communities: res.num_communities,
            iterations: res.iterations,
            converged: res.converged,
        })
    }
}

#[pymethods]
impl PyLabelPropagationResult {
    /// Canonical community assignment in `0..num_communities` for each vertex (`uint32` array).
    #[getter]
    fn labels(&self, py: Python<'_>) -> Py<PyArray1<u32>> {
        self.labels.clone_ref(py)
    }

    /// Number of distinct non-empty communities detected.
    #[getter]
    fn num_communities(&self) -> usize {
        self.num_communities
    }

    /// Number of asynchronous propagation sweeps executed.
    #[getter]
    fn iterations(&self) -> usize {
        self.iterations
    }

    /// Whether propagation reached a fixed point.
    #[getter]
    fn converged(&self) -> bool {
        self.converged
    }

    fn __repr__(&self) -> String {
        format!(
            "LabelPropagationResult(num_communities={}, iterations={}, converged={})",
            self.num_communities, self.iterations, self.converged
        )
    }
}

/// Result returned by `spectral_clustering`.
#[pyclass(name = "SpectralClusteringResult", module = "egraph._core")]
pub struct PySpectralClusteringResult {
    labels: Py<PyArray1<u32>>,
    num_clusters: usize,
    eigen_iterations: usize,
}

impl PySpectralClusteringResult {
    pub(super) fn from_core(
        py: Python<'_>,
        res: egraph::clustering::SpectralClusteringResult,
    ) -> PyResult<Self> {
        Ok(Self {
            labels: PyArray1::from_vec(py, res.labels).unbind(),
            num_clusters: res.num_clusters,
            eigen_iterations: res.eigen_iterations,
        })
    }
}

#[pymethods]
impl PySpectralClusteringResult {
    /// Canonical cluster assignment in `0..num_clusters` for each vertex (`uint32` array).
    #[getter]
    fn labels(&self, py: Python<'_>) -> Py<PyArray1<u32>> {
        self.labels.clone_ref(py)
    }

    /// Number of distinct non-empty clusters detected.
    #[getter]
    fn num_clusters(&self) -> usize {
        self.num_clusters
    }

    /// Number of Lanczos thick-restart eigensolver iterations executed.
    #[getter]
    fn eigen_iterations(&self) -> usize {
        self.eigen_iterations
    }

    fn __repr__(&self) -> String {
        format!(
            "SpectralClusteringResult(num_clusters={}, eigen_iterations={})",
            self.num_clusters, self.eigen_iterations
        )
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLouvainResult>()?;
    m.add_class::<PyLeidenResult>()?;
    m.add_class::<PyKMeansResult>()?;
    m.add_class::<PyLabelPropagationResult>()?;
    m.add_class::<PySpectralClusteringResult>()?;
    Ok(())
}
