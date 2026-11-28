//! Community detection, graph coarsening, and hierarchical projection bindings.

mod coarsen_backbone;
mod results;

#[cfg(test)]
mod tests;

use egraph::clustering as egraph_clustering;
use egraph::core as egraph_core;
use egraph_clustering::{
    KMeansConfig, LabelPropagationConfig, LeidenConfig, LeidenObjective, LouvainConfig,
    SpectralClusteringConfig, coarsen_graph as clu_coarsen_graph,
    kmeans_plus_plus as clu_kmeans_plus_plus, label_propagation as clu_label_propagation,
    leiden as clu_leiden, louvain as clu_louvain,
    project_clustered_layout as clu_project_clustered_layout,
    spectral_clustering as clu_spectral_clustering,
};
use egraph_core::CoreError;
use numpy::PyArray1;
use pyo3::prelude::*;

use self::results::{
    PyKMeansResult, PyLabelPropagationResult, PyLeidenResult, PyLouvainResult,
    PySpectralClusteringResult,
};
use crate::array_extract::extract_f64_2d;
use crate::csr_graph::PyCsrGraph;
use crate::errors::{invalid_input_err, non_finite_value_err};
use crate::ffi_convert::{
    StrictF64, StrictU64, StrictUsize, core_error_to_py, extract_u32_1d, normalize_enum_key,
};
use crate::layout_buffers::{Layout2dInner, PyLayout2d};

pub(super) fn parse_leiden_objective(raw: &str) -> Result<LeidenObjective, String> {
    match normalize_enum_key(raw).as_str() {
        "modularity" | "mod" => Ok(LeidenObjective::Modularity),
        "cpm" | "constantpottsmodel" => Ok(LeidenObjective::Cpm),
        _ => Err(format!(
            "unknown Leiden objective {raw:?}; expected 'modularity' or 'cpm'"
        )),
    }
}

/// Cluster `points` of shape `(N, D)` using deterministic K-Means++ initialization and Lloyd refinement.
///
/// Returns a `KMeansResult` with `.labels`, `.centroids`, `.k`, `.dims`, `.inertia`,
/// `.iterations`, and `.converged`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (points, *, k = StrictUsize(2), max_iterations = StrictUsize(100), tol = StrictF64(1e-4), n_init = StrictUsize(1), seed = StrictU64(42)),
    text_signature = "(points, *, k=2, max_iterations=100, tol=1e-4, n_init=1, seed=42)"
)]
fn kmeans_plus_plus(
    py: Python<'_>,
    points: &Bound<'_, PyAny>,
    k: StrictUsize,
    max_iterations: StrictUsize,
    tol: StrictF64,
    n_init: StrictUsize,
    seed: StrictU64,
) -> PyResult<PyKMeansResult> {
    let (flat, n, dims) = extract_f64_2d(points, "points")?;
    let cfg = KMeansConfig::new()
        .with_k(k.0)
        .with_max_iterations(max_iterations.0)
        .with_tol(tol.0)
        .with_n_init(n_init.0)
        .with_seed(seed.0);
    let res = py
        .detach(|| clu_kmeans_plus_plus(&flat, n, dims, &cfg))
        .map_err(core_error_to_py)?;
    PyKMeansResult::from_core(py, res)
}

/// Partition an undirected `CsrGraph` into `k` clusters using Ng-Jordan-Weiss spectral clustering.
///
/// Rejects directed graphs with `ValueError` and returns a `SpectralClusteringResult`.
/// Releases the Python GIL during computation.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (graph, *, k = StrictUsize(2), max_eigen_iterations = StrictUsize(300), eigen_tol = StrictF64(1e-4), kmeans_max_iterations = StrictUsize(100), kmeans_tol = StrictF64(1e-4), seed = StrictU64(42)),
    text_signature = "(graph, *, k=2, max_eigen_iterations=300, eigen_tol=1e-4, kmeans_max_iterations=100, kmeans_tol=1e-4, seed=42)"
)]
fn spectral_clustering(
    py: Python<'_>,
    graph: &PyCsrGraph,
    k: StrictUsize,
    max_eigen_iterations: StrictUsize,
    eigen_tol: StrictF64,
    kmeans_max_iterations: StrictUsize,
    kmeans_tol: StrictF64,
    seed: StrictU64,
) -> PyResult<PySpectralClusteringResult> {
    let cfg = SpectralClusteringConfig::new()
        .with_k(k.0)
        .with_max_eigen_iterations(max_eigen_iterations.0)
        .with_eigen_tol(eigen_tol.0)
        .with_kmeans_max_iterations(kmeans_max_iterations.0)
        .with_kmeans_tol(kmeans_tol.0)
        .with_seed(seed.0);
    let g = graph.inner();
    let res = py
        .detach(|| clu_spectral_clustering(&*g, &cfg))
        .map_err(core_error_to_py)?;
    PySpectralClusteringResult::from_core(py, res)
}

/// Detect communities on an undirected `CsrGraph` via multi-level Louvain modularity optimization.
///
/// Returns a `LouvainResult` with `.labels`, `.num_communities`, `.modularity`, and `.passes`.
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (graph, *, max_passes = StrictUsize(100), resolution = StrictF64(1.0), min_modularity_gain = StrictF64(1e-7), seed = StrictU64(42)),
    text_signature = "(graph, *, max_passes=100, resolution=1.0, min_modularity_gain=1e-7, seed=42)"
)]
fn louvain_communities(
    py: Python<'_>,
    graph: &PyCsrGraph,
    max_passes: StrictUsize,
    resolution: StrictF64,
    min_modularity_gain: StrictF64,
    seed: StrictU64,
) -> PyResult<PyLouvainResult> {
    let cfg = LouvainConfig::new()
        .with_max_passes(max_passes.0)
        .with_resolution(resolution.0)
        .with_min_modularity_gain(min_modularity_gain.0)
        .with_seed(seed.0);
    let g = graph.inner();
    let res = py
        .detach(|| clu_louvain(&*g, &cfg))
        .map_err(core_error_to_py)?;
    PyLouvainResult::from_core(py, res)
}

/// Detect communities on an undirected `CsrGraph` using the Leiden algorithm (Traag et al., 2019).
///
/// Supports `"modularity"` and `"cpm"` objectives while guaranteeing internally connected
/// communities. Returns a `LeidenResult` with `.labels`, `.num_communities`, `.quality`,
/// `.modularity`, and `.passes`. Releases the Python GIL during computation.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (
        graph,
        *,
        objective = "modularity",
        resolution = StrictF64(1.0),
        randomness = StrictF64(0.01),
        max_passes = StrictUsize(100),
        min_quality_gain = StrictF64(1e-7),
        seed = StrictU64(42)
    ),
    text_signature = "(graph, *, objective='modularity', resolution=1.0, randomness=0.01, max_passes=100, min_quality_gain=1e-7, seed=42)"
)]
fn leiden_communities(
    py: Python<'_>,
    graph: &PyCsrGraph,
    objective: &str,
    resolution: StrictF64,
    randomness: StrictF64,
    max_passes: StrictUsize,
    min_quality_gain: StrictF64,
    seed: StrictU64,
) -> PyResult<PyLeidenResult> {
    let parsed_objective = parse_leiden_objective(objective).map_err(invalid_input_err)?;
    let cfg = LeidenConfig::new()
        .with_objective(parsed_objective)
        .with_resolution(resolution.0)
        .with_randomness(randomness.0)
        .with_max_passes(max_passes.0)
        .with_min_quality_gain(min_quality_gain.0)
        .with_seed(seed.0);
    let g = graph.inner();
    let res = py
        .detach(|| clu_leiden(&*g, &cfg))
        .map_err(core_error_to_py)?;
    PyLeidenResult::from_core(py, res)
}

/// Detect communities on an undirected `CsrGraph` via asynchronous deterministic Label Propagation.
///
/// Returns a `LabelPropagationResult` with `.labels`, `.num_communities`, `.iterations`, and
/// `.converged`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (graph, *, max_iterations = StrictUsize(100), seed = StrictU64(42)),
    text_signature = "(graph, *, max_iterations=100, seed=42)"
)]
fn label_propagation(
    py: Python<'_>,
    graph: &PyCsrGraph,
    max_iterations: StrictUsize,
    seed: StrictU64,
) -> PyResult<PyLabelPropagationResult> {
    let cfg = LabelPropagationConfig::new()
        .with_max_iterations(max_iterations.0)
        .with_seed(seed.0);
    let g = graph.inner();
    let res = py
        .detach(|| clu_label_propagation(&*g, &cfg))
        .map_err(core_error_to_py)?;
    PyLabelPropagationResult::from_core(py, res)
}

type CoarsenReturn<'py> = (
    PyCsrGraph,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    usize,
);

/// Coarsen an undirected `CsrGraph` by contracting vertices sharing a cluster label.
///
/// Returns `(coarse_graph, node_to_coarse, coarse_sizes, num_clusters)`.
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (graph, labels), text_signature = "(graph, labels)")]
fn coarsen_graph<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
    labels: &Bound<'py, PyAny>,
) -> PyResult<CoarsenReturn<'py>> {
    let labels_vec = extract_u32_1d(labels, "labels")?;
    let g = graph.inner();
    let res = py
        .detach(|| clu_coarsen_graph(&*g, &labels_vec))
        .map_err(core_error_to_py)?;
    Ok((
        PyCsrGraph::from_csr_graph(res.graph),
        PyArray1::from_vec(py, res.node_to_coarse),
        PyArray1::from_vec(py, res.coarse_sizes),
        res.num_clusters,
    ))
}

/// Project a coarse cluster `Layout2d` onto fine vertices specified by `node_to_coarse`.
///
/// Preserves `coarse_layout.dtype` (`"float64"` or `"float32"`). Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (coarse_layout, node_to_coarse, *, spread_radius = StrictF64(0.1), seed = StrictU64(42)),
    text_signature = "(coarse_layout, node_to_coarse, *, spread_radius=0.1, seed=42)"
)]
fn project_clustered_layout(
    py: Python<'_>,
    coarse_layout: &PyLayout2d,
    node_to_coarse: &Bound<'_, PyAny>,
    spread_radius: StrictF64,
    seed: StrictU64,
) -> PyResult<PyLayout2d> {
    if !spread_radius.0.is_finite() {
        return Err(non_finite_value_err(
            "non-finite floating-point value in spread_radius",
        ));
    }
    let n2c = extract_u32_1d(node_to_coarse, "node_to_coarse")?;
    let r_f64 = spread_radius.0;
    let seed_val = seed.0;
    py.detach(|| match coarse_layout.inner() {
        Layout2dInner::F64(buf) => {
            let soa = buf.to_layout2d();
            let fine = clu_project_clustered_layout(&soa, &n2c, r_f64, seed_val)?;
            Ok(PyLayout2d::from_f64_layout(&fine))
        }
        Layout2dInner::F32(buf) => {
            let r_f32 = r_f64 as f32;
            if !r_f32.is_finite() {
                return Err(CoreError::non_finite(
                    "project_clustered_layout::spread_radius",
                ));
            }
            let soa = buf.to_layout2d();
            let fine = clu_project_clustered_layout(&soa, &n2c, r_f32, seed_val)?;
            Ok(PyLayout2d::from_f32_layout(&fine))
        }
    })
    .map_err(core_error_to_py)
}

/// Register community detection and coarsening functions on `egraph._core`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    results::register(m)?;
    m.add_function(wrap_pyfunction!(kmeans_plus_plus, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_clustering, m)?)?;
    m.add_function(wrap_pyfunction!(louvain_communities, m)?)?;
    m.add_function(wrap_pyfunction!(leiden_communities, m)?)?;
    m.add_function(wrap_pyfunction!(label_propagation, m)?)?;
    m.add_function(wrap_pyfunction!(coarsen_graph, m)?)?;
    m.add_function(wrap_pyfunction!(project_clustered_layout, m)?)?;
    coarsen_backbone::register(m)?;
    Ok(())
}
