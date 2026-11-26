//! High-dimensional probability, symmetrization, UMAP curve, and repulsion scaling bindings.

use egraph::neighborhood::{self as nb, PerplexityConfig, SymmetrizationMode};
use pyo3::prelude::*;

use super::types::{PyKnnGraph, PySparseProbabilities};
use crate::array_extract::{extract_f64_1d, extract_perplexities, extract_usize_1d};
use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{StrictF64, StrictU64, StrictUsize, core_error_to_py, extract_u32_1d};

/// Compute symmetric high-dimensional joint probabilities `P` from a `KnnGraph` and target perplexities.
///
/// Accepts either a 1D array of length `knn.num_nodes` or a scalar float perplexity broadcast to all vertices.
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (knn, perplexities, *, power = StrictF64(2.0), sigma_iters = StrictUsize(50), sigma_tolerance = StrictF64(1e-5), row_normalize = false),
    text_signature = "(knn, perplexities, *, power=2.0, sigma_iters=50, sigma_tolerance=1e-5, row_normalize=False)"
)]
fn compute_knn_probabilities(
    py: Python<'_>,
    knn: &PyKnnGraph,
    perplexities: &Bound<'_, PyAny>,
    power: StrictF64,
    sigma_iters: StrictUsize,
    sigma_tolerance: StrictF64,
    row_normalize: bool,
) -> PyResult<PySparseProbabilities> {
    let perps_vec = extract_perplexities(perplexities, knn.inner().num_nodes())?;
    let cfg = PerplexityConfig {
        power: power.0,
        sigma_iters: sigma_iters.0,
        sigma_tolerance: sigma_tolerance.0,
        symmetrization: if row_normalize {
            SymmetrizationMode::RowNormalized
        } else {
            SymmetrizationMode::StandardTsne
        },
    };
    let probs = py
        .detach(|| nb::compute_knn_probabilities(knn.inner(), &perps_vec, &cfg))
        .map_err(core_error_to_py)?;
    Ok(PySparseProbabilities::from_inner(probs))
}

/// Compute row-normalized symmetrized graph probabilities `P = (D^{-1} A + A^T D^{-1}) / (2N)`.
#[pyfunction]
#[pyo3(signature = (graph), text_signature = "(graph)")]
fn graph_row_normalized_probabilities(
    py: Python<'_>,
    graph: &PyCsrGraph,
) -> PyResult<PySparseProbabilities> {
    let g = graph.inner();
    let probs = py
        .detach(|| nb::graph_row_normalized_probabilities(g))
        .map_err(core_error_to_py)?;
    Ok(PySparseProbabilities::from_inner(probs))
}

/// Compute row-normalized symmetrized 1+2-hop joint probabilities with optional capping and `(d_i d_j)^(-degree_beta)` decay.
#[pyfunction]
#[pyo3(
    signature = (graph, w2 = StrictF64(0.5), max_2hop_per_node = None, *, degree_beta = StrictF64(0.0), seed = StrictU64(42)),
    text_signature = "(graph, w2=0.5, max_2hop_per_node=None, *, degree_beta=0.0, seed=42)"
)]
fn graph_12hop_row_normalized_probabilities(
    py: Python<'_>,
    graph: &PyCsrGraph,
    w2: StrictF64,
    max_2hop_per_node: Option<StrictUsize>,
    degree_beta: StrictF64,
    seed: StrictU64,
) -> PyResult<PySparseProbabilities> {
    let g = graph.inner();
    let w2_val = w2.0;
    let cap_opt = max_2hop_per_node.map(|v| v.0);
    let beta_val = degree_beta.0;
    let seed_val = seed.0;
    let probs = py
        .detach(|| {
            nb::graph_12hop_row_normalized_probabilities_with_degree_decay(
                g, w2_val, cap_opt, beta_val, seed_val,
            )
        })
        .map_err(core_error_to_py)?;
    Ok(PySparseProbabilities::from_inner(probs))
}

/// Compute row-normalized symmetrized joint probabilities from raw directed CSR affinities.
#[pyfunction]
#[pyo3(
    signature = (num_nodes, indptr, indices, weights),
    text_signature = "(num_nodes, indptr, indices, weights)"
)]
fn symmetrize_row_normalized(
    py: Python<'_>,
    num_nodes: StrictUsize,
    indptr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    weights: &Bound<'_, PyAny>,
) -> PyResult<PySparseProbabilities> {
    let indptr_vec = extract_usize_1d(indptr, "indptr")?;
    let indices_vec = extract_u32_1d(indices, "indices")?;
    let weights_vec = extract_f64_1d(weights, "weights")?;
    let probs = py
        .detach(|| {
            nb::symmetrize_row_normalized(num_nodes.0, &indptr_vec, &indices_vec, &weights_vec)
        })
        .map_err(core_error_to_py)?;
    Ok(PySparseProbabilities::from_inner(probs))
}

/// Compute the UMAP / GuMap fuzzy simplicial set probability matrix from a `KnnGraph`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (knn, *, sigma_iters = StrictUsize(64), sigma_tolerance = StrictF64(1e-5)),
    text_signature = "(knn, *, sigma_iters=64, sigma_tolerance=1e-5)"
)]
fn compute_fuzzy_simplicial_set(
    py: Python<'_>,
    knn: &PyKnnGraph,
    sigma_iters: StrictUsize,
    sigma_tolerance: StrictF64,
) -> PyResult<PySparseProbabilities> {
    let probs = py
        .detach(|| nb::compute_fuzzy_simplicial_set(knn.inner(), sigma_iters.0, sigma_tolerance.0))
        .map_err(core_error_to_py)?;
    Ok(PySparseProbabilities::from_inner(probs))
}

/// Fit low-dimensional UMAP curve parameters `(a, b)` from `min_dist` and `spread`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (min_dist = StrictF64(0.1), spread = StrictF64(1.0)),
    text_signature = "(min_dist=0.1, spread=1.0)"
)]
fn find_ab_params(py: Python<'_>, min_dist: StrictF64, spread: StrictF64) -> PyResult<(f64, f64)> {
    py.detach(|| nb::find_ab_params(min_dist.0, spread.0))
        .map_err(core_error_to_py)
}

/// Compute Horvitz-Thompson repulsion density scaling `gamma_non2 * sqrt(N / max(k_neg, 1))`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (gamma_non2, num_nodes, neg_per_node),
    text_signature = "(gamma_non2, num_nodes, neg_per_node)"
)]
fn compute_horvitz_thompson_gamma(
    py: Python<'_>,
    gamma_non2: StrictF64,
    num_nodes: StrictUsize,
    neg_per_node: StrictUsize,
) -> PyResult<f64> {
    py.detach(|| nb::compute_horvitz_thompson_gamma(gamma_non2.0, num_nodes.0, neg_per_node.0))
        .map_err(core_error_to_py)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_knn_probabilities, m)?)?;
    m.add_function(wrap_pyfunction!(graph_row_normalized_probabilities, m)?)?;
    m.add_function(wrap_pyfunction!(
        graph_12hop_row_normalized_probabilities,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(symmetrize_row_normalized, m)?)?;
    m.add_function(wrap_pyfunction!(compute_fuzzy_simplicial_set, m)?)?;
    m.add_function(wrap_pyfunction!(find_ab_params, m)?)?;
    m.add_function(wrap_pyfunction!(compute_horvitz_thompson_gamma, m)?)?;
    Ok(())
}
