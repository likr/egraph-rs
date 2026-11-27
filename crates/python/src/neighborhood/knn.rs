//! k-NN graph extraction, adaptive perplexity, and distance-floor bindings.

use egraph::neighborhood::{self as nb, HybridKnnConfig, TwoHopSamplingMode};
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::types::{PyKnnGraph, PySparse12HopPairs, knn_stats_to_py_dict};
use crate::array_extract::{extract_f64_1d, extract_perplexities};
use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    StrictF64, StrictU64, StrictUsize, core_error_to_py, extract_coords_any, extract_u32_1d,
    normalize_enum_key,
};

pub(crate) fn parse_two_hop_mode(raw: &str) -> PyResult<TwoHopSamplingMode> {
    match normalize_enum_key(raw).as_str() {
        "degreeorderedhubsuppressed" | "degreeordered" | "hubsuppressed" => {
            Ok(TwoHopSamplingMode::DegreeOrderedHubSuppressed)
        }
        "twosteprandomwalk" | "randomwalk" => Ok(TwoHopSamplingMode::TwoStepRandomWalk),
        _ => Err(PyValueError::new_err(format!(
            "unknown 2-hop sampling mode {raw:?}; expected 'degree_ordered_hub_suppressed' or 'two_step_random_walk'"
        ))),
    }
}

/// Extract disjoint, sorted, self-loop-free unweighted 1-hop and pure 2-hop ``CsrGraph`` instances.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (graph), text_signature = "(graph)")]
fn extract_1hop_2hop_csr(py: Python<'_>, graph: &PyCsrGraph) -> PyResult<(PyCsrGraph, PyCsrGraph)> {
    let g = graph.inner();
    let (adj1, adj2) = py
        .detach(|| nb::extract_1hop_2hop_csr(&*g))
        .map_err(core_error_to_py)?;
    Ok((
        PyCsrGraph::from_csr_graph(adj1),
        PyCsrGraph::from_csr_graph(adj2),
    ))
}

/// Extract canonical undirected 1-hop and hub-safe capped 2-hop vertex pairs ``Sparse12HopPairs``.
#[pyfunction]
#[pyo3(
    signature = (graph, max_2hop_per_node = StrictUsize(32), seed = StrictU64(42), *, mode = "degree_ordered_hub_suppressed", max_bridge_degree = None),
    text_signature = "(graph, max_2hop_per_node=32, seed=42, *, mode='degree_ordered_hub_suppressed', max_bridge_degree=None)"
)]
fn extract_sparse_1hop_2hop_pairs(
    py: Python<'_>,
    graph: &PyCsrGraph,
    max_2hop_per_node: StrictUsize,
    seed: StrictU64,
    mode: &str,
    max_bridge_degree: Option<StrictUsize>,
) -> PyResult<PySparse12HopPairs> {
    let sampling_mode = parse_two_hop_mode(mode)?;
    let g = graph.inner();
    let cap = max_2hop_per_node.0;
    let bridge_cap = max_bridge_degree.map(|v| v.0);
    let seed_val = seed.0;
    let inner = py
        .detach(|| {
            nb::extract_sparse_1hop_2hop_pairs_with_mode(
                &*g,
                cap,
                sampling_mode,
                bridge_cap,
                seed_val,
            )
        })
        .map_err(core_error_to_py)?;
    Ok(PySparse12HopPairs::from_inner(inner))
}

/// Build a pure 1+2-hop `KnnGraph` with `{1.0, 2.0}` distances, `k2_deg`, and summary `stats`.
#[pyfunction]
#[pyo3(signature = (graph), text_signature = "(graph)")]
fn build_pure_2hop_knn<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
) -> PyResult<(PyKnnGraph, Bound<'py, PyArray1<u32>>, Py<PyDict>)> {
    let g = graph.inner();
    let (knn, k2_deg, stats) = py
        .detach(|| nb::build_pure_2hop_knn(&*g))
        .map_err(core_error_to_py)?;
    let stats_dict = knn_stats_to_py_dict(py, &stats)?;
    Ok((
        PyKnnGraph::from_inner(knn),
        PyArray1::from_vec(py, k2_deg),
        stats_dict,
    ))
}

/// Build a hub-safe capped 1+2-hop `KnnGraph` with `{1.0, 2.0}` distances, `k2_deg`, and summary `stats`.
#[pyfunction]
#[pyo3(
    signature = (graph, max_2hop_per_node = StrictUsize(32), *, seed = StrictU64(42)),
    text_signature = "(graph, max_2hop_per_node=32, *, seed=42)"
)]
fn build_capped_12hop_knn<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
    max_2hop_per_node: StrictUsize,
    seed: StrictU64,
) -> PyResult<(PyKnnGraph, Bound<'py, PyArray1<u32>>, Py<PyDict>)> {
    let g = graph.inner();
    let cap = max_2hop_per_node.0;
    let seed_val = seed.0;
    let (knn, k2_deg, stats) = py
        .detach(|| nb::build_capped_12hop_knn(&*g, cap, seed_val))
        .map_err(core_error_to_py)?;
    let stats_dict = knn_stats_to_py_dict(py, &stats)?;
    Ok((
        PyKnnGraph::from_inner(knn),
        PyArray1::from_vec(py, k2_deg),
        stats_dict,
    ))
}

/// Compute exact distinct 1-hop + 2-hop neighborhood sizes for every vertex in `graph`.
#[pyfunction]
#[pyo3(signature = (graph), text_signature = "(graph)")]
fn compute_2hop_neighborhood_sizes<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
) -> Bound<'py, PyArray1<u32>> {
    let g = graph.inner();
    let sizes = py.detach(|| nb::compute_2hop_neighborhood_sizes(&*g));
    PyArray1::from_vec(py, sizes)
}

/// Compute per-node 2-hop adaptive perplexities `clip(c * k2_i, min_perp, upper)`.
#[pyfunction]
#[pyo3(
    signature = (graph, *, c = StrictF64(0.9), min_perplexity = StrictF64(5.0), max_perplexity = StrictF64(1200.0)),
    text_signature = "(graph, *, c=0.9, min_perplexity=5.0, max_perplexity=1200.0)"
)]
fn compute_nodewise_2hop_perplexities<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
    c: StrictF64,
    min_perplexity: StrictF64,
    max_perplexity: StrictF64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let g = graph.inner();
    let out = py
        .detach(|| {
            nb::compute_nodewise_2hop_perplexities(&*g, c.0, min_perplexity.0, max_perplexity.0)
        })
        .map_err(core_error_to_py)?;
    Ok(PyArray1::from_vec(py, out))
}

/// Compute scalar median 2-hop adaptive perplexity `clip(c * median(k2), min_perp, upper)`.
#[pyfunction]
#[pyo3(
    signature = (graph, *, c = StrictF64(1.5), min_perplexity = StrictF64(15.0), max_perplexity = StrictF64(1200.0)),
    text_signature = "(graph, *, c=1.5, min_perplexity=15.0, max_perplexity=1200.0)"
)]
fn compute_median_2hop_perplexity(
    py: Python<'_>,
    graph: &PyCsrGraph,
    c: StrictF64,
    min_perplexity: StrictF64,
    max_perplexity: StrictF64,
) -> PyResult<f64> {
    let g = graph.inner();
    py.detach(|| nb::compute_median_2hop_perplexity(&*g, c.0, min_perplexity.0, max_perplexity.0))
        .map_err(core_error_to_py)
}

/// Build a hybrid 1+2-hop + 2-stage spectral rerank `KnnGraph` and summary `stats`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (graph, coords, perplexities, *, k_factor = StrictF64(3.0), k = None, margin_gap = StrictF64(0.1), coarse_dim = StrictUsize(10), rerank_factor = StrictUsize(16), max_row_k = None),
    text_signature = "(graph, coords, perplexities, *, k_factor=3.0, k=None, margin_gap=0.1, coarse_dim=10, rerank_factor=16, max_row_k=None)"
)]
fn build_hybrid_adaptive_knn(
    py: Python<'_>,
    graph: &PyCsrGraph,
    coords: &Bound<'_, PyAny>,
    perplexities: &Bound<'_, PyAny>,
    k_factor: StrictF64,
    k: Option<StrictUsize>,
    margin_gap: StrictF64,
    coarse_dim: StrictUsize,
    rerank_factor: StrictUsize,
    max_row_k: Option<StrictUsize>,
) -> PyResult<(PyKnnGraph, Py<PyDict>)> {
    let g = graph.inner();
    let n = g.num_nodes();
    let (coords_flat, nrows, dim) = extract_coords_any(coords, "coords")?;
    if nrows != n {
        return Err(PyValueError::new_err(format!(
            "coords row count {nrows} does not match graph.num_nodes ({n})"
        )));
    }
    let perps_vec = extract_perplexities(perplexities, n)?;
    let cfg = HybridKnnConfig {
        k_factor: k_factor.0,
        k_override: k.map(|v| v.0),
        margin_gap: margin_gap.0,
        coarse_dim: coarse_dim.0,
        rerank_factor: rerank_factor.0,
        max_row_k: max_row_k.map(|v| v.0),
    };
    let (knn, stats) = py
        .detach(|| nb::build_hybrid_adaptive_knn(&*g, &coords_flat, dim, &perps_vec, &cfg))
        .map_err(core_error_to_py)?;
    let stats_dict = knn_stats_to_py_dict(py, &stats)?;
    Ok((PyKnnGraph::from_inner(knn), stats_dict))
}

/// Build an unweighted hop-shell `KnnGraph` via Partial BFS with deterministic tie-breaking.
#[pyfunction]
#[pyo3(
    signature = (graph, k = StrictUsize(30), *, seed = StrictU64(42)),
    text_signature = "(graph, k=30, *, seed=42)"
)]
fn partial_bfs_knn(
    py: Python<'_>,
    graph: &PyCsrGraph,
    k: StrictUsize,
    seed: StrictU64,
) -> PyResult<PyKnnGraph> {
    let g = graph.inner();
    let knn = py
        .detach(|| nb::partial_bfs_knn(&*g, k.0, seed.0))
        .map_err(core_error_to_py)?;
    Ok(PyKnnGraph::from_inner(knn))
}

/// Construct a 2-stage coarse `StaticKdTree` + full-dimensional Euclidean reranked `KnnGraph`.
#[pyfunction]
#[pyo3(
    signature = (coords, k = StrictUsize(15), *, coarse_dim = StrictUsize(10), rerank_factor = StrictUsize(16)),
    text_signature = "(coords, k=15, *, coarse_dim=10, rerank_factor=16)"
)]
fn build_rerank_knn(
    py: Python<'_>,
    coords: &Bound<'_, PyAny>,
    k: StrictUsize,
    coarse_dim: StrictUsize,
    rerank_factor: StrictUsize,
) -> PyResult<PyKnnGraph> {
    let (flat, n_nodes, dim) = extract_coords_any(coords, "coords")?;
    let knn = py
        .detach(|| nb::build_rerank_knn(&flat, n_nodes, dim, k.0, coarse_dim.0, rerank_factor.0))
        .map_err(core_error_to_py)?;
    Ok(PyKnnGraph::from_inner(knn))
}

/// Build a sorted symmetric CSR `KnnGraph` from 1-hop and 2-hop pair distances enforcing `d1 < tier2_margin * d2`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (num_nodes, u1, v1, d1, u2, v2, d2, *, tier2_margin = StrictF64(1.1)),
    text_signature = "(num_nodes, u1, v1, d1, u2, v2, d2, *, tier2_margin=1.1)"
)]
fn build_gumap_12hop_priority_csr(
    py: Python<'_>,
    num_nodes: StrictUsize,
    u1: &Bound<'_, PyAny>,
    v1: &Bound<'_, PyAny>,
    d1: &Bound<'_, PyAny>,
    u2: &Bound<'_, PyAny>,
    v2: &Bound<'_, PyAny>,
    d2: &Bound<'_, PyAny>,
    tier2_margin: StrictF64,
) -> PyResult<PyKnnGraph> {
    let u1_v = extract_u32_1d(u1, "u1")?;
    let v1_v = extract_u32_1d(v1, "v1")?;
    let d1_v = extract_f64_1d(d1, "d1")?;
    let u2_v = extract_u32_1d(u2, "u2")?;
    let v2_v = extract_u32_1d(v2, "v2")?;
    let d2_v = extract_f64_1d(d2, "d2")?;
    let margin = tier2_margin.0;
    let knn = py
        .detach(|| {
            nb::build_gumap_12hop_priority_csr(
                num_nodes.0,
                &u1_v,
                &v1_v,
                &d1_v,
                &u2_v,
                &v2_v,
                &d2_v,
                margin,
            )
        })
        .map_err(core_error_to_py)?;
    Ok(PyKnnGraph::from_inner(knn))
}

type DistanceFloorReturn<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    f64,
);

/// Apply relative distance floor `max(eps_dist, min_ratio * median_pos(d1))` to `(d1, d2, d3)`.
///
/// Returns `(d1_clamped, d2_clamped, d3_clamped, d_floor)`.
#[pyfunction]
#[pyo3(
    signature = (d1, d2, d3, *, eps_dist = StrictF64(1e-6), min_ratio = StrictF64(0.5)),
    text_signature = "(d1, d2, d3, *, eps_dist=1e-6, min_ratio=0.5)"
)]
fn apply_relative_distance_floor<'py>(
    py: Python<'py>,
    d1: &Bound<'_, PyAny>,
    d2: &Bound<'_, PyAny>,
    d3: &Bound<'_, PyAny>,
    eps_dist: StrictF64,
    min_ratio: StrictF64,
) -> PyResult<DistanceFloorReturn<'py>> {
    let mut d1_v = extract_f64_1d(d1, "d1")?;
    let mut d2_v = extract_f64_1d(d2, "d2")?;
    let mut d3_v = extract_f64_1d(d3, "d3")?;
    let eps = eps_dist.0;
    let ratio = min_ratio.0;
    let d_floor = py
        .detach(|| nb::apply_relative_distance_floor(&mut d1_v, &mut d2_v, &mut d3_v, eps, ratio))
        .map_err(core_error_to_py)?;
    Ok((
        PyArray1::from_vec(py, d1_v),
        PyArray1::from_vec(py, d2_v),
        PyArray1::from_vec(py, d3_v),
        d_floor,
    ))
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_1hop_2hop_csr, m)?)?;
    m.add_function(wrap_pyfunction!(extract_sparse_1hop_2hop_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(build_pure_2hop_knn, m)?)?;
    m.add_function(wrap_pyfunction!(build_capped_12hop_knn, m)?)?;
    m.add_function(wrap_pyfunction!(build_rerank_knn, m)?)?;
    m.add_function(wrap_pyfunction!(compute_2hop_neighborhood_sizes, m)?)?;
    m.add_function(wrap_pyfunction!(compute_nodewise_2hop_perplexities, m)?)?;
    m.add_function(wrap_pyfunction!(compute_median_2hop_perplexity, m)?)?;
    m.add_function(wrap_pyfunction!(build_hybrid_adaptive_knn, m)?)?;
    m.add_function(wrap_pyfunction!(partial_bfs_knn, m)?)?;
    m.add_function(wrap_pyfunction!(build_gumap_12hop_priority_csr, m)?)?;
    m.add_function(wrap_pyfunction!(apply_relative_distance_floor, m)?)?;
    Ok(())
}
