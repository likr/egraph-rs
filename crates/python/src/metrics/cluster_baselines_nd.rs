//! Cluster separation, cluster quality, and internal validity metrics across 2D, N-D, and manifolds.

use egraph::metrics as egraph_metrics;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::ffi_convert::{StrictF64, StrictU64, StrictUsize, extract_u32_1d};

use super::nd_manifold_helpers::{
    MetricLayoutInput, dispatch_owned_metric_layout, partition_agreement_to_py_dict,
};

/// Compute K-Means++ Cluster Quality (`CQ-kmeans`) partition agreement on `layout`.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, labels, *, n_init = StrictUsize(5), max_iterations = StrictUsize(100), seed = StrictU64(42)),
    text_signature = "(layout, labels, *, n_init=5, max_iterations=100, seed=42)"
)]
fn cq_kmeans(
    py: Python<'_>,
    layout: MetricLayoutInput<'_>,
    labels: &Bound<'_, PyAny>,
    n_init: StrictUsize,
    max_iterations: StrictUsize,
    seed: StrictU64,
) -> PyResult<Py<PyDict>> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let owned = layout.extract_owned()?;
    let ni = n_init.0;
    let mi = max_iterations.0;
    let s = seed.0;
    let pa = dispatch_owned_metric_layout(
        py,
        owned,
        |l| egraph_metrics::cq_kmeans(l, &lbl, ni, mi, s),
        |l| egraph_metrics::cq_kmeans(l, &lbl, ni, mi, s),
        |l| egraph_metrics::cq_kmeans_nd(l, &lbl, ni, mi, s),
        |l| egraph_metrics::cq_kmeans_nd(l, &lbl, ni, mi, s),
        |l| egraph_metrics::cq_kmeans_manifold(l, &lbl, ni, mi, s),
        |l| egraph_metrics::cq_kmeans_manifold(l, &lbl, ni, mi, s),
    )?;
    partition_agreement_to_py_dict(py, pa)
}

/// Compute average-linkage HAC Cluster Quality (`CQ-HAC`) partition agreement on `layout`.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, labels),
    text_signature = "(layout, labels)"
)]
fn cq_hac(
    py: Python<'_>,
    layout: MetricLayoutInput<'_>,
    labels: &Bound<'_, PyAny>,
) -> PyResult<Py<PyDict>> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let owned = layout.extract_owned()?;
    let pa = dispatch_owned_metric_layout(
        py,
        owned,
        |l| egraph_metrics::cq_hac(l, &lbl),
        |l| egraph_metrics::cq_hac(l, &lbl),
        |l| egraph_metrics::cq_hac_nd(l, &lbl),
        |l| egraph_metrics::cq_hac_nd(l, &lbl),
        |l| egraph_metrics::cq_hac_manifold(l, &lbl),
        |l| egraph_metrics::cq_hac_manifold(l, &lbl),
    )?;
    partition_agreement_to_py_dict(py, pa)
}

/// Compute Distance Consistency (`DSC`) in `[0.0, 1.0]` for `layout` and discrete `labels`.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (layout, labels), text_signature = "(layout, labels)")]
fn distance_consistency(
    py: Python<'_>,
    layout: MetricLayoutInput<'_>,
    labels: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let owned = layout.extract_owned()?;
    dispatch_owned_metric_layout(
        py,
        owned,
        |l| egraph_metrics::distance_consistency(l, &lbl),
        |l| egraph_metrics::distance_consistency(l, &lbl),
        |l| egraph_metrics::distance_consistency_nd(l, &lbl),
        |l| egraph_metrics::distance_consistency_nd(l, &lbl),
        |l| egraph_metrics::distance_consistency_manifold(l, &lbl),
        |l| egraph_metrics::distance_consistency_manifold(l, &lbl),
    )
}

/// Compute Distribution Consistency (`DC`) in `[0.0, 1.0]` on a `grid_size` partition.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, labels, grid_size = StrictUsize(20)),
    text_signature = "(layout, labels, grid_size=20)"
)]
fn distribution_consistency(
    py: Python<'_>,
    layout: MetricLayoutInput<'_>,
    labels: &Bound<'_, PyAny>,
    grid_size: StrictUsize,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let owned = layout.extract_owned()?;
    let gs = grid_size.0;
    dispatch_owned_metric_layout(
        py,
        owned,
        |l| egraph_metrics::distribution_consistency(l, &lbl, gs),
        |l| egraph_metrics::distribution_consistency(l, &lbl, gs),
        |l| egraph_metrics::distribution_consistency_nd(l, &lbl, gs),
        |l| egraph_metrics::distribution_consistency_nd(l, &lbl, gs),
        |l| egraph_metrics::distribution_consistency_manifold(l, &lbl, gs),
        |l| egraph_metrics::distribution_consistency_manifold(l, &lbl, gs),
    )
}

/// Compute SepMe $k$-NN Graph DIR CPT (`sepme_knng`) in `[0.0, 1.0]`.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, labels, k = StrictUsize(2)),
    text_signature = "(layout, labels, k=2)"
)]
fn sepme_knng(
    py: Python<'_>,
    layout: MetricLayoutInput<'_>,
    labels: &Bound<'_, PyAny>,
    k: StrictUsize,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let owned = layout.extract_owned()?;
    let kv = k.0;
    dispatch_owned_metric_layout(
        py,
        owned,
        |l| egraph_metrics::sepme_knng(l, &lbl, kv),
        |l| egraph_metrics::sepme_knng(l, &lbl, kv),
        |l| egraph_metrics::sepme_knng_nd(l, &lbl, kv),
        |l| egraph_metrics::sepme_knng_nd(l, &lbl, kv),
        |l| egraph_metrics::sepme_knng_manifold(l, &lbl, kv),
        |l| egraph_metrics::sepme_knng_manifold(l, &lbl, kv),
    )
}

/// Compute SepMe Relative Neighborhood Graph DIR CPT (`sepme_rng`) in `[0.0, 1.0]`.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (layout, labels), text_signature = "(layout, labels)")]
fn sepme_rng(
    py: Python<'_>,
    layout: MetricLayoutInput<'_>,
    labels: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let owned = layout.extract_owned()?;
    dispatch_owned_metric_layout(
        py,
        owned,
        |l| egraph_metrics::sepme_rng(l, &lbl),
        |l| egraph_metrics::sepme_rng(l, &lbl),
        |l| egraph_metrics::sepme_rng_nd(l, &lbl),
        |l| egraph_metrics::sepme_rng_nd(l, &lbl),
        |l| egraph_metrics::sepme_rng_manifold(l, &lbl),
        |l| egraph_metrics::sepme_rng_manifold(l, &lbl),
    )
}

/// Compute SepMe Gabriel Graph DIR CPT (`sepme_gab`) in `[0.0, 1.0]`.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (layout, labels), text_signature = "(layout, labels)")]
fn sepme_gab(
    py: Python<'_>,
    layout: MetricLayoutInput<'_>,
    labels: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let owned = layout.extract_owned()?;
    dispatch_owned_metric_layout(
        py,
        owned,
        |l| egraph_metrics::sepme_gab(l, &lbl),
        |l| egraph_metrics::sepme_gab(l, &lbl),
        |l| egraph_metrics::sepme_gab_nd(l, &lbl),
        |l| egraph_metrics::sepme_gab_nd(l, &lbl),
        |l| egraph_metrics::sepme_gab_manifold(l, &lbl),
        |l| egraph_metrics::sepme_gab_manifold(l, &lbl),
    )
}

/// Compute SepMe $\gamma$-Observable Neighbor Graph DIR CPT (`sepme_gong`) in `[0.0, 1.0]`.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, labels, gamma = StrictF64(0.35)),
    text_signature = "(layout, labels, gamma=0.35)"
)]
fn sepme_gong(
    py: Python<'_>,
    layout: MetricLayoutInput<'_>,
    labels: &Bound<'_, PyAny>,
    gamma: StrictF64,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let owned = layout.extract_owned()?;
    let g = gamma.0;
    dispatch_owned_metric_layout(
        py,
        owned,
        |l| egraph_metrics::sepme_gong(l, &lbl, g),
        |l| egraph_metrics::sepme_gong(l, &lbl, g),
        |l| egraph_metrics::sepme_gong_nd(l, &lbl, g),
        |l| egraph_metrics::sepme_gong_nd(l, &lbl, g),
        |l| egraph_metrics::sepme_gong_manifold(l, &lbl, g),
        |l| egraph_metrics::sepme_gong_manifold(l, &lbl, g),
    )
}

/// Compute Neighborhood Hit (`M_NH`) in `[0.0, 1.0]` over each node's `k` nearest neighbors.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, labels, k = StrictUsize(7)),
    text_signature = "(layout, labels, k=7)"
)]
fn neighborhood_hit(
    py: Python<'_>,
    layout: MetricLayoutInput<'_>,
    labels: &Bound<'_, PyAny>,
    k: StrictUsize,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let owned = layout.extract_owned()?;
    let kv = k.0;
    dispatch_owned_metric_layout(
        py,
        owned,
        |l| egraph_metrics::neighborhood_hit(l, &lbl, kv),
        |l| egraph_metrics::neighborhood_hit(l, &lbl, kv),
        |l| egraph_metrics::neighborhood_hit_nd(l, &lbl, kv),
        |l| egraph_metrics::neighborhood_hit_nd(l, &lbl, kv),
        |l| egraph_metrics::neighborhood_hit_manifold(l, &lbl, kv),
        |l| egraph_metrics::neighborhood_hit_manifold(l, &lbl, kv),
    )
}

/// Compute mean Silhouette coefficient in `[-1.0, 1.0]` for `layout` and `labels`.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (layout, labels), text_signature = "(layout, labels)")]
fn silhouette_score(
    py: Python<'_>,
    layout: MetricLayoutInput<'_>,
    labels: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let owned = layout.extract_owned()?;
    dispatch_owned_metric_layout(
        py,
        owned,
        |l| egraph_metrics::silhouette_score(l, &lbl),
        |l| egraph_metrics::silhouette_score(l, &lbl),
        |l| egraph_metrics::silhouette_score_nd(l, &lbl),
        |l| egraph_metrics::silhouette_score_nd(l, &lbl),
        |l| egraph_metrics::silhouette_score_manifold(l, &lbl),
        |l| egraph_metrics::silhouette_score_manifold(l, &lbl),
    )
}

/// Compute Calinski-Harabasz Index (`>= 0.0`) for `layout` and `labels`.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (layout, labels), text_signature = "(layout, labels)")]
fn calinski_harabasz_score(
    py: Python<'_>,
    layout: MetricLayoutInput<'_>,
    labels: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let owned = layout.extract_owned()?;
    dispatch_owned_metric_layout(
        py,
        owned,
        |l| egraph_metrics::calinski_harabasz_score(l, &lbl),
        |l| egraph_metrics::calinski_harabasz_score(l, &lbl),
        |l| egraph_metrics::calinski_harabasz_score_nd(l, &lbl),
        |l| egraph_metrics::calinski_harabasz_score_nd(l, &lbl),
        |l| egraph_metrics::calinski_harabasz_score_manifold(l, &lbl),
        |l| egraph_metrics::calinski_harabasz_score_manifold(l, &lbl),
    )
}

/// Compute Davies-Bouldin Index (`>= 0.0`, lower is better) for `layout` and `labels`.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (layout, labels), text_signature = "(layout, labels)")]
fn davies_bouldin_score(
    py: Python<'_>,
    layout: MetricLayoutInput<'_>,
    labels: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let owned = layout.extract_owned()?;
    dispatch_owned_metric_layout(
        py,
        owned,
        |l| egraph_metrics::davies_bouldin_score(l, &lbl),
        |l| egraph_metrics::davies_bouldin_score(l, &lbl),
        |l| egraph_metrics::davies_bouldin_score_nd(l, &lbl),
        |l| egraph_metrics::davies_bouldin_score_nd(l, &lbl),
        |l| egraph_metrics::davies_bouldin_score_manifold(l, &lbl),
        |l| egraph_metrics::davies_bouldin_score_manifold(l, &lbl),
    )
}

/// Compute Pairwise Distance-Overlap Error (`PDOE`) in `[0.0, 1.0]` using `num_bins` histogram bins.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, labels, num_bins = StrictUsize(100)),
    text_signature = "(layout, labels, num_bins=100)"
)]
fn pairwise_distance_overlap_error(
    py: Python<'_>,
    layout: MetricLayoutInput<'_>,
    labels: &Bound<'_, PyAny>,
    num_bins: StrictUsize,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let owned = layout.extract_owned()?;
    let bins = num_bins.0;
    dispatch_owned_metric_layout(
        py,
        owned,
        |l| egraph_metrics::pairwise_distance_overlap_error(l, &lbl, bins),
        |l| egraph_metrics::pairwise_distance_overlap_error(l, &lbl, bins),
        |l| egraph_metrics::pairwise_distance_overlap_error_nd(l, &lbl, bins),
        |l| egraph_metrics::pairwise_distance_overlap_error_nd(l, &lbl, bins),
        |l| egraph_metrics::pairwise_distance_overlap_error_manifold(l, &lbl, bins),
        |l| egraph_metrics::pairwise_distance_overlap_error_manifold(l, &lbl, bins),
    )
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cq_kmeans, m)?)?;
    m.add_function(wrap_pyfunction!(cq_hac, m)?)?;
    m.add_function(wrap_pyfunction!(distance_consistency, m)?)?;
    m.add_function(wrap_pyfunction!(distribution_consistency, m)?)?;
    m.add_function(wrap_pyfunction!(sepme_knng, m)?)?;
    m.add_function(wrap_pyfunction!(sepme_rng, m)?)?;
    m.add_function(wrap_pyfunction!(sepme_gab, m)?)?;
    m.add_function(wrap_pyfunction!(sepme_gong, m)?)?;
    m.add_function(wrap_pyfunction!(neighborhood_hit, m)?)?;
    m.add_function(wrap_pyfunction!(silhouette_score, m)?)?;
    m.add_function(wrap_pyfunction!(calinski_harabasz_score, m)?)?;
    m.add_function(wrap_pyfunction!(davies_bouldin_score, m)?)?;
    m.add_function(wrap_pyfunction!(pairwise_distance_overlap_error, m)?)?;
    Ok(())
}
