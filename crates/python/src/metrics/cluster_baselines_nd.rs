//! Cluster separation, cluster quality, and internal validity metrics across 2D, 3D, N-D, and manifolds.

use egraph::metrics as egraph_metrics;
use pyo3::prelude::*;

use crate::ffi_convert::{StrictF64, StrictU64, StrictUsize, extract_u32_1d};

use super::cluster_result_types::PyPartitionAgreement;
use super::nd_manifold_helpers::{MetricLayoutInput, dispatch_owned_metric};

/// Compute K-Means++ Cluster Quality (`CQ-kmeans`) partition agreement on `layout`.
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
) -> PyResult<PyPartitionAgreement> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let owned = layout.extract_owned()?;
    let (ni, mi, s) = (n_init.0, max_iterations.0, seed.0);
    let inner =
        dispatch_owned_metric!(py, owned, |l| egraph_metrics::cq_kmeans(l, &lbl, ni, mi, s))?;
    Ok(PyPartitionAgreement { inner })
}

/// Compute average-linkage HAC Cluster Quality (`CQ-HAC`) partition agreement on `layout`.
#[pyfunction]
#[pyo3(signature = (layout, labels), text_signature = "(layout, labels)")]
fn cq_hac(
    py: Python<'_>,
    layout: MetricLayoutInput<'_>,
    labels: &Bound<'_, PyAny>,
) -> PyResult<PyPartitionAgreement> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let owned = layout.extract_owned()?;
    let inner = dispatch_owned_metric!(py, owned, |l| egraph_metrics::cq_hac(l, &lbl))?;
    Ok(PyPartitionAgreement { inner })
}

/// Compute Distance Consistency (`DSC`) in `[0.0, 1.0]` for `layout` and discrete `labels`.
#[pyfunction]
#[pyo3(signature = (layout, labels), text_signature = "(layout, labels)")]
fn distance_consistency(
    py: Python<'_>,
    layout: MetricLayoutInput<'_>,
    labels: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let owned = layout.extract_owned()?;
    dispatch_owned_metric!(py, owned, |l| egraph_metrics::distance_consistency(l, &lbl))
}

/// Compute Distribution Consistency (`DC`) in `[0.0, 1.0]` on a `grid_size` partition.
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
    dispatch_owned_metric!(py, owned, |l| {
        egraph_metrics::distribution_consistency(l, &lbl, gs)
    })
}

/// Compute SepMe k-NN Graph DIR CPT (`sepme_knng`) in `[0.0, 1.0]`.
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
    dispatch_owned_metric!(py, owned, |l| egraph_metrics::sepme_knng(l, &lbl, kv))
}

/// Compute SepMe Relative Neighborhood Graph DIR CPT (`sepme_rng`) in `[0.0, 1.0]`.
#[pyfunction]
#[pyo3(signature = (layout, labels), text_signature = "(layout, labels)")]
fn sepme_rng(
    py: Python<'_>,
    layout: MetricLayoutInput<'_>,
    labels: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let owned = layout.extract_owned()?;
    dispatch_owned_metric!(py, owned, |l| egraph_metrics::sepme_rng(l, &lbl))
}

/// Compute SepMe Gabriel Graph DIR CPT (`sepme_gab`) in `[0.0, 1.0]`.
#[pyfunction]
#[pyo3(signature = (layout, labels), text_signature = "(layout, labels)")]
fn sepme_gab(
    py: Python<'_>,
    layout: MetricLayoutInput<'_>,
    labels: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let owned = layout.extract_owned()?;
    dispatch_owned_metric!(py, owned, |l| egraph_metrics::sepme_gab(l, &lbl))
}

/// Compute SepMe gamma-Observable Neighbor Graph DIR CPT (`sepme_gong`) in `[0.0, 1.0]`.
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
    dispatch_owned_metric!(py, owned, |l| egraph_metrics::sepme_gong(l, &lbl, g))
}

/// Compute Neighborhood Hit (`M_NH`) in `[0.0, 1.0]` over each node's `k` nearest neighbors.
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
    dispatch_owned_metric!(py, owned, |l| egraph_metrics::neighborhood_hit(l, &lbl, kv))
}

/// Compute mean Silhouette coefficient in `[-1.0, 1.0]` for `layout` and `labels`.
#[pyfunction]
#[pyo3(signature = (layout, labels), text_signature = "(layout, labels)")]
fn silhouette_score(
    py: Python<'_>,
    layout: MetricLayoutInput<'_>,
    labels: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let owned = layout.extract_owned()?;
    dispatch_owned_metric!(py, owned, |l| egraph_metrics::silhouette_score(l, &lbl))
}

/// Compute Calinski-Harabasz Index (`>= 0.0`) for `layout` and `labels`.
#[pyfunction]
#[pyo3(signature = (layout, labels), text_signature = "(layout, labels)")]
fn calinski_harabasz_score(
    py: Python<'_>,
    layout: MetricLayoutInput<'_>,
    labels: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let owned = layout.extract_owned()?;
    dispatch_owned_metric!(py, owned, |l| {
        egraph_metrics::calinski_harabasz_score(l, &lbl)
    })
}

/// Compute Davies-Bouldin Index (`>= 0.0`, lower is better) for `layout` and `labels`.
#[pyfunction]
#[pyo3(signature = (layout, labels), text_signature = "(layout, labels)")]
fn davies_bouldin_score(
    py: Python<'_>,
    layout: MetricLayoutInput<'_>,
    labels: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let owned = layout.extract_owned()?;
    dispatch_owned_metric!(py, owned, |l| {
        egraph_metrics::davies_bouldin_score(l, &lbl)
    })
}

/// Compute Pairwise Distance-Overlap Error (`PDOE`) in `[0.0, 1.0]` using `num_bins` histogram bins.
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
    dispatch_owned_metric!(py, owned, |l| {
        egraph_metrics::pairwise_distance_overlap_error(l, &lbl, bins)
    })
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
