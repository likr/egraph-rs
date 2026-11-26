//! Cluster separation, cluster validity index, Sprawlter-NN, and NOR metric bindings.

use egraph::metrics as egraph_metrics;
use egraph_metrics::{NorMetrics, PartitionAgreement};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::array_extract::extract_f64_1d;
use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{StrictF64, StrictU64, StrictUsize, core_error_to_py, extract_u32_1d};
use crate::layout_buffers::{Layout2dInner, PyLayout2d, dispatch_layout2d, layout_f32_to_f64};
use crate::solver_common::vec_f64_to_f32_checked;

fn partition_agreement_to_py_dict(py: Python<'_>, pa: PartitionAgreement) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("ari", pa.ari)?;
    dict.set_item("ami", pa.ami)?;
    dict.set_item("homogeneity", pa.homogeneity)?;
    dict.set_item("completeness", pa.completeness)?;
    dict.set_item("v_measure", pa.v_measure)?;
    dict.set_item("fmi", pa.fmi)?;
    Ok(dict.unbind())
}

fn nor_metrics_to_py_dict(py: Python<'_>, nor: NorMetrics) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("oo_nni", nor.oo_nni)?;
    dict.set_item("sp_ch_a", nor.sp_ch_a)?;
    dict.set_item("gs_bb_iar", nor.gs_bb_iar)?;
    dict.set_item("nm_dm_imse", nor.nm_dm_imse)?;
    dict.set_item("el_rsd", nor.el_rsd)?;
    Ok(dict.unbind())
}

/// Compute partition agreement metrics (`ari`, `ami`, `homogeneity`, `completeness`, `v_measure`, `fmi`).
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (labels_true, labels_pred),
    text_signature = "(labels_true, labels_pred)"
)]
fn partition_agreement(
    py: Python<'_>,
    labels_true: &Bound<'_, PyAny>,
    labels_pred: &Bound<'_, PyAny>,
) -> PyResult<Py<PyDict>> {
    let lt = extract_u32_1d(labels_true, "labels_true")?;
    let lp = extract_u32_1d(labels_pred, "labels_pred")?;
    let pa = py
        .detach(|| egraph_metrics::partition_agreement(&lt, &lp))
        .map_err(core_error_to_py)?;
    partition_agreement_to_py_dict(py, pa)
}

/// Compute K-Means++ Cluster Quality (`CQ-kmeans`) partition agreement on `layout`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, community_labels, *, n_init = StrictUsize(5), max_iters = StrictUsize(100), seed = StrictU64(42)),
    text_signature = "(layout, community_labels, *, n_init=5, max_iters=100, seed=42)"
)]
fn cq_kmeans(
    py: Python<'_>,
    layout: &PyLayout2d,
    community_labels: &Bound<'_, PyAny>,
    n_init: StrictUsize,
    max_iters: StrictUsize,
    seed: StrictU64,
) -> PyResult<Py<PyDict>> {
    let labels = extract_u32_1d(community_labels, "community_labels")?;
    let ni = n_init.0;
    let mi = max_iters.0;
    let s = seed.0;
    let pa = dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::cq_kmeans(&soa, &labels, ni, mi, s)
    })?;
    partition_agreement_to_py_dict(py, pa)
}

/// Compute average-linkage HAC Cluster Quality (`CQ-HAC`) partition agreement on `layout`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, community_labels),
    text_signature = "(layout, community_labels)"
)]
fn cq_hac(
    py: Python<'_>,
    layout: &PyLayout2d,
    community_labels: &Bound<'_, PyAny>,
) -> PyResult<Py<PyDict>> {
    let labels = extract_u32_1d(community_labels, "community_labels")?;
    let pa = dispatch_layout2d!(py, layout, |soa| egraph_metrics::cq_hac(&soa, &labels))?;
    partition_agreement_to_py_dict(py, pa)
}

/// Compute Distance Consistency (`DSC`) in `[0.0, 1.0]` for `layout` and discrete `labels`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (layout, labels), text_signature = "(layout, labels)")]
fn distance_consistency(
    py: Python<'_>,
    layout: &PyLayout2d,
    labels: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::distance_consistency(&soa, &lbl)
    })
}

/// Compute Distribution Consistency (`DC`) in `[0.0, 1.0]` on a `grid_size x grid_size` grid.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, labels, grid_size = StrictUsize(20)),
    text_signature = "(layout, labels, grid_size=20)"
)]
fn distribution_consistency(
    py: Python<'_>,
    layout: &PyLayout2d,
    labels: &Bound<'_, PyAny>,
    grid_size: StrictUsize,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let gs = grid_size.0;
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::distribution_consistency(&soa, &lbl, gs)
    })
}

/// Compute Class Density Measure and 2D Histogram Density Measure `(cdm, hdm)` in `[0.0, 1.0]`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, labels, *, grid_size = StrictUsize(64), bandwidth = None),
    text_signature = "(layout, labels, *, grid_size=64, bandwidth=None)"
)]
fn class_density_measure(
    py: Python<'_>,
    layout: &PyLayout2d,
    labels: &Bound<'_, PyAny>,
    grid_size: StrictUsize,
    bandwidth: Option<StrictF64>,
) -> PyResult<(f64, f64)> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let gs = grid_size.0;
    let bw = bandwidth.map(|v| v.0);
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::class_density_measure(&soa, &lbl, gs, bw)
    })
}

/// Compute SepMe $k$-NN Graph DIR CPT (`sepme_knng`) in `[0.0, 1.0]`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, labels, k = StrictUsize(2)),
    text_signature = "(layout, labels, k=2)"
)]
fn sepme_knng(
    py: Python<'_>,
    layout: &PyLayout2d,
    labels: &Bound<'_, PyAny>,
    k: StrictUsize,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let kv = k.0;
    dispatch_layout2d!(py, layout, |soa| egraph_metrics::sepme_knng(&soa, &lbl, kv))
}

/// Compute SepMe Relative Neighborhood Graph DIR CPT (`sepme_rng`) in `[0.0, 1.0]`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (layout, labels), text_signature = "(layout, labels)")]
fn sepme_rng(py: Python<'_>, layout: &PyLayout2d, labels: &Bound<'_, PyAny>) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    dispatch_layout2d!(py, layout, |soa| egraph_metrics::sepme_rng(&soa, &lbl))
}

/// Compute SepMe Gabriel Graph DIR CPT (`sepme_gab`) in `[0.0, 1.0]`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (layout, labels), text_signature = "(layout, labels)")]
fn sepme_gab(py: Python<'_>, layout: &PyLayout2d, labels: &Bound<'_, PyAny>) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    dispatch_layout2d!(py, layout, |soa| egraph_metrics::sepme_gab(&soa, &lbl))
}

/// Compute SepMe $\gamma$-Observable Neighbor Graph DIR CPT (`sepme_gong`) in `[0.0, 1.0]`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, labels, gamma = StrictF64(0.35)),
    text_signature = "(layout, labels, gamma=0.35)"
)]
fn sepme_gong(
    py: Python<'_>,
    layout: &PyLayout2d,
    labels: &Bound<'_, PyAny>,
    gamma: StrictF64,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let g = gamma.0;
    dispatch_layout2d!(py, layout, |soa| egraph_metrics::sepme_gong(&soa, &lbl, g))
}

/// Compute Neighborhood Hit (`M_NH`) in `[0.0, 1.0]` over each node's `k` nearest neighbors.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, labels, k = StrictUsize(7)),
    text_signature = "(layout, labels, k=7)"
)]
fn neighborhood_hit(
    py: Python<'_>,
    layout: &PyLayout2d,
    labels: &Bound<'_, PyAny>,
    k: StrictUsize,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let kv = k.0;
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::neighborhood_hit(&soa, &lbl, kv)
    })
}

/// Compute mean Silhouette coefficient in `[-1.0, 1.0]` for `layout` and `labels`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (layout, labels), text_signature = "(layout, labels)")]
fn silhouette_score(
    py: Python<'_>,
    layout: &PyLayout2d,
    labels: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::silhouette_score(&soa, &lbl)
    })
}

/// Compute Calinski-Harabasz Index (`>= 0.0`) for `layout` and `labels`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (layout, labels), text_signature = "(layout, labels)")]
fn calinski_harabasz_score(
    py: Python<'_>,
    layout: &PyLayout2d,
    labels: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::calinski_harabasz_score(&soa, &lbl)
    })
}

/// Compute Davies-Bouldin Index (`>= 0.0`, lower is better) for `layout` and `labels`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (layout, labels), text_signature = "(layout, labels)")]
fn davies_bouldin_score(
    py: Python<'_>,
    layout: &PyLayout2d,
    labels: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::davies_bouldin_score(&soa, &lbl)
    })
}

/// Compute Pairwise Distance-Overlap Error (`PDOE`) in `[0.0, 1.0]` using `num_bins` histogram bins.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, labels, num_bins = StrictUsize(100)),
    text_signature = "(layout, labels, num_bins=100)"
)]
fn pairwise_distance_overlap_error(
    py: Python<'_>,
    layout: &PyLayout2d,
    labels: &Bound<'_, PyAny>,
    num_bins: StrictUsize,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let bins = num_bins.0;
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::pairwise_distance_overlap_error(&soa, &lbl, bins)
    })
}

/// Compute Sprawlter-NN composite overlap clutter measure (`>= 0.0`).
///
/// When `half_heights` is `None`, treats `half_widths` as uniform per-node radii (`sprawlter_nn_radii`).
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, half_widths, half_heights = None),
    text_signature = "(layout, half_widths, half_heights=None)"
)]
fn sprawlter_nn(
    py: Python<'_>,
    layout: &PyLayout2d,
    half_widths: &Bound<'_, PyAny>,
    half_heights: Option<&Bound<'_, PyAny>>,
) -> PyResult<f64> {
    let hw_f64 = extract_f64_1d(half_widths, "half_widths")?;
    let hh_opt = match half_heights {
        Some(obj) if !obj.is_none() => Some(extract_f64_1d(obj, "half_heights")?),
        _ => None,
    };
    match layout.inner() {
        Layout2dInner::F64(buf) => {
            let soa = buf.to_layout2d();
            py.detach(|| match hh_opt.as_deref() {
                Some(hh) => egraph_metrics::sprawlter_nn(&soa, &hw_f64, hh),
                None => egraph_metrics::sprawlter_nn_radii(&soa, &hw_f64),
            })
            .map_err(core_error_to_py)
        }
        Layout2dInner::F32(buf) => {
            let hw_f32 = vec_f64_to_f32_checked(&hw_f64, "half_widths")?;
            let hh_f32_opt = match hh_opt {
                Some(hh) => Some(vec_f64_to_f32_checked(&hh, "half_heights")?),
                None => None,
            };
            let soa = buf.to_layout2d();
            py.detach(|| match hh_f32_opt.as_deref() {
                Some(hh) => egraph_metrics::sprawlter_nn(&soa, &hw_f32, hh),
                None => egraph_metrics::sprawlter_nn_radii(&soa, &hw_f32),
            })
            .map_err(core_error_to_py)
        }
    }
}

/// Compute Chen et al. (2020) Node Overlap Removal (NOR) 5-metric evaluation summary.
///
/// Returns a dict with keys `"oo_nni"`, `"sp_ch_a"`, `"gs_bb_iar"`, `"nm_dm_imse"`, and
/// `"el_rsd"` (`el_rsd` is `0.0` when `graph` is `None`). Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (orig_layout, adjusted_layout, half_widths, half_heights, *, graph = None),
    text_signature = "(orig_layout, adjusted_layout, half_widths, half_heights, *, graph=None)"
)]
fn nor_metrics(
    py: Python<'_>,
    orig_layout: &PyLayout2d,
    adjusted_layout: &PyLayout2d,
    half_widths: &Bound<'_, PyAny>,
    half_heights: &Bound<'_, PyAny>,
    graph: Option<&PyCsrGraph>,
) -> PyResult<Py<PyDict>> {
    let hw_f64 = extract_f64_1d(half_widths, "half_widths")?;
    let hh_f64 = extract_f64_1d(half_heights, "half_heights")?;
    let g_opt = graph.map(PyCsrGraph::inner);
    let nor = match (orig_layout.inner(), adjusted_layout.inner()) {
        (Layout2dInner::F64(orig), Layout2dInner::F64(adj)) => {
            let o = orig.to_layout2d();
            let a = adj.to_layout2d();
            py.detach(|| match g_opt {
                Some(g) => egraph_metrics::nor_metrics(g, &o, &a, &hw_f64, &hh_f64),
                None => egraph_metrics::nor_metrics_without_graph(&o, &a, &hw_f64, &hh_f64),
            })
            .map_err(core_error_to_py)?
        }
        (Layout2dInner::F32(orig), Layout2dInner::F32(adj)) => {
            let hw_f32 = vec_f64_to_f32_checked(&hw_f64, "half_widths")?;
            let hh_f32 = vec_f64_to_f32_checked(&hh_f64, "half_heights")?;
            let o = orig.to_layout2d();
            let a = adj.to_layout2d();
            py.detach(|| match g_opt {
                Some(g) => egraph_metrics::nor_metrics(g, &o, &a, &hw_f32, &hh_f32),
                None => egraph_metrics::nor_metrics_without_graph(&o, &a, &hw_f32, &hh_f32),
            })
            .map_err(core_error_to_py)?
        }
        (Layout2dInner::F64(orig), Layout2dInner::F32(adj)) => {
            let o = orig.to_layout2d();
            let a = layout_f32_to_f64(&adj.to_layout2d());
            py.detach(|| match g_opt {
                Some(g) => egraph_metrics::nor_metrics(g, &o, &a, &hw_f64, &hh_f64),
                None => egraph_metrics::nor_metrics_without_graph(&o, &a, &hw_f64, &hh_f64),
            })
            .map_err(core_error_to_py)?
        }
        (Layout2dInner::F32(orig), Layout2dInner::F64(adj)) => {
            let o = layout_f32_to_f64(&orig.to_layout2d());
            let a = adj.to_layout2d();
            py.detach(|| match g_opt {
                Some(g) => egraph_metrics::nor_metrics(g, &o, &a, &hw_f64, &hh_f64),
                None => egraph_metrics::nor_metrics_without_graph(&o, &a, &hw_f64, &hh_f64),
            })
            .map_err(core_error_to_py)?
        }
    };
    nor_metrics_to_py_dict(py, nor)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(partition_agreement, m)?)?;
    m.add_function(wrap_pyfunction!(cq_kmeans, m)?)?;
    m.add_function(wrap_pyfunction!(cq_hac, m)?)?;
    m.add_function(wrap_pyfunction!(distance_consistency, m)?)?;
    m.add_function(wrap_pyfunction!(distribution_consistency, m)?)?;
    m.add_function(wrap_pyfunction!(class_density_measure, m)?)?;
    m.add_function(wrap_pyfunction!(sepme_knng, m)?)?;
    m.add_function(wrap_pyfunction!(sepme_rng, m)?)?;
    m.add_function(wrap_pyfunction!(sepme_gab, m)?)?;
    m.add_function(wrap_pyfunction!(sepme_gong, m)?)?;
    m.add_function(wrap_pyfunction!(neighborhood_hit, m)?)?;
    m.add_function(wrap_pyfunction!(silhouette_score, m)?)?;
    m.add_function(wrap_pyfunction!(calinski_harabasz_score, m)?)?;
    m.add_function(wrap_pyfunction!(davies_bouldin_score, m)?)?;
    m.add_function(wrap_pyfunction!(pairwise_distance_overlap_error, m)?)?;
    m.add_function(wrap_pyfunction!(sprawlter_nn, m)?)?;
    m.add_function(wrap_pyfunction!(nor_metrics, m)?)?;
    Ok(())
}
