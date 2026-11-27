//! Partition agreement, 2D density consistency, Sprawlter-NN, and NOR metric bindings.

use egraph::metrics as egraph_metrics;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::array_extract::extract_f64_1d;
use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{StrictF64, StrictUsize, core_error_to_py, extract_u32_1d};
use crate::layout_buffers::{Layout2dInner, PyLayout2d, dispatch_layout2d, layout_f32_to_f64};
use crate::solver_common::vec_f64_to_f32_checked;

use super::cluster_baselines_nd;
use super::nd_manifold_helpers::{nor_metrics_to_py_dict, partition_agreement_to_py_dict};

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

/// Compute 3x3 Moore-smoothed 2D Histogram Density Measure (`2D-HDM`) in `[0.0, 1.0]`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, labels, *, grid_size = StrictUsize(64)),
    text_signature = "(layout, labels, *, grid_size=64)"
)]
fn histogram_density_measure(
    py: Python<'_>,
    layout: &PyLayout2d,
    labels: &Bound<'_, PyAny>,
    grid_size: StrictUsize,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let gs = grid_size.0;
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::histogram_density_measure(&soa, &lbl, gs)
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

/// Compute Sprawlter-NN composite overlap clutter measure using uniform per-node radii.
///
/// Equivalent to `sprawlter_nn(layout, radii)`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (layout, radii), text_signature = "(layout, radii)")]
fn sprawlter_nn_radii(
    py: Python<'_>,
    layout: &PyLayout2d,
    radii: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    sprawlter_nn(py, layout, radii, None)
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
            py.detach(|| match g_opt.as_ref() {
                Some(g) => egraph_metrics::nor_metrics(&**g, &o, &a, &hw_f64, &hh_f64),
                None => egraph_metrics::nor_metrics_without_graph(&o, &a, &hw_f64, &hh_f64),
            })
            .map_err(core_error_to_py)?
        }
        (Layout2dInner::F32(orig), Layout2dInner::F32(adj)) => {
            let hw_f32 = vec_f64_to_f32_checked(&hw_f64, "half_widths")?;
            let hh_f32 = vec_f64_to_f32_checked(&hh_f64, "half_heights")?;
            let o = orig.to_layout2d();
            let a = adj.to_layout2d();
            py.detach(|| match g_opt.as_ref() {
                Some(g) => egraph_metrics::nor_metrics(&**g, &o, &a, &hw_f32, &hh_f32),
                None => egraph_metrics::nor_metrics_without_graph(&o, &a, &hw_f32, &hh_f32),
            })
            .map_err(core_error_to_py)?
        }
        (Layout2dInner::F64(orig), Layout2dInner::F32(adj)) => {
            let o = orig.to_layout2d();
            let a = layout_f32_to_f64(&adj.to_layout2d());
            py.detach(|| match g_opt.as_ref() {
                Some(g) => egraph_metrics::nor_metrics(&**g, &o, &a, &hw_f64, &hh_f64),
                None => egraph_metrics::nor_metrics_without_graph(&o, &a, &hw_f64, &hh_f64),
            })
            .map_err(core_error_to_py)?
        }
        (Layout2dInner::F32(orig), Layout2dInner::F64(adj)) => {
            let o = layout_f32_to_f64(&orig.to_layout2d());
            let a = adj.to_layout2d();
            py.detach(|| match g_opt.as_ref() {
                Some(g) => egraph_metrics::nor_metrics(&**g, &o, &a, &hw_f64, &hh_f64),
                None => egraph_metrics::nor_metrics_without_graph(&o, &a, &hw_f64, &hh_f64),
            })
            .map_err(core_error_to_py)?
        }
    };
    nor_metrics_to_py_dict(py, nor)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    cluster_baselines_nd::register(m)?;
    m.add_function(wrap_pyfunction!(partition_agreement, m)?)?;
    m.add_function(wrap_pyfunction!(class_density_measure, m)?)?;
    m.add_function(wrap_pyfunction!(histogram_density_measure, m)?)?;
    m.add_function(wrap_pyfunction!(sprawlter_nn, m)?)?;
    m.add_function(wrap_pyfunction!(sprawlter_nn_radii, m)?)?;
    m.add_function(wrap_pyfunction!(nor_metrics, m)?)?;
    Ok(())
}
