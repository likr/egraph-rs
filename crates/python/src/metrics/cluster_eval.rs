//! Cluster-area faithfulness (`G-CAF`) and Adjusted Patch-NIG bindings.

use std::collections::BTreeMap;

use egraph::metrics as egraph_metrics;
use egraph_metrics::ClusterAreaConfig;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use super::cluster_eval_types::{PyAdjustedPatchNigDetail, PyClusterAreaReport};
use crate::ffi_convert::{
    StrictF64, StrictU64, StrictUsize, extract_non_negative_usize, extract_optional_weights_1d,
    extract_u32_1d, is_non_sequence_container,
};
use crate::layout_buffers::{PyLayout2d, dispatch_layout2d};

fn extract_positive_usize_seq(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<usize>> {
    if is_non_sequence_container(obj) {
        return Err(PyTypeError::new_err(format!(
            "{name} must be a sequence or iterable of positive integers"
        )));
    }
    let iter = obj.try_iter().map_err(|_| {
        PyTypeError::new_err(format!(
            "{name} must be a sequence or iterable of positive integers"
        ))
    })?;
    let mut out = Vec::new();
    for item_res in iter {
        let item = item_res?;
        let val = extract_non_negative_usize(&item, name)?;
        out.push(val);
    }
    Ok(out)
}

/// Compute single-scale Adjusted Patch Normalized Information Gain (`Adjusted Patch-NIG`).
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, labels, *, patch_radius = StrictUsize(8), canvas_size = StrictUsize(2048), pad_frac = StrictF64(0.04), seed = StrictU64(42)),
    text_signature = "(layout, labels, *, patch_radius=8, canvas_size=2048, pad_frac=0.04, seed=42)"
)]
fn adjusted_patch_nig(
    py: Python<'_>,
    layout: &PyLayout2d,
    labels: &Bound<'_, PyAny>,
    patch_radius: StrictUsize,
    canvas_size: StrictUsize,
    pad_frac: StrictF64,
    seed: StrictU64,
) -> PyResult<f64> {
    let lbls = extract_u32_1d(labels, "labels")?;
    let radius = patch_radius.0;
    let c_sz = canvas_size.0;
    let pad = pad_frac.0;
    let seed_val = seed.0;
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::adjusted_patch_nig(&soa, &lbls, radius, c_sz, pad, seed_val)
    })
}

/// Compute single-scale Adjusted Patch-NIG with entropy and patch occupancy diagnostics.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, labels, *, patch_radius = StrictUsize(8), canvas_size = StrictUsize(2048), pad_frac = StrictF64(0.04), seed = StrictU64(42)),
    text_signature = "(layout, labels, *, patch_radius=8, canvas_size=2048, pad_frac=0.04, seed=42)"
)]
fn adjusted_patch_nig_detail(
    py: Python<'_>,
    layout: &PyLayout2d,
    labels: &Bound<'_, PyAny>,
    patch_radius: StrictUsize,
    canvas_size: StrictUsize,
    pad_frac: StrictF64,
    seed: StrictU64,
) -> PyResult<PyAdjustedPatchNigDetail> {
    let lbls = extract_u32_1d(labels, "labels")?;
    let radius = patch_radius.0;
    let c_sz = canvas_size.0;
    let pad = pad_frac.0;
    let seed_val = seed.0;
    let detail = dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::adjusted_patch_nig_detail(&soa, &lbls, radius, c_sz, pad, seed_val)
    })?;
    Ok(PyAdjustedPatchNigDetail::from_detail(detail))
}

/// Compute multi-scale Adjusted Patch-NIG across `radii`, returning `{"patch_nig_r{r}": value}`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, labels, *, radii = None, canvas_size = StrictUsize(2048), pad_frac = StrictF64(0.04), seed = StrictU64(42)),
    text_signature = "(layout, labels, *, radii=(2, 8, 32), canvas_size=2048, pad_frac=0.04, seed=42)"
)]
fn adjusted_patch_nig_multiscale(
    py: Python<'_>,
    layout: &PyLayout2d,
    labels: &Bound<'_, PyAny>,
    radii: Option<&Bound<'_, PyAny>>,
    canvas_size: StrictUsize,
    pad_frac: StrictF64,
    seed: StrictU64,
) -> PyResult<BTreeMap<String, f64>> {
    let lbls = extract_u32_1d(labels, "labels")?;
    let radii_vec = match radii {
        Some(obj) if !obj.is_none() => extract_positive_usize_seq(obj, "radii")?,
        _ => vec![2, 8, 32],
    };
    let c_sz = canvas_size.0;
    let pad = pad_frac.0;
    let seed_val = seed.0;
    let vals = dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::adjusted_patch_nig_multiscale(&soa, &lbls, &radii_vec, c_sz, pad, seed_val)
    })?;
    let mut out = BTreeMap::new();
    for (&r, &val) in radii_vec.iter().zip(vals.iter()) {
        out.insert(format!("patch_nig_r{r}"), val);
    }
    Ok(out)
}

/// Compute Grid-based Cluster Area Faithfulness (`G-CAF`) in `[0.0, 1.0]`.
///
/// Evaluates how faithfully cluster areas in a 2D layout represent community sizes while
/// accounting for spatial purity, visual area proportion, density/deconvolution resolution,
/// and 8-connected spatial coherence.
///
/// Optional positive finite `node_weights` (length `N`) or `cluster_weights` (length `K`
/// ordered by sorted ascending unique community IDs) override uniform node counts before
/// applying `stevens_gamma`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, labels, *, g_max = StrictUsize(128), sigma_cells = StrictF64(0.85), kappa = StrictF64(10.0), tau_frac = StrictF64(0.18), boundary_r95 = StrictF64(2.0), stevens_gamma = StrictF64(1.0), node_weights = None, cluster_weights = None),
    text_signature = "(layout, labels, *, g_max=128, sigma_cells=0.85, kappa=10.0, tau_frac=0.18, boundary_r95=2.0, stevens_gamma=1.0, node_weights=None, cluster_weights=None)"
)]
#[allow(clippy::too_many_arguments)]
fn cluster_area_faithfulness(
    py: Python<'_>,
    layout: &PyLayout2d,
    labels: &Bound<'_, PyAny>,
    g_max: StrictUsize,
    sigma_cells: StrictF64,
    kappa: StrictF64,
    tau_frac: StrictF64,
    boundary_r95: StrictF64,
    stevens_gamma: StrictF64,
    node_weights: Option<&Bound<'_, PyAny>>,
    cluster_weights: Option<&Bound<'_, PyAny>>,
) -> PyResult<f64> {
    let lbls = extract_u32_1d(labels, "labels")?;
    let nw = extract_optional_weights_1d(node_weights, "node_weights")?;
    let cw = extract_optional_weights_1d(cluster_weights, "cluster_weights")?;
    let config = ClusterAreaConfig::new()
        .with_g_max(g_max.0)
        .with_sigma_cells(sigma_cells.0)
        .with_kappa(kappa.0)
        .with_tau_frac(tau_frac.0)
        .with_boundary_r95(boundary_r95.0)
        .with_stevens_gamma(stevens_gamma.0);
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::cluster_area_faithfulness_with_weights(
            &soa,
            &lbls,
            &config,
            nw.as_deref(),
            cw.as_deref(),
        )
        .map(|r| r.caf)
    })
}

/// Compute Grid-based Cluster Area Faithfulness (`G-CAF`) with summary and optional per-cluster diagnostics.
///
/// Returns a frozen `ClusterAreaReport` containing summary metrics (`caf`, `purity_adj`, `prop_vis`,
/// `res_density`, `res_deconv`, `res_combined`, `spatial_coherence`, `in_frac`,
/// `effective_boundary_r95`, `g_eff`, `num_clusters`, `num_nodes`). When `return_per_cluster=True`,
/// also includes `cluster_ids`, `counts`, `p`, `q`, `purity`, `resolution`,
/// `res_density_per_cluster`, `res_deconv_per_cluster`, `caf_per_cluster`, and `coherence`
/// as 1D NumPy arrays. Optional positive finite `node_weights` or `cluster_weights`
/// (in sorted ascending `cluster_ids` order) override uniform node counts before `stevens_gamma`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, labels, *, g_max = StrictUsize(128), sigma_cells = StrictF64(0.85), kappa = StrictF64(10.0), tau_frac = StrictF64(0.18), boundary_r95 = StrictF64(2.0), stevens_gamma = StrictF64(1.0), node_weights = None, cluster_weights = None, return_per_cluster = false),
    text_signature = "(layout, labels, *, g_max=128, sigma_cells=0.85, kappa=10.0, tau_frac=0.18, boundary_r95=2.0, stevens_gamma=1.0, node_weights=None, cluster_weights=None, return_per_cluster=False)"
)]
#[allow(clippy::too_many_arguments)]
fn cluster_area_faithfulness_detail(
    py: Python<'_>,
    layout: &PyLayout2d,
    labels: &Bound<'_, PyAny>,
    g_max: StrictUsize,
    sigma_cells: StrictF64,
    kappa: StrictF64,
    tau_frac: StrictF64,
    boundary_r95: StrictF64,
    stevens_gamma: StrictF64,
    node_weights: Option<&Bound<'_, PyAny>>,
    cluster_weights: Option<&Bound<'_, PyAny>>,
    return_per_cluster: bool,
) -> PyResult<PyClusterAreaReport> {
    let lbls = extract_u32_1d(labels, "labels")?;
    let nw = extract_optional_weights_1d(node_weights, "node_weights")?;
    let cw = extract_optional_weights_1d(cluster_weights, "cluster_weights")?;
    let config = ClusterAreaConfig::new()
        .with_g_max(g_max.0)
        .with_sigma_cells(sigma_cells.0)
        .with_kappa(kappa.0)
        .with_tau_frac(tau_frac.0)
        .with_boundary_r95(boundary_r95.0)
        .with_stevens_gamma(stevens_gamma.0);
    if !return_per_cluster {
        let report = dispatch_layout2d!(py, layout, |soa| {
            egraph_metrics::cluster_area_faithfulness_with_weights(
                &soa,
                &lbls,
                &config,
                nw.as_deref(),
                cw.as_deref(),
            )
        })?;
        Ok(PyClusterAreaReport::from_report(report))
    } else {
        let detail = dispatch_layout2d!(py, layout, |soa| {
            egraph_metrics::cluster_area_faithfulness_detail_with_weights(
                &soa,
                &lbls,
                &config,
                nw.as_deref(),
                cw.as_deref(),
            )
        })?;
        Ok(PyClusterAreaReport::from_detail(detail))
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(adjusted_patch_nig, m)?)?;
    m.add_function(wrap_pyfunction!(adjusted_patch_nig_detail, m)?)?;
    m.add_function(wrap_pyfunction!(adjusted_patch_nig_multiscale, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_area_faithfulness, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_area_faithfulness_detail, m)?)?;
    Ok(())
}
