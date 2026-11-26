//! Cluster-area faithfulness (`G-CAF`), Patch-NIG, and landmark-stratified RTA bindings.

use std::collections::BTreeMap;

use egraph::metrics as egraph_metrics;
use egraph_metrics::ClusterAreaConfig;
use numpy::{PyArray1, PyReadonlyArrayDyn, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict};

use crate::csr_graph::PyCsrGraph;
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
    signature = (layout, community_labels, *, patch_radius = StrictUsize(8), canvas_size = StrictUsize(2048), pad_frac = StrictF64(0.04), seed = StrictU64(42)),
    text_signature = "(layout, community_labels, *, patch_radius=8, canvas_size=2048, pad_frac=0.04, seed=42)"
)]
fn adjusted_patch_nig(
    py: Python<'_>,
    layout: &PyLayout2d,
    community_labels: &Bound<'_, PyAny>,
    patch_radius: StrictUsize,
    canvas_size: StrictUsize,
    pad_frac: StrictF64,
    seed: StrictU64,
) -> PyResult<f64> {
    let labels = extract_u32_1d(community_labels, "community_labels")?;
    let radius = patch_radius.0;
    let c_sz = canvas_size.0;
    let pad = pad_frac.0;
    let seed_val = seed.0;
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::adjusted_patch_nig(&soa, &labels, radius, c_sz, pad, seed_val)
    })
}

/// Compute multi-scale Adjusted Patch-NIG across `radii`, returning `{"patch_nig_r{r}": value}`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, community_labels, *, radii = None, canvas_size = StrictUsize(2048), pad_frac = StrictF64(0.04), seed = StrictU64(42)),
    text_signature = "(layout, community_labels, *, radii=(2, 8, 32), canvas_size=2048, pad_frac=0.04, seed=42)"
)]
fn adjusted_patch_nig_multiscale(
    py: Python<'_>,
    layout: &PyLayout2d,
    community_labels: &Bound<'_, PyAny>,
    radii: Option<&Bound<'_, PyAny>>,
    canvas_size: StrictUsize,
    pad_frac: StrictF64,
    seed: StrictU64,
) -> PyResult<BTreeMap<String, f64>> {
    let labels = extract_u32_1d(community_labels, "community_labels")?;
    let radii_vec = match radii {
        Some(obj) if !obj.is_none() => extract_positive_usize_seq(obj, "radii")?,
        _ => vec![2, 8, 32],
    };
    let c_sz = canvas_size.0;
    let pad = pad_frac.0;
    let seed_val = seed.0;
    let vals = dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::adjusted_patch_nig_multiscale(
            &soa, &labels, &radii_vec, c_sz, pad, seed_val,
        )
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
    signature = (layout, community_labels, *, g_max = StrictUsize(128), sigma_cells = StrictF64(0.85), kappa = StrictF64(10.0), tau_frac = StrictF64(0.18), boundary_r95 = StrictF64(2.0), stevens_gamma = StrictF64(1.0), node_weights = None, cluster_weights = None),
    text_signature = "(layout, community_labels, *, g_max=128, sigma_cells=0.85, kappa=10.0, tau_frac=0.18, boundary_r95=2.0, stevens_gamma=1.0, node_weights=None, cluster_weights=None)"
)]
#[allow(clippy::too_many_arguments)]
fn cluster_area_faithfulness(
    py: Python<'_>,
    layout: &PyLayout2d,
    community_labels: &Bound<'_, PyAny>,
    g_max: StrictUsize,
    sigma_cells: StrictF64,
    kappa: StrictF64,
    tau_frac: StrictF64,
    boundary_r95: StrictF64,
    stevens_gamma: StrictF64,
    node_weights: Option<&Bound<'_, PyAny>>,
    cluster_weights: Option<&Bound<'_, PyAny>>,
) -> PyResult<f64> {
    let labels = extract_u32_1d(community_labels, "community_labels")?;
    let nw = extract_optional_weights_1d(node_weights, "node_weights")?;
    let cw = extract_optional_weights_1d(cluster_weights, "cluster_weights")?;
    let config = ClusterAreaConfig {
        g_max: g_max.0,
        sigma_cells: sigma_cells.0,
        kappa: kappa.0,
        tau_frac: tau_frac.0,
        boundary_r95: boundary_r95.0,
        stevens_gamma: stevens_gamma.0,
    };
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::cluster_area_faithfulness_with_weights(
            &soa,
            &labels,
            &config,
            nw.as_deref(),
            cw.as_deref(),
        )
        .map(|r| r.caf)
    })
}

fn populate_cluster_area_report_dict(
    dict: &Bound<'_, PyDict>,
    report: &egraph_metrics::ClusterAreaReport,
) -> PyResult<()> {
    dict.set_item("caf", report.caf)?;
    dict.set_item("purity_adj", report.purity_adj)?;
    dict.set_item("prop_vis", report.prop_vis)?;
    dict.set_item("res_density", report.res_density)?;
    dict.set_item("res_deconv", report.res_deconv)?;
    dict.set_item("res_combined", report.res_combined)?;
    dict.set_item("spatial_coherence", report.spatial_coherence)?;
    dict.set_item("in_frac", report.in_frac)?;
    dict.set_item("effective_boundary_r95", report.effective_boundary_r95)?;
    dict.set_item("g_eff", report.g_eff)?;
    dict.set_item("num_clusters", report.num_clusters)?;
    dict.set_item("num_nodes", report.num_nodes)?;
    Ok(())
}

/// Compute Grid-based Cluster Area Faithfulness (`G-CAF`) with summary and optional per-cluster diagnostics.
///
/// Returns a dictionary containing summary metrics (`caf`, `purity_adj`, `prop_vis`,
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
    signature = (layout, community_labels, *, g_max = StrictUsize(128), sigma_cells = StrictF64(0.85), kappa = StrictF64(10.0), tau_frac = StrictF64(0.18), boundary_r95 = StrictF64(2.0), stevens_gamma = StrictF64(1.0), node_weights = None, cluster_weights = None, return_per_cluster = false),
    text_signature = "(layout, community_labels, *, g_max=128, sigma_cells=0.85, kappa=10.0, tau_frac=0.18, boundary_r95=2.0, stevens_gamma=1.0, node_weights=None, cluster_weights=None, return_per_cluster=False)"
)]
#[allow(clippy::too_many_arguments)]
fn cluster_area_faithfulness_detail(
    py: Python<'_>,
    layout: &PyLayout2d,
    community_labels: &Bound<'_, PyAny>,
    g_max: StrictUsize,
    sigma_cells: StrictF64,
    kappa: StrictF64,
    tau_frac: StrictF64,
    boundary_r95: StrictF64,
    stevens_gamma: StrictF64,
    node_weights: Option<&Bound<'_, PyAny>>,
    cluster_weights: Option<&Bound<'_, PyAny>>,
    return_per_cluster: bool,
) -> PyResult<Py<PyDict>> {
    let labels = extract_u32_1d(community_labels, "community_labels")?;
    let nw = extract_optional_weights_1d(node_weights, "node_weights")?;
    let cw = extract_optional_weights_1d(cluster_weights, "cluster_weights")?;
    let config = ClusterAreaConfig {
        g_max: g_max.0,
        sigma_cells: sigma_cells.0,
        kappa: kappa.0,
        tau_frac: tau_frac.0,
        boundary_r95: boundary_r95.0,
        stevens_gamma: stevens_gamma.0,
    };
    let dict = PyDict::new(py);
    if !return_per_cluster {
        let report = dispatch_layout2d!(py, layout, |soa| {
            egraph_metrics::cluster_area_faithfulness_with_weights(
                &soa,
                &labels,
                &config,
                nw.as_deref(),
                cw.as_deref(),
            )
        })?;
        populate_cluster_area_report_dict(&dict, &report)?;
    } else {
        let detail = dispatch_layout2d!(py, layout, |soa| {
            egraph_metrics::cluster_area_faithfulness_detail_with_weights(
                &soa,
                &labels,
                &config,
                nw.as_deref(),
                cw.as_deref(),
            )
        })?;
        populate_cluster_area_report_dict(&dict, &detail.report)?;
        dict.set_item("cluster_ids", PyArray1::from_vec(py, detail.cluster_ids))?;
        dict.set_item("counts", PyArray1::from_vec(py, detail.counts))?;
        dict.set_item("p", PyArray1::from_vec(py, detail.p))?;
        dict.set_item("q", PyArray1::from_vec(py, detail.q))?;
        dict.set_item("purity", PyArray1::from_vec(py, detail.purity))?;
        dict.set_item("resolution", PyArray1::from_vec(py, detail.resolution))?;
        // Rename per-cluster resolution fields to avoid colliding with scalar report keys.
        dict.set_item(
            "res_density_per_cluster",
            PyArray1::from_vec(py, detail.res_density),
        )?;
        dict.set_item(
            "res_deconv_per_cluster",
            PyArray1::from_vec(py, detail.res_deconv),
        )?;
        dict.set_item(
            "caf_per_cluster",
            PyArray1::from_vec(py, detail.caf_per_cluster),
        )?;
        dict.set_item("coherence", PyArray1::from_vec(py, detail.coherence))?;
    }
    Ok(dict.unbind())
}

/// Compute Landmark-Stratified Random Triplet Accuracy (`L-Stratified-RTA`) in `[0.0, 1.0]`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (graph, layout, *, num_landmarks = StrictUsize(64), num_triplets = StrictUsize(1_000_000), min_degree = StrictUsize(2), seed = StrictU64(42)),
    text_signature = "(graph, layout, *, num_landmarks=64, num_triplets=1000000, min_degree=2, seed=42)"
)]
fn landmark_stratified_rta(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
    num_landmarks: StrictUsize,
    num_triplets: StrictUsize,
    min_degree: StrictUsize,
    seed: StrictU64,
) -> PyResult<f64> {
    let g = graph.inner();
    let nl = num_landmarks.0;
    let nt = num_triplets.0;
    let md = min_degree.0;
    let s = seed.0;
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::landmark_stratified_rta(g, &soa, nl, nt, md, s)
    })
}

/// Compute Landmark-Stratified Random Triplet Accuracy (`L-Stratified-RTA`) with metadata dictionary.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (graph, layout, *, num_landmarks = StrictUsize(64), num_triplets = StrictUsize(1_000_000), min_degree = StrictUsize(2), seed = StrictU64(42)),
    text_signature = "(graph, layout, *, num_landmarks=64, num_triplets=1000000, min_degree=2, seed=42)"
)]
fn landmark_stratified_rta_detail(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
    num_landmarks: StrictUsize,
    num_triplets: StrictUsize,
    min_degree: StrictUsize,
    seed: StrictU64,
) -> PyResult<Py<PyDict>> {
    let g = graph.inner();
    let nl = num_landmarks.0;
    let nt = num_triplets.0;
    let md = min_degree.0;
    let s = seed.0;
    let eval = dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::landmark_stratified_rta_detail(g, &soa, nl, nt, md, s)
    })?;
    let dict = PyDict::new(py);
    dict.set_item("rta", eval.rta)?;
    dict.set_item("num_landmarks", eval.num_landmarks)?;
    dict.set_item("num_triplets", eval.num_triplets)?;
    dict.set_item("num_strata", eval.num_strata)?;
    dict.set_item("is_exact", eval.is_exact)?;
    Ok(dict.unbind())
}

fn float_to_landmark_u32(v: f64) -> PyResult<u32> {
    if v.is_nan() {
        return Err(PyValueError::new_err("landmark_dists contains NaN"));
    }
    if v == f64::INFINITY {
        return Ok(u32::MAX);
    }
    if v < 0.0 {
        return Err(PyValueError::new_err(format!(
            "landmark_dists entry {v} must be non-negative"
        )));
    }
    if v.fract() != 0.0 || v > u32::MAX as f64 {
        return Err(PyValueError::new_err(format!(
            "landmark_dists float entry {v} must be a non-negative integer or +inf"
        )));
    }
    Ok(v as u32)
}

fn is_numpy_bool_scalar(obj: &Bound<'_, PyAny>) -> bool {
    obj.getattr("dtype")
        .and_then(|dt| dt.str())
        .and_then(|s| s.to_str().map(|v| v == "bool"))
        .unwrap_or(false)
}

fn extract_landmark_item_u32(item: &Bound<'_, PyAny>) -> PyResult<u32> {
    if item.is_instance_of::<PyBool>() || is_numpy_bool_scalar(item) {
        return Err(PyTypeError::new_err(
            "landmark_dists entries must be integers or floats, not bool",
        ));
    }
    if let Ok(iv) = item.extract::<i128>() {
        return u32::try_from(iv).map_err(|_| {
            PyValueError::new_err(format!("landmark_dists entry {iv} out of range for u32"))
        });
    }
    if let Ok(fv) = item.extract::<f64>() {
        return float_to_landmark_u32(fv);
    }
    Err(PyTypeError::new_err(
        "landmark_dists entries must be integers or floats",
    ))
}

fn extract_landmark_dists_flat(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    if let Ok(untyped) = obj.cast::<PyUntypedArray>() {
        let ndim = untyped.ndim();
        if ndim != 1 && ndim != 2 {
            return Err(PyValueError::new_err(format!(
                "landmark_dists must be a 1D or 2D array, got {ndim}D"
            )));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<'_, f64>>() {
            return arr
                .as_array()
                .iter()
                .map(|&v| float_to_landmark_u32(v))
                .collect();
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<'_, f32>>() {
            return arr
                .as_array()
                .iter()
                .map(|&v| float_to_landmark_u32(f64::from(v)))
                .collect();
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<'_, u32>>() {
            return Ok(arr.as_array().iter().copied().collect());
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<'_, u16>>() {
            return Ok(arr
                .as_array()
                .iter()
                .map(|&v| {
                    if v == u16::MAX {
                        u32::MAX
                    } else {
                        u32::from(v)
                    }
                })
                .collect());
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<'_, u8>>() {
            return Ok(arr.as_array().iter().map(|&v| u32::from(v)).collect());
        }
        macro_rules! try_int_array {
            ($ty:ty) => {
                if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<'_, $ty>>() {
                    return arr
                        .as_array()
                        .iter()
                        .map(|&v| {
                            u32::try_from(v).map_err(|_| {
                                PyValueError::new_err(format!(
                                    "landmark_dists entry {v} out of range for u32"
                                ))
                            })
                        })
                        .collect();
                }
            };
        }
        try_int_array!(i64);
        try_int_array!(i32);
        try_int_array!(i16);
        try_int_array!(i8);
        try_int_array!(u64);
        return Err(PyTypeError::new_err(
            "landmark_dists NumPy array must have integer or float dtype",
        ));
    }

    if is_non_sequence_container(obj) {
        return Err(PyTypeError::new_err(
            "landmark_dists must be a 1D or 2D array or sequence",
        ));
    }
    let iter = obj
        .try_iter()
        .map_err(|_| PyTypeError::new_err("landmark_dists must be a 1D or 2D array or sequence"))?;
    let mut items = Vec::new();
    for item_res in iter {
        items.push(item_res?);
    }
    let Some(first) = items.first() else {
        return Ok(Vec::new());
    };
    if !is_non_sequence_container(first) && first.try_iter().is_ok() {
        let mut flat = Vec::new();
        let mut expected_cols: Option<usize> = None;
        for (r_idx, row_obj) in items.iter().enumerate() {
            if is_non_sequence_container(row_obj) {
                return Err(PyTypeError::new_err(
                    "landmark_dists rows must be sequences",
                ));
            }
            let row_iter = row_obj
                .try_iter()
                .map_err(|_| PyTypeError::new_err("landmark_dists rows must be sequences"))?;
            let mut row_len = 0usize;
            for elem_res in row_iter {
                flat.push(extract_landmark_item_u32(&elem_res?)?);
                row_len += 1;
            }
            match expected_cols {
                None => expected_cols = Some(row_len),
                Some(cols) if cols != row_len => {
                    return Err(PyValueError::new_err(format!(
                        "landmark_dists has ragged row {r_idx}: expected {cols} columns, got {row_len}"
                    )));
                }
                Some(_) => {}
            }
        }
        return Ok(flat);
    }

    items.iter().map(extract_landmark_item_u32).collect()
}

/// Compute Landmark-Stratified RTA from a precomputed `(P_L, N)` hop-distance table.
///
/// Accepts integer arrays (`2**32 - 1`, or `65535` for `uint16`, = unreachable) as well as
/// `float64`/`float32` tables from `all_pairs_bfs` (`np.inf` = unreachable).
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, landmark_ids, landmark_dists, *, num_triplets = StrictUsize(1_000_000), seed = StrictU64(42)),
    text_signature = "(layout, landmark_ids, landmark_dists, *, num_triplets=1000000, seed=42)"
)]
fn landmark_stratified_rta_from_landmarks(
    py: Python<'_>,
    layout: &PyLayout2d,
    landmark_ids: &Bound<'_, PyAny>,
    landmark_dists: &Bound<'_, PyAny>,
    num_triplets: StrictUsize,
    seed: StrictU64,
) -> PyResult<Py<PyDict>> {
    let l_ids = extract_u32_1d(landmark_ids, "landmark_ids")?;
    let l_dists = extract_landmark_dists_flat(landmark_dists)?;
    let nt = num_triplets.0;
    let s = seed.0;
    let eval = dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::landmark_stratified_rta_from_landmarks(&soa, &l_ids, &l_dists, nt, s)
    })?;
    let dict = PyDict::new(py);
    dict.set_item("rta", eval.rta)?;
    dict.set_item("num_landmarks", eval.num_landmarks)?;
    dict.set_item("num_triplets", eval.num_triplets)?;
    dict.set_item("num_strata", eval.num_strata)?;
    dict.set_item("is_exact", eval.is_exact)?;
    Ok(dict.unbind())
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(adjusted_patch_nig, m)?)?;
    m.add_function(wrap_pyfunction!(adjusted_patch_nig_multiscale, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_area_faithfulness, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_area_faithfulness_detail, m)?)?;
    m.add_function(wrap_pyfunction!(landmark_stratified_rta, m)?)?;
    m.add_function(wrap_pyfunction!(landmark_stratified_rta_detail, m)?)?;
    m.add_function(wrap_pyfunction!(landmark_stratified_rta_from_landmarks, m)?)?;
    Ok(())
}
