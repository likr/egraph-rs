//! Multi-level Louvain coarsening, centroid prolongation, and pivoted RHGH landmark helpers.

use egraph::{clustering as egraph_clustering, core as egraph_core};
use egraph_clustering::{
    CoarseLouvainConfig, align_clustered_centroids as clu_align_clustered_centroids,
    louvain_coarsen as clu_louvain_coarsen,
    project_clustered_layout_with_radii as clu_project_clustered_layout_with_radii,
};
use egraph_core::{CoreError, Layout2d};
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::array_extract::extract_f64_1d;
use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    StrictF64, StrictU64, StrictUsize, core_error_to_py, extract_u32_1d, ffi_error_to_py,
};
use crate::layout_buffers::{Layout2dInner, PyLayout2d};

/// Run multi-level Louvain coarsening with resolution decay until `num_clusters <= max_coarse_nodes`
/// or the connected-component floor is reached.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (
        graph,
        *,
        max_coarse_nodes = StrictUsize(10_000),
        resolution = StrictF64(1.0),
        resolution_decay = StrictF64(0.5),
        max_passes = StrictUsize(100),
        min_modularity_gain = StrictF64(1e-7),
        max_outer_rounds = StrictUsize(16),
        seed = StrictU64(42)
    ),
    text_signature = "(graph, *, max_coarse_nodes=10000, resolution=1.0, resolution_decay=0.5, max_passes=100, min_modularity_gain=1e-7, max_outer_rounds=16, seed=42)"
)]
fn louvain_coarsen(
    py: Python<'_>,
    graph: &PyCsrGraph,
    max_coarse_nodes: StrictUsize,
    resolution: StrictF64,
    resolution_decay: StrictF64,
    max_passes: StrictUsize,
    min_modularity_gain: StrictF64,
    max_outer_rounds: StrictUsize,
    seed: StrictU64,
) -> PyResult<Py<PyDict>> {
    let cfg = CoarseLouvainConfig {
        max_coarse_nodes: max_coarse_nodes.0,
        resolution: resolution.0,
        resolution_decay: resolution_decay.0,
        max_passes: max_passes.0,
        min_modularity_gain: min_modularity_gain.0,
        max_outer_rounds: max_outer_rounds.0,
        seed: seed.0,
    };
    let g = graph.inner();
    let hierarchy = py
        .detach(|| clu_louvain_coarsen(&*g, &cfg))
        .map_err(core_error_to_py)?;

    let level_num_clusters: Vec<usize> = hierarchy
        .levels
        .iter()
        .map(|lvl| lvl.coarse.num_clusters)
        .collect();
    let level_resolutions: Vec<f64> = hierarchy.levels.iter().map(|lvl| lvl.resolution).collect();
    let level_modularities: Vec<f64> = hierarchy.levels.iter().map(|lvl| lvl.modularity).collect();

    let dict = PyDict::new(py);
    dict.set_item(
        "coarse_graph",
        PyCsrGraph::from_csr_graph(hierarchy.coarsest.graph),
    )?;
    dict.set_item(
        "node_to_coarse",
        PyArray1::from_vec(py, hierarchy.coarsest.node_to_coarse),
    )?;
    dict.set_item(
        "coarse_sizes",
        PyArray1::from_vec(py, hierarchy.coarsest.coarse_sizes),
    )?;
    dict.set_item("num_clusters", hierarchy.coarsest.num_clusters)?;
    dict.set_item("num_levels", hierarchy.levels.len())?;
    dict.set_item("level_num_clusters", level_num_clusters)?;
    dict.set_item("level_resolutions", level_resolutions)?;
    dict.set_item("level_modularities", level_modularities)?;
    Ok(dict.unbind())
}

/// Project a coarse cluster `Layout2d` onto fine vertices using per-cluster `cluster_radii`.
///
/// Preserves `coarse_layout.dtype` (`"float64"` or `"float32"`). Releases the Python GIL.
#[pyfunction]
#[pyo3(
    signature = (coarse_layout, node_to_coarse, cluster_radii, *, jitter_scale = StrictF64(0.5), seed = StrictU64(42)),
    text_signature = "(coarse_layout, node_to_coarse, cluster_radii, *, jitter_scale=0.5, seed=42)"
)]
fn project_clustered_layout_with_radii(
    py: Python<'_>,
    coarse_layout: &PyLayout2d,
    node_to_coarse: &Bound<'_, PyAny>,
    cluster_radii: &Bound<'_, PyAny>,
    jitter_scale: StrictF64,
    seed: StrictU64,
) -> PyResult<PyLayout2d> {
    let n2c = extract_u32_1d(node_to_coarse, "node_to_coarse")?;
    let radii = extract_f64_1d(cluster_radii, "cluster_radii")?;
    let j_scale = jitter_scale.0;
    let seed_val = seed.0;
    py.detach(|| match coarse_layout.inner() {
        Layout2dInner::F64(buf) => {
            let soa = buf.to_layout2d();
            let fine =
                clu_project_clustered_layout_with_radii(&soa, &n2c, &radii, j_scale, seed_val)?;
            Ok(PyLayout2d::from_f64_layout(&fine))
        }
        Layout2dInner::F32(buf) => {
            let soa = buf.to_layout2d();
            let fine =
                clu_project_clustered_layout_with_radii(&soa, &n2c, &radii, j_scale, seed_val)?;
            Ok(PyLayout2d::from_f32_layout(&fine))
        }
    })
    .map_err(core_error_to_py)
}

/// Align fine-level cluster centroids toward `coarse_layout` via rigid per-cluster translation
/// `x_i <- x_i + gamma_tether * (x_C - mean(x_C))` in place without rigid disk containment.
///
/// Preserves `fine_layout.dtype`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (fine_layout, coarse_layout, node_to_coarse, *, gamma_tether = StrictF64(0.5)),
    text_signature = "(fine_layout, coarse_layout, node_to_coarse, *, gamma_tether=0.5)"
)]
fn align_clustered_centroids(
    py: Python<'_>,
    fine_layout: &Bound<'_, PyLayout2d>,
    coarse_layout: &Bound<'_, PyLayout2d>,
    node_to_coarse: &Bound<'_, PyAny>,
    gamma_tether: StrictF64,
) -> PyResult<()> {
    let n2c = extract_u32_1d(node_to_coarse, "node_to_coarse")?;
    let tether = gamma_tether.0;
    let coarse_borrowed = coarse_layout.try_borrow()?;
    let mut fine_borrowed = fine_layout.try_borrow_mut()?;

    let coarse_64 = match coarse_borrowed.inner() {
        Layout2dInner::F64(b) => b.to_layout2d(),
        Layout2dInner::F32(b) => {
            let s32 = b.to_layout2d();
            Layout2d::from_xy(
                s32.x().iter().map(|&v| f64::from(v)).collect(),
                s32.y().iter().map(|&v| f64::from(v)).collect(),
            )
            .map_err(core_error_to_py)?
        }
    };

    match fine_borrowed.inner_mut() {
        Layout2dInner::F64(buf) => {
            let mut fine_soa = buf.to_layout2d();
            py.detach(|| clu_align_clustered_centroids(&mut fine_soa, &coarse_64, &n2c, tether))
                .map_err(core_error_to_py)?;
            buf.copy_from_layout2d(&fine_soa).map_err(ffi_error_to_py)
        }
        Layout2dInner::F32(buf) => {
            let mut fine_soa = buf.to_layout2d();
            let cx32: Vec<f32> = coarse_64
                .x()
                .iter()
                .map(|&v| {
                    let f = v as f32;
                    if !f.is_finite() {
                        Err(CoreError::non_finite(
                            "align_clustered_centroids::coarse_layout",
                        ))
                    } else {
                        Ok(f)
                    }
                })
                .collect::<Result<_, _>>()
                .map_err(core_error_to_py)?;
            let cy32: Vec<f32> = coarse_64
                .y()
                .iter()
                .map(|&v| {
                    let f = v as f32;
                    if !f.is_finite() {
                        Err(CoreError::non_finite(
                            "align_clustered_centroids::coarse_layout",
                        ))
                    } else {
                        Ok(f)
                    }
                })
                .collect::<Result<_, _>>()
                .map_err(core_error_to_py)?;
            let coarse_32 = Layout2d::from_xy(cx32, cy32).map_err(core_error_to_py)?;
            py.detach(|| clu_align_clustered_centroids(&mut fine_soa, &coarse_32, &n2c, tether))
                .map_err(core_error_to_py)?;
            buf.copy_from_layout2d(&fine_soa).map_err(ffi_error_to_py)
        }
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(louvain_coarsen, m)?)?;
    m.add_function(wrap_pyfunction!(project_clustered_layout_with_radii, m)?)?;
    m.add_function(wrap_pyfunction!(align_clustered_centroids, m)?)?;
    Ok(())
}
