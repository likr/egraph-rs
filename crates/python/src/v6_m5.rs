//! M5 v6 Python bindings for Louvain coarsening, centroid prolongation,
//! pure-Rust pivoted `RHGH` landmarks, coarse edge-length-first area reservation,
//! and `PaCMAP-Graph` coarse backbone initialization.

use egraph::{
    clustering as egraph_clustering, core as egraph_core, layout as egraph_layout,
    neighborhood as egraph_neighborhood, spatial as egraph_spatial,
};
use egraph_clustering::{
    CoarseLouvainConfig, align_clustered_centroids as clu_align_clustered_centroids,
    louvain_coarsen as clu_louvain_coarsen,
    project_clustered_layout_with_radii as clu_project_clustered_layout_with_radii,
};
use egraph_core::{CoreError, Layout2d};
use egraph_layout::{
    CoarseBackboneConfig, CoarseBackboneResult, CoarseBackboneStats, CoarseDistanceBackend,
    CoarseEdgeLengthMode, PacmapGraphConfig, ScreenedPoissonRefiner, ScreenedPoissonRefinerConfig,
    SgdSolver, coarse_backbone_init as layout_coarse_backbone_init,
};
use egraph_neighborhood::{
    RhghLandmarkConfig, compute_pivoted_rhgh_landmarks as nb_compute_pivoted_rhgh_landmarks,
};
use egraph_spatial::{
    DensityFieldKernel, ScreenedPoissonConfig, ScreenedPoissonField2d, ScreenedPoissonOutput,
    ScreenedPoissonStats,
};
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::v6_m3::{PySgdSolver, SgdInner};
use crate::array_extract::extract_f64_1d;
use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, extract_u32_1d,
    ffi_error_to_py, normalize_enum_key, solver_status_str,
};
use crate::layout_buffers::{Layout2dInner, PyLayout2d};
use crate::neighborhood::PyDogVaradhanLandmarks;
use crate::solver_common::{
    DualIterativeSolver, SolverAction, SolverLayout, execute_solver_action, extract_solver_layout,
    impl_dual_solver, write_back_solver_layout,
};

fn parse_coarse_edge_length_mode(raw: &str) -> Result<CoarseEdgeLengthMode, String> {
    match normalize_enum_key(raw).as_str() {
        "conductanceinverse" | "conductance" | "inverseconductance" => {
            Ok(CoarseEdgeLengthMode::ConductanceInverse)
        }
        "resistancecurvatureflow" | "resistancecurvature" | "rcf" => {
            Ok(CoarseEdgeLengthMode::ResistanceCurvatureFlow)
        }
        _ => Err(format!(
            "unknown edge_length_mode {raw:?}; expected 'conductance_inverse' or 'resistance_curvature_flow'"
        )),
    }
}

fn parse_coarse_distance_backend(raw: &str) -> Result<CoarseDistanceBackend, String> {
    match normalize_enum_key(raw).as_str() {
        "rhgh" => Ok(CoarseDistanceBackend::Rhgh),
        "chebyshevdog" | "dog" => Ok(CoarseDistanceBackend::ChebyshevDog),
        "shortestpath" | "sp" => Ok(CoarseDistanceBackend::ShortestPath),
        _ => Err(format!(
            "unknown distance_backend {raw:?}; expected 'rhgh', 'chebyshev_dog', or 'shortest_path'"
        )),
    }
}

fn backbone_stats_to_dict(py: Python<'_>, st: &CoarseBackboneStats) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("num_fine_nodes", st.num_fine_nodes)?;
    dict.set_item("num_coarse_nodes", st.num_coarse_nodes)?;
    dict.set_item("num_coarse_edges", st.num_coarse_edges)?;
    dict.set_item("louvain_levels", st.louvain_levels)?;
    dict.set_item("mean_cluster_size", st.mean_cluster_size)?;
    dict.set_item("max_cluster_size", st.max_cluster_size)?;
    dict.set_item("median_l_edge", st.median_l_edge)?;
    dict.set_item("max_radius", st.max_radius)?;
    dict.set_item(
        "coarse_overlap_pairs_initial",
        st.coarse_overlap_pairs_initial,
    )?;
    dict.set_item("coarse_overlap_pairs_final", st.coarse_overlap_pairs_final)?;
    dict.set_item("scale_factor_applied", st.scale_factor_applied)?;
    Ok(dict.unbind())
}

enum CoarseBackboneOut {
    F64(CoarseBackboneResult<f64>),
    F32(CoarseBackboneResult<f32>),
}

fn backbone_out_to_dict(py: Python<'_>, out: CoarseBackboneOut) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    match out {
        CoarseBackboneOut::F64(res) => {
            dict.set_item("fine_layout", PyLayout2d::from_f64_layout(&res.fine_layout))?;
            dict.set_item(
                "coarse_layout",
                PyLayout2d::from_f64_layout(&res.coarse_layout),
            )?;
            dict.set_item(
                "coarse_graph",
                PyCsrGraph::from_csr_graph(res.hierarchy.coarsest.graph),
            )?;
            dict.set_item(
                "node_to_coarse",
                PyArray1::from_vec(py, res.hierarchy.coarsest.node_to_coarse),
            )?;
            dict.set_item(
                "coarse_sizes",
                PyArray1::from_vec(py, res.hierarchy.coarsest.coarse_sizes),
            )?;
            dict.set_item("radii", PyArray1::from_vec(py, res.radii))?;
            dict.set_item("stats", backbone_stats_to_dict(py, &res.stats)?)?;
        }
        CoarseBackboneOut::F32(res) => {
            dict.set_item("fine_layout", PyLayout2d::from_f32_layout(&res.fine_layout))?;
            dict.set_item(
                "coarse_layout",
                PyLayout2d::from_f32_layout(&res.coarse_layout),
            )?;
            dict.set_item(
                "coarse_graph",
                PyCsrGraph::from_csr_graph(res.hierarchy.coarsest.graph),
            )?;
            dict.set_item(
                "node_to_coarse",
                PyArray1::from_vec(py, res.hierarchy.coarsest.node_to_coarse),
            )?;
            dict.set_item(
                "coarse_sizes",
                PyArray1::from_vec(py, res.hierarchy.coarsest.coarse_sizes),
            )?;
            dict.set_item("radii", PyArray1::from_vec(py, res.radii))?;
            dict.set_item("stats", backbone_stats_to_dict(py, &res.stats)?)?;
        }
    }
    Ok(dict.unbind())
}

#[allow(clippy::too_many_arguments)]
fn build_coarse_backbone_config(
    max_coarse_nodes: StrictUsize,
    resolution: StrictF64,
    resolution_decay: StrictF64,
    max_passes: StrictUsize,
    min_modularity_gain: StrictF64,
    max_outer_rounds: StrictUsize,
    edge_length_mode: &str,
    distance_backend: &str,
    alpha: StrictF64,
    c0: StrictF64,
    kappa: StrictF64,
    gamma_w: StrictF64,
    gamma_m: StrictF64,
    mu_ov: StrictF64,
    m_gap: StrictF64,
    coarse_pivots: StrictUsize,
    coarse_neg_per_node: StrictUsize,
    coarse_t_max: StrictUsize,
    rhgh_gamma: Option<StrictF64>,
    rhgh_p: Option<StrictF64>,
    rcf_steps: StrictUsize,
    rcf_eta: StrictF64,
    jitter_scale: StrictF64,
    gamma_tether: StrictF64,
    target_std: Option<StrictF64>,
    robust_quantile: StrictF64,
    seed: StrictU64,
) -> PyResult<CoarseBackboneConfig> {
    let mode = parse_coarse_edge_length_mode(edge_length_mode).map_err(PyValueError::new_err)?;
    let backend = parse_coarse_distance_backend(distance_backend).map_err(PyValueError::new_err)?;
    let cfg = CoarseBackboneConfig {
        louvain: CoarseLouvainConfig {
            max_coarse_nodes: max_coarse_nodes.0,
            resolution: resolution.0,
            resolution_decay: resolution_decay.0,
            max_passes: max_passes.0,
            min_modularity_gain: min_modularity_gain.0,
            max_outer_rounds: max_outer_rounds.0,
            seed: seed.0,
        },
        edge_length_mode: mode,
        distance_backend: backend,
        alpha: alpha.0,
        c0: c0.0,
        kappa: kappa.0,
        gamma_w: gamma_w.0,
        gamma_m: gamma_m.0,
        mu_ov: mu_ov.0,
        m_gap: m_gap.0,
        coarse_pivots: coarse_pivots.0,
        coarse_neg_per_node: coarse_neg_per_node.0,
        coarse_t_max: coarse_t_max.0,
        rhgh_gamma: rhgh_gamma.map(|v| v.0),
        rhgh_p: rhgh_p.map(|v| v.0),
        rcf_steps: rcf_steps.0,
        rcf_eta: rcf_eta.0,
        jitter_scale: jitter_scale.0,
        gamma_tether: gamma_tether.0,
        target_std: target_std.map(|v| v.0),
        robust_quantile: robust_quantile.0,
        seed: seed.0,
    };
    cfg.validate().map_err(core_error_to_py)?;
    Ok(cfg)
}

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
        .detach(|| clu_louvain_coarsen(g, &cfg))
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

/// Compute pure-Rust pivoted Resolvent Heat-Geodesic Hybrid (`RHGH`) landmark tables
/// using `AmgPcgSolver` + Spielman-Srivastava JL probes.
///
/// Returns `(DogVaradhanLandmarks, stats_dict)`. Releases the Python GIL during computation.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (
        graph,
        *,
        gamma = None,
        p_exp = None,
        num_landmarks = StrictUsize(64),
        min_degree = StrictUsize(2),
        num_probes = StrictUsize(64),
        rel_tol = StrictF64(1e-12),
        max_iterations = StrictUsize(500),
        normalize_1hop = true,
        seed = StrictU64(42)
    ),
    text_signature = "(graph, *, gamma=None, p_exp=None, num_landmarks=64, min_degree=2, num_probes=64, rel_tol=1e-12, max_iterations=500, normalize_1hop=True, seed=42)"
)]
fn compute_pivoted_rhgh_landmarks(
    py: Python<'_>,
    graph: &PyCsrGraph,
    gamma: Option<StrictF64>,
    p_exp: Option<StrictF64>,
    num_landmarks: StrictUsize,
    min_degree: StrictUsize,
    num_probes: StrictUsize,
    rel_tol: StrictF64,
    max_iterations: StrictUsize,
    normalize_1hop: bool,
    seed: StrictU64,
) -> PyResult<(PyDogVaradhanLandmarks, Py<PyDict>)> {
    let cfg = RhghLandmarkConfig {
        gamma: gamma.map(|v| v.0),
        p_exp: p_exp.map(|v| v.0),
        n_landmarks: num_landmarks.0,
        min_degree: min_degree.0,
        num_probes: num_probes.0,
        rel_tol: rel_tol.0,
        max_iterations: max_iterations.0,
        normalize_1hop,
        seed: seed.0,
    };
    let g = graph.inner();
    let (inner, st) = py
        .detach(|| nb_compute_pivoted_rhgh_landmarks(g, &cfg))
        .map_err(core_error_to_py)?;

    let dict = PyDict::new(py);
    dict.set_item("gamma", st.gamma)?;
    dict.set_item("p_exp", st.p_exp)?;
    dict.set_item("deg_cv", st.deg_cv)?;
    dict.set_item("leaf_fraction", st.leaf_fraction)?;
    dict.set_item("mean_degree", st.mean_degree)?;
    dict.set_item("exp2", st.exp2)?;
    dict.set_item("s_sw", st.s_sw)?;
    dict.set_item("n_landmarks", st.n_landmarks)?;
    dict.set_item("num_probes", st.num_probes)?;
    dict.set_item("max_pcg_iters", st.max_pcg_iters)?;
    dict.set_item("total_pcg_iters", st.total_pcg_iters)?;
    dict.set_item("all_converged", st.all_converged)?;
    dict.set_item("edge_med_scale", st.edge_med_scale)?;

    Ok((
        PyDogVaradhanLandmarks::from_rhgh_parts(inner, &st),
        dict.unbind(),
    ))
}

/// Run coarse Louvain edge-length-first area reservation and non-SPD backbone initialization.
///
/// Returns a dictionary with keys `'fine_layout'`, `'coarse_layout'`, `'coarse_graph'`,
/// `'node_to_coarse'`, `'coarse_sizes'`, `'radii'`, and `'stats'`.
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
        edge_length_mode = "conductance_inverse",
        distance_backend = "rhgh",
        alpha = StrictF64(0.75),
        c0 = StrictF64(0.5),
        kappa = StrictF64(1.0),
        gamma_w = StrictF64(0.5),
        gamma_m = StrictF64(0.5),
        mu_ov = StrictF64(1.0),
        m_gap = StrictF64(0.1),
        coarse_pivots = StrictUsize(64),
        coarse_neg_per_node = StrictUsize(16),
        coarse_t_max = StrictUsize(80),
        rhgh_gamma = None,
        rhgh_p = None,
        rcf_steps = StrictUsize(2),
        rcf_eta = StrictF64(0.2),
        jitter_scale = StrictF64(0.5),
        gamma_tether = StrictF64(0.5),
        target_std = Some(StrictF64(1.0)),
        robust_quantile = StrictF64(0.95),
        seed = StrictU64(42),
        dtype = ScalarDtype::F64
    ),
    text_signature = "(graph, *, max_coarse_nodes=10000, resolution=1.0, resolution_decay=0.5, max_passes=100, min_modularity_gain=1e-7, max_outer_rounds=16, edge_length_mode='conductance_inverse', distance_backend='rhgh', alpha=0.75, c0=0.5, kappa=1.0, gamma_w=0.5, gamma_m=0.5, mu_ov=1.0, m_gap=0.1, coarse_pivots=64, coarse_neg_per_node=16, coarse_t_max=80, rhgh_gamma=None, rhgh_p=None, rcf_steps=2, rcf_eta=0.2, jitter_scale=0.5, gamma_tether=0.5, target_std=1.0, robust_quantile=0.95, seed=42, dtype='float64')"
)]
fn coarse_backbone_init(
    py: Python<'_>,
    graph: &PyCsrGraph,
    max_coarse_nodes: StrictUsize,
    resolution: StrictF64,
    resolution_decay: StrictF64,
    max_passes: StrictUsize,
    min_modularity_gain: StrictF64,
    max_outer_rounds: StrictUsize,
    edge_length_mode: &str,
    distance_backend: &str,
    alpha: StrictF64,
    c0: StrictF64,
    kappa: StrictF64,
    gamma_w: StrictF64,
    gamma_m: StrictF64,
    mu_ov: StrictF64,
    m_gap: StrictF64,
    coarse_pivots: StrictUsize,
    coarse_neg_per_node: StrictUsize,
    coarse_t_max: StrictUsize,
    rhgh_gamma: Option<StrictF64>,
    rhgh_p: Option<StrictF64>,
    rcf_steps: StrictUsize,
    rcf_eta: StrictF64,
    jitter_scale: StrictF64,
    gamma_tether: StrictF64,
    target_std: Option<StrictF64>,
    robust_quantile: StrictF64,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<Py<PyDict>> {
    let cfg = build_coarse_backbone_config(
        max_coarse_nodes,
        resolution,
        resolution_decay,
        max_passes,
        min_modularity_gain,
        max_outer_rounds,
        edge_length_mode,
        distance_backend,
        alpha,
        c0,
        kappa,
        gamma_w,
        gamma_m,
        mu_ov,
        m_gap,
        coarse_pivots,
        coarse_neg_per_node,
        coarse_t_max,
        rhgh_gamma,
        rhgh_p,
        rcf_steps,
        rcf_eta,
        jitter_scale,
        gamma_tether,
        target_std,
        robust_quantile,
        seed,
    )?;
    let g = graph.inner();
    let out = py
        .detach(|| match dtype {
            ScalarDtype::F64 => {
                layout_coarse_backbone_init::<f64, f64>(g, &cfg).map(CoarseBackboneOut::F64)
            }
            ScalarDtype::F32 => {
                layout_coarse_backbone_init::<f64, f32>(g, &cfg).map(CoarseBackboneOut::F32)
            }
        })
        .map_err(core_error_to_py)?;
    backbone_out_to_dict(py, out)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_pacmap_graph_with_backbone(
    py: Python<'_>,
    graph: &PyCsrGraph,
    n_neighbors: StrictUsize,
    mn_ratio: StrictF64,
    fp_ratio: StrictF64,
    phase1_iters: StrictUsize,
    phase2_iters: StrictUsize,
    phase3_iters: StrictUsize,
    learning_rate: StrictF64,
    keep_all_1hop: bool,
    max_1hop_per_node: Option<StrictUsize>,
    degree_weight_beta: StrictF64,
    num_landmarks: StrictUsize,
    max_coarse_nodes: StrictUsize,
    resolution: StrictF64,
    resolution_decay: StrictF64,
    max_passes: StrictUsize,
    min_modularity_gain: StrictF64,
    max_outer_rounds: StrictUsize,
    edge_length_mode: &str,
    distance_backend: &str,
    alpha: StrictF64,
    c0: StrictF64,
    kappa: StrictF64,
    gamma_w: StrictF64,
    gamma_m: StrictF64,
    mu_ov: StrictF64,
    m_gap: StrictF64,
    coarse_pivots: StrictUsize,
    coarse_neg_per_node: StrictUsize,
    coarse_t_max: StrictUsize,
    rhgh_gamma: Option<StrictF64>,
    rhgh_p: Option<StrictF64>,
    rcf_steps: StrictUsize,
    rcf_eta: StrictF64,
    jitter_scale: StrictF64,
    gamma_tether: StrictF64,
    target_std: Option<StrictF64>,
    robust_quantile: StrictF64,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<(PySgdSolver, PyLayout2d, Py<PyDict>)> {
    let pm_cfg = PacmapGraphConfig {
        n_neighbors: n_neighbors.0,
        mn_ratio: mn_ratio.0,
        fp_ratio: fp_ratio.0,
        phase1_iters: phase1_iters.0,
        phase2_iters: phase2_iters.0,
        phase3_iters: phase3_iters.0,
        learning_rate: learning_rate.0,
        seed: seed.0,
        keep_all_1hop,
        max_1hop_per_node: max_1hop_per_node.map(|v| v.0),
        degree_weight_beta: degree_weight_beta.0,
        num_landmarks: num_landmarks.0,
    };
    pm_cfg.validate().map_err(core_error_to_py)?;
    let bb_cfg = build_coarse_backbone_config(
        max_coarse_nodes,
        resolution,
        resolution_decay,
        max_passes,
        min_modularity_gain,
        max_outer_rounds,
        edge_length_mode,
        distance_backend,
        alpha,
        c0,
        kappa,
        gamma_w,
        gamma_m,
        mu_ov,
        m_gap,
        coarse_pivots,
        coarse_neg_per_node,
        coarse_t_max,
        rhgh_gamma,
        rhgh_p,
        rcf_steps,
        rcf_eta,
        jitter_scale,
        gamma_tether,
        target_std,
        robust_quantile,
        seed,
    )?;

    let g = graph.inner();
    match dtype {
        ScalarDtype::F64 => {
            let (solver, init_layout, bb) = py
                .detach(|| SgdSolver::<f64>::pacmap_graph_with_backbone(g, pm_cfg, &bb_cfg))
                .map_err(core_error_to_py)?;
            Ok((
                PySgdSolver::from_inner(SgdInner::F64(solver)),
                PyLayout2d::from_f64_layout(&init_layout),
                backbone_out_to_dict(py, CoarseBackboneOut::F64(bb))?,
            ))
        }
        ScalarDtype::F32 => {
            let (solver, init_layout, bb) = py
                .detach(|| SgdSolver::<f32>::pacmap_graph_with_backbone(g, pm_cfg, &bb_cfg))
                .map_err(core_error_to_py)?;
            Ok((
                PySgdSolver::from_inner(SgdInner::F32(solver)),
                PyLayout2d::from_f32_layout(&init_layout),
                backbone_out_to_dict(py, CoarseBackboneOut::F32(bb))?,
            ))
        }
    }
}

/// Register M5 coarsening, pivoted RHGH landmark, coarse backbone, and Screened Poisson
/// density field bindings on `egraph._core`.
pub(super) fn register_m5(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(louvain_coarsen, m)?)?;
    m.add_function(wrap_pyfunction!(project_clustered_layout_with_radii, m)?)?;
    m.add_function(wrap_pyfunction!(align_clustered_centroids, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pivoted_rhgh_landmarks, m)?)?;
    m.add_function(wrap_pyfunction!(coarse_backbone_init, m)?)?;
    m.add_class::<PyScreenedPoissonField2d>()?;
    m.add_class::<PyScreenedPoissonRefiner>()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// M5.2 Screened Poisson Unilateral Upper-Bound Density Field & Refiner
// ---------------------------------------------------------------------------

fn parse_density_field_kernel(raw: &str) -> Result<DensityFieldKernel, String> {
    match normalize_enum_key(raw).as_str() {
        "screenedpoisson" | "sp" => Ok(DensityFieldKernel::ScreenedPoisson),
        "singlegaussian" | "gaussian" | "sg" => Ok(DensityFieldKernel::SingleGaussian),
        _ => Err(format!(
            "unknown density field kernel {raw:?}; expected 'screened_poisson' or 'single_gaussian'"
        )),
    }
}

fn density_field_kernel_str(kernel: DensityFieldKernel) -> &'static str {
    match kernel {
        DensityFieldKernel::ScreenedPoisson => "screened_poisson",
        DensityFieldKernel::SingleGaussian => "single_gaussian",
    }
}

#[allow(clippy::too_many_arguments)]
fn build_screened_poisson_config(
    kernel: &str,
    grid_size: StrictUsize,
    padding_factor: StrictUsize,
    margin_ratio: StrictF64,
    sigma_ratio: StrictF64,
    sigma: Option<StrictF64>,
    screening_length_ratio: StrictF64,
    kappa: Option<StrictF64>,
    rho_max_factor: StrictF64,
    rho_max: Option<StrictF64>,
    exact_adjoint: bool,
) -> PyResult<ScreenedPoissonConfig> {
    let kernel_enum = parse_density_field_kernel(kernel).map_err(PyValueError::new_err)?;
    let cfg = ScreenedPoissonConfig {
        kernel: kernel_enum,
        grid_size: grid_size.0,
        padding_factor: padding_factor.0,
        margin_ratio: margin_ratio.0,
        sigma_ratio: sigma_ratio.0,
        sigma: sigma.map(|v| v.0),
        screening_length_ratio: screening_length_ratio.0,
        kappa: kappa.map(|v| v.0),
        rho_max_factor: rho_max_factor.0,
        rho_max: rho_max.map(|v| v.0),
        exact_adjoint,
    };
    cfg.validate().map_err(core_error_to_py)?;
    Ok(cfg)
}

fn screened_poisson_stats_to_dict(
    py: Python<'_>,
    st: &ScreenedPoissonStats,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("energy", st.energy)?;
    dict.set_item("max_density", st.max_density)?;
    dict.set_item("excess_fraction", st.excess_fraction)?;
    dict.set_item("excess_mass", st.excess_mass)?;
    dict.set_item("sigma_used", st.sigma_used)?;
    dict.set_item("kappa_used", st.kappa_used)?;
    dict.set_item("rho_max_used", st.rho_max_used)?;
    dict.set_item("grid_dim", st.grid_dim)?;
    dict.set_item("pad_dim", st.pad_dim)?;
    dict.set_item("fft_2d_calls", st.fft_2d_calls)?;
    dict.set_item("early_exit_no_excess", st.early_exit_no_excess)?;
    Ok(dict.unbind())
}

enum ScreenedPoissonFieldInner {
    F64(ScreenedPoissonField2d<f64>),
    F32(ScreenedPoissonField2d<f32>),
}

enum ScreenedPoissonOutInner {
    F64(ScreenedPoissonOutput<f64>),
    F32(ScreenedPoissonOutput<f32>),
}

fn screened_poisson_output_to_dict(
    py: Python<'_>,
    out: ScreenedPoissonOutInner,
) -> PyResult<Py<PyDict>> {
    match out {
        ScreenedPoissonOutInner::F64(o) => {
            let dict = screened_poisson_stats_to_dict(py, &o.stats)?;
            let bound = dict.bind(py);
            bound.set_item("force_x", PyArray1::from_vec(py, o.force_x))?;
            bound.set_item("force_y", PyArray1::from_vec(py, o.force_y))?;
            Ok(dict)
        }
        ScreenedPoissonOutInner::F32(o) => {
            let dict = screened_poisson_stats_to_dict(py, &o.stats)?;
            let bound = dict.bind(py);
            bound.set_item("force_x", PyArray1::from_vec(py, o.force_x))?;
            bound.set_item("force_y", PyArray1::from_vec(py, o.force_y))?;
            Ok(dict)
        }
    }
}

/// Reusable `f32` R2C-FFT Screened Poisson unilateral upper-bound 2D density field evaluator.
///
/// Releases the Python GIL during evaluation.
#[pyclass(name = "ScreenedPoissonField2d", module = "egraph._core")]
pub struct PyScreenedPoissonField2d {
    inner: ScreenedPoissonFieldInner,
}

#[pymethods]
impl PyScreenedPoissonField2d {
    /// Create a `ScreenedPoissonField2d` evaluator workspace.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            *,
            kernel = "screened_poisson",
            grid_size = StrictUsize(2048),
            padding_factor = StrictUsize(2),
            margin_ratio = StrictF64(0.1),
            sigma_ratio = StrictF64(0.015),
            sigma = None,
            screening_length_ratio = StrictF64(0.20),
            kappa = None,
            rho_max_factor = StrictF64(4.0),
            rho_max = None,
            exact_adjoint = true,
            dtype = ScalarDtype::F64
        ),
        text_signature = "(*, kernel='screened_poisson', grid_size=2048, padding_factor=2, margin_ratio=0.1, sigma_ratio=0.015, sigma=None, screening_length_ratio=0.2, kappa=None, rho_max_factor=4.0, rho_max=None, exact_adjoint=True, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        kernel: &str,
        grid_size: StrictUsize,
        padding_factor: StrictUsize,
        margin_ratio: StrictF64,
        sigma_ratio: StrictF64,
        sigma: Option<StrictF64>,
        screening_length_ratio: StrictF64,
        kappa: Option<StrictF64>,
        rho_max_factor: StrictF64,
        rho_max: Option<StrictF64>,
        exact_adjoint: bool,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = build_screened_poisson_config(
            kernel,
            grid_size,
            padding_factor,
            margin_ratio,
            sigma_ratio,
            sigma,
            screening_length_ratio,
            kappa,
            rho_max_factor,
            rho_max,
            exact_adjoint,
        )?;
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    ScreenedPoissonField2d::<f64>::new(cfg).map(ScreenedPoissonFieldInner::F64)
                }
                ScalarDtype::F32 => {
                    ScreenedPoissonField2d::<f32>::new(cfg).map(ScreenedPoissonFieldInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Evaluate unilateral density energy and descent forces over `layout`'s adaptive bounding box.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn evaluate(&mut self, py: Python<'_>, layout: &PyLayout2d) -> PyResult<Py<PyDict>> {
        let out = match (&mut self.inner, layout.inner()) {
            (ScreenedPoissonFieldInner::F64(field), Layout2dInner::F64(buf)) => {
                let soa = buf.to_layout2d();
                py.detach(|| {
                    let mut out = ScreenedPoissonOutput::new();
                    field.evaluate(&soa, &mut out)?;
                    Ok::<_, CoreError>(ScreenedPoissonOutInner::F64(out))
                })
                .map_err(core_error_to_py)?
            }
            (ScreenedPoissonFieldInner::F32(field), Layout2dInner::F32(buf)) => {
                let soa = buf.to_layout2d();
                py.detach(|| {
                    let mut out = ScreenedPoissonOutput::new();
                    field.evaluate(&soa, &mut out)?;
                    Ok::<_, CoreError>(ScreenedPoissonOutInner::F32(out))
                })
                .map_err(core_error_to_py)?
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "layout dtype '{}' does not match ScreenedPoissonField2d dtype '{}'",
                    layout.dtype(),
                    self.dtype()
                )));
            }
        };
        screened_poisson_output_to_dict(py, out)
    }

    /// Evaluate unilateral density energy and descent forces inside explicit `bounds = [min_x, max_x, min_y, max_y]`.
    #[pyo3(signature = (layout, bounds), text_signature = "($self, layout, bounds)")]
    fn evaluate_in_bounds(
        &mut self,
        py: Python<'_>,
        layout: &PyLayout2d,
        bounds: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyDict>> {
        let bvec = extract_f64_1d(bounds, "bounds")?;
        if bvec.len() != 4 {
            return Err(PyValueError::new_err(format!(
                "bounds must contain exactly 4 elements [min_x, max_x, min_y, max_y], got {}",
                bvec.len()
            )));
        }
        let b_arr = [bvec[0], bvec[1], bvec[2], bvec[3]];
        let out = match (&mut self.inner, layout.inner()) {
            (ScreenedPoissonFieldInner::F64(field), Layout2dInner::F64(buf)) => {
                let soa = buf.to_layout2d();
                py.detach(|| {
                    let mut out = ScreenedPoissonOutput::new();
                    field.evaluate_in_bounds(&soa, b_arr, &mut out)?;
                    Ok::<_, CoreError>(ScreenedPoissonOutInner::F64(out))
                })
                .map_err(core_error_to_py)?
            }
            (ScreenedPoissonFieldInner::F32(field), Layout2dInner::F32(buf)) => {
                let soa = buf.to_layout2d();
                py.detach(|| {
                    let mut out = ScreenedPoissonOutput::new();
                    field.evaluate_in_bounds(&soa, b_arr, &mut out)?;
                    Ok::<_, CoreError>(ScreenedPoissonOutInner::F32(out))
                })
                .map_err(core_error_to_py)?
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "layout dtype '{}' does not match ScreenedPoissonField2d dtype '{}'",
                    layout.dtype(),
                    self.dtype()
                )));
            }
        };
        screened_poisson_output_to_dict(py, out)
    }

    /// Return total heap bytes allocated by the internal `f32` FFT grids and bitmasks.
    #[pyo3(text_signature = "($self)")]
    fn workspace_memory_bytes(&self) -> usize {
        match &self.inner {
            ScreenedPoissonFieldInner::F64(f) => f.workspace_memory_bytes(),
            ScreenedPoissonFieldInner::F32(f) => f.workspace_memory_bytes(),
        }
    }

    /// Active interior grid dimension $G$.
    #[getter]
    fn grid_size(&self) -> usize {
        match &self.inner {
            ScreenedPoissonFieldInner::F64(f) => f.config().grid_size,
            ScreenedPoissonFieldInner::F32(f) => f.config().grid_size,
        }
    }

    /// Zero-padding multiplier (`1` or `2`).
    #[getter]
    fn padding_factor(&self) -> usize {
        match &self.inner {
            ScreenedPoissonFieldInner::F64(f) => f.config().padding_factor,
            ScreenedPoissonFieldInner::F32(f) => f.config().padding_factor,
        }
    }

    /// Density kernel name (`"screened_poisson"` or `"single_gaussian"`).
    #[getter]
    fn kernel(&self) -> &'static str {
        let k = match &self.inner {
            ScreenedPoissonFieldInner::F64(f) => f.config().kernel,
            ScreenedPoissonFieldInner::F32(f) => f.config().kernel,
        };
        density_field_kernel_str(k)
    }

    /// Whether exact variational adjoint masking is enabled.
    #[getter]
    fn exact_adjoint(&self) -> bool {
        match &self.inner {
            ScreenedPoissonFieldInner::F64(f) => f.config().exact_adjoint,
            ScreenedPoissonFieldInner::F32(f) => f.config().exact_adjoint,
        }
    }

    /// Field coordinate precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        match &self.inner {
            ScreenedPoissonFieldInner::F64(_) => "float64",
            ScreenedPoissonFieldInner::F32(_) => "float32",
        }
    }

    /// Return a concise string representation of the `ScreenedPoissonField2d`.
    fn __repr__(&self) -> String {
        format!(
            "ScreenedPoissonField2d(kernel='{}', grid_size={}, padding_factor={}, exact_adjoint={}, dtype='{}')",
            self.kernel(),
            self.grid_size(),
            self.padding_factor(),
            self.exact_adjoint(),
            self.dtype()
        )
    }
}

impl_dual_solver!(ScreenedPoissonRefinerInner, ScreenedPoissonRefiner);

#[allow(clippy::too_many_arguments)]
fn build_screened_poisson_refiner_config(
    kernel: &str,
    grid_size: StrictUsize,
    padding_factor: StrictUsize,
    margin_ratio: StrictF64,
    sigma_ratio: StrictF64,
    sigma: Option<StrictF64>,
    screening_length_ratio: StrictF64,
    kappa: Option<StrictF64>,
    rho_max_factor: StrictF64,
    rho_max: Option<StrictF64>,
    exact_adjoint: bool,
    iterations: StrictUsize,
    step_size: StrictF64,
    momentum: StrictF64,
    max_step_ratio: StrictF64,
    anchor_weight: StrictF64,
    gamma_tether: StrictF64,
    recenter: bool,
    energy_tol: StrictF64,
    seed: StrictU64,
) -> PyResult<ScreenedPoissonRefinerConfig> {
    let field = build_screened_poisson_config(
        kernel,
        grid_size,
        padding_factor,
        margin_ratio,
        sigma_ratio,
        sigma,
        screening_length_ratio,
        kappa,
        rho_max_factor,
        rho_max,
        exact_adjoint,
    )?;
    let cfg = ScreenedPoissonRefinerConfig {
        field,
        iterations: iterations.0,
        step_size: step_size.0,
        momentum: momentum.0,
        max_step_ratio: max_step_ratio.0,
        anchor_weight: anchor_weight.0,
        gamma_tether: gamma_tether.0,
        recenter,
        energy_tol: energy_tol.0,
        seed: seed.0,
    };
    cfg.validate().map_err(core_error_to_py)?;
    Ok(cfg)
}

/// Cooperative Screened Poisson unilateral upper-bound density field refiner.
///
/// Supports `step()`, `run_steps()`, `run()`, and `apply_epoch_step()` with optional 1-hop edge
/// anchors and Pitfall #7 cluster centroid tethering. Releases the Python GIL during stepping.
#[pyclass(name = "ScreenedPoissonRefiner", module = "egraph._core")]
pub struct PyScreenedPoissonRefiner {
    inner: ScreenedPoissonRefinerInner,
}

#[pymethods]
impl PyScreenedPoissonRefiner {
    /// Create a standalone `ScreenedPoissonRefiner` for `num_nodes` vertices.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            num_nodes,
            *,
            kernel = "screened_poisson",
            grid_size = StrictUsize(2048),
            padding_factor = StrictUsize(2),
            margin_ratio = StrictF64(0.1),
            sigma_ratio = StrictF64(0.015),
            sigma = None,
            screening_length_ratio = StrictF64(0.20),
            kappa = None,
            rho_max_factor = StrictF64(4.0),
            rho_max = None,
            exact_adjoint = true,
            iterations = StrictUsize(20),
            step_size = StrictF64(0.25),
            momentum = StrictF64(0.5),
            max_step_ratio = StrictF64(0.05),
            anchor_weight = StrictF64(0.0),
            gamma_tether = StrictF64(0.0),
            recenter = true,
            energy_tol = StrictF64(1e-12),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(num_nodes, *, kernel='screened_poisson', grid_size=2048, padding_factor=2, margin_ratio=0.1, sigma_ratio=0.015, sigma=None, screening_length_ratio=0.2, kappa=None, rho_max_factor=4.0, rho_max=None, exact_adjoint=True, iterations=20, step_size=0.25, momentum=0.5, max_step_ratio=0.05, anchor_weight=0.0, gamma_tether=0.0, recenter=True, energy_tol=1e-12, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        num_nodes: StrictUsize,
        kernel: &str,
        grid_size: StrictUsize,
        padding_factor: StrictUsize,
        margin_ratio: StrictF64,
        sigma_ratio: StrictF64,
        sigma: Option<StrictF64>,
        screening_length_ratio: StrictF64,
        kappa: Option<StrictF64>,
        rho_max_factor: StrictF64,
        rho_max: Option<StrictF64>,
        exact_adjoint: bool,
        iterations: StrictUsize,
        step_size: StrictF64,
        momentum: StrictF64,
        max_step_ratio: StrictF64,
        anchor_weight: StrictF64,
        gamma_tether: StrictF64,
        recenter: bool,
        energy_tol: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = build_screened_poisson_refiner_config(
            kernel,
            grid_size,
            padding_factor,
            margin_ratio,
            sigma_ratio,
            sigma,
            screening_length_ratio,
            kappa,
            rho_max_factor,
            rho_max,
            exact_adjoint,
            iterations,
            step_size,
            momentum,
            max_step_ratio,
            anchor_weight,
            gamma_tether,
            recenter,
            energy_tol,
            seed,
        )?;
        let n = num_nodes.0;
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    ScreenedPoissonRefiner::<f64>::new(n, cfg).map(ScreenedPoissonRefinerInner::F64)
                }
                ScalarDtype::F32 => {
                    ScreenedPoissonRefiner::<f32>::new(n, cfg).map(ScreenedPoissonRefinerInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Create a `ScreenedPoissonRefiner` retaining 1-hop edges from `graph` for optional anchors.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            kernel = "screened_poisson",
            grid_size = StrictUsize(2048),
            padding_factor = StrictUsize(2),
            margin_ratio = StrictF64(0.1),
            sigma_ratio = StrictF64(0.015),
            sigma = None,
            screening_length_ratio = StrictF64(0.20),
            kappa = None,
            rho_max_factor = StrictF64(4.0),
            rho_max = None,
            exact_adjoint = true,
            iterations = StrictUsize(20),
            step_size = StrictF64(0.25),
            momentum = StrictF64(0.5),
            max_step_ratio = StrictF64(0.05),
            anchor_weight = StrictF64(0.0),
            gamma_tether = StrictF64(0.0),
            recenter = true,
            energy_tol = StrictF64(1e-12),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, kernel='screened_poisson', grid_size=2048, padding_factor=2, margin_ratio=0.1, sigma_ratio=0.015, sigma=None, screening_length_ratio=0.2, kappa=None, rho_max_factor=4.0, rho_max=None, exact_adjoint=True, iterations=20, step_size=0.25, momentum=0.5, max_step_ratio=0.05, anchor_weight=0.0, gamma_tether=0.0, recenter=True, energy_tol=1e-12, seed=42, dtype='float64')"
    )]
    fn from_graph(
        py: Python<'_>,
        graph: &PyCsrGraph,
        kernel: &str,
        grid_size: StrictUsize,
        padding_factor: StrictUsize,
        margin_ratio: StrictF64,
        sigma_ratio: StrictF64,
        sigma: Option<StrictF64>,
        screening_length_ratio: StrictF64,
        kappa: Option<StrictF64>,
        rho_max_factor: StrictF64,
        rho_max: Option<StrictF64>,
        exact_adjoint: bool,
        iterations: StrictUsize,
        step_size: StrictF64,
        momentum: StrictF64,
        max_step_ratio: StrictF64,
        anchor_weight: StrictF64,
        gamma_tether: StrictF64,
        recenter: bool,
        energy_tol: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = build_screened_poisson_refiner_config(
            kernel,
            grid_size,
            padding_factor,
            margin_ratio,
            sigma_ratio,
            sigma,
            screening_length_ratio,
            kappa,
            rho_max_factor,
            rho_max,
            exact_adjoint,
            iterations,
            step_size,
            momentum,
            max_step_ratio,
            anchor_weight,
            gamma_tether,
            recenter,
            energy_tol,
            seed,
        )?;
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => ScreenedPoissonRefiner::<f64>::with_graph(g, cfg)
                    .map(ScreenedPoissonRefinerInner::F64),
                ScalarDtype::F32 => ScreenedPoissonRefiner::<f32>::with_graph(g, cfg)
                    .map(ScreenedPoissonRefinerInner::F32),
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Attach or replace a coarse-level cluster centroid tether (`align_clustered_centroids`).
    #[pyo3(signature = (coarse_layout, node_to_coarse), text_signature = "($self, coarse_layout, node_to_coarse)")]
    fn set_coarse_tether(
        &mut self,
        py: Python<'_>,
        coarse_layout: &PyLayout2d,
        node_to_coarse: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let n2c = extract_u32_1d(node_to_coarse, "node_to_coarse")?;
        match (&mut self.inner, coarse_layout.inner()) {
            (ScreenedPoissonRefinerInner::F64(refiner), Layout2dInner::F64(buf)) => {
                let coarse_soa = buf.to_layout2d();
                py.detach(|| refiner.set_coarse_tether(coarse_soa, n2c))
                    .map_err(core_error_to_py)
            }
            (ScreenedPoissonRefinerInner::F32(refiner), Layout2dInner::F32(buf)) => {
                let coarse_soa = buf.to_layout2d();
                py.detach(|| refiner.set_coarse_tether(coarse_soa, n2c))
                    .map_err(core_error_to_py)
            }
            _ => Err(PyValueError::new_err(format!(
                "coarse_layout dtype '{}' does not match ScreenedPoissonRefiner dtype '{}'",
                coarse_layout.dtype(),
                self.dtype()
            ))),
        }
    }

    /// Execute one density refinement step in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` density refinement steps in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run density refinement to completion or early convergence in place on `layout`.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Execute one decoupled epoch-boundary density step in place on `layout` and return stats.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn apply_epoch_step(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
    ) -> PyResult<Py<PyDict>> {
        let target_dtype = self.inner.dtype();
        let mut soa = extract_solver_layout(layout, target_dtype)?;
        let stats = py
            .detach(|| match (&mut self.inner, &mut soa) {
                (ScreenedPoissonRefinerInner::F64(r), SolverLayout::F64(l)) => {
                    r.apply_epoch_step(l)
                }
                (ScreenedPoissonRefinerInner::F32(r), SolverLayout::F32(l)) => {
                    r.apply_epoch_step(l)
                }
                _ => Err(CoreError::invalid_input(
                    "internal solver layout dtype mismatch",
                )),
            })
            .map_err(core_error_to_py)?;
        write_back_solver_layout(layout, &soa)?;
        screened_poisson_stats_to_dict(py, &stats)
    }

    /// Return telemetry from the most recent density field evaluation as a dict.
    #[pyo3(text_signature = "($self)")]
    fn last_stats(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let st = match &self.inner {
            ScreenedPoissonRefinerInner::F64(r) => *r.last_stats(),
            ScreenedPoissonRefinerInner::F32(r) => *r.last_stats(),
        };
        screened_poisson_stats_to_dict(py, &st)
    }

    /// Return total heap bytes allocated by the underlying density field workspace.
    #[pyo3(text_signature = "($self)")]
    fn workspace_memory_bytes(&self) -> usize {
        match &self.inner {
            ScreenedPoissonRefinerInner::F64(r) => r.workspace_memory_bytes(),
            ScreenedPoissonRefinerInner::F32(r) => r.workspace_memory_bytes(),
        }
    }

    /// Most recently evaluated variational density energy.
    #[getter]
    fn last_energy(&self) -> f64 {
        match &self.inner {
            ScreenedPoissonRefinerInner::F64(r) => r.last_energy(),
            ScreenedPoissonRefinerInner::F32(r) => r.last_energy(),
        }
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of refinement iterations executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Configured maximum number of refinement iterations.
    #[getter]
    fn total_iterations(&self) -> usize {
        self.inner.solver_total_iterations()
    }

    /// Whether the refiner has reached `"converged"` or `"completed"`.
    #[getter]
    fn is_finished(&self) -> bool {
        self.inner.solver_status().is_finished()
    }

    /// Expected number of vertices in `Layout2d`.
    #[getter]
    fn num_nodes(&self) -> usize {
        match &self.inner {
            ScreenedPoissonRefinerInner::F64(r) => r.num_nodes(),
            ScreenedPoissonRefinerInner::F32(r) => r.num_nodes(),
        }
    }

    /// Refiner scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `ScreenedPoissonRefiner`.
    fn __repr__(&self) -> String {
        format!(
            "ScreenedPoissonRefiner(num_nodes={}, status='{}', iter={}/{}, dtype='{}')",
            self.num_nodes(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CoarseDistanceBackend, CoarseEdgeLengthMode, DensityFieldKernel,
        parse_coarse_distance_backend, parse_coarse_edge_length_mode, parse_density_field_kernel,
    };

    #[test]
    fn test_pure_rust_m5_enum_parsers() {
        assert_eq!(
            parse_coarse_edge_length_mode("conductance_inverse").unwrap(),
            CoarseEdgeLengthMode::ConductanceInverse
        );
        assert_eq!(
            parse_coarse_edge_length_mode("rcf").unwrap(),
            CoarseEdgeLengthMode::ResistanceCurvatureFlow
        );
        assert!(parse_coarse_edge_length_mode("invalid").is_err());

        assert_eq!(
            parse_coarse_distance_backend("rhgh").unwrap(),
            CoarseDistanceBackend::Rhgh
        );
        assert_eq!(
            parse_coarse_distance_backend("chebyshev_dog").unwrap(),
            CoarseDistanceBackend::ChebyshevDog
        );
        assert_eq!(
            parse_coarse_distance_backend("sp").unwrap(),
            CoarseDistanceBackend::ShortestPath
        );
        assert!(parse_coarse_distance_backend("unknown").is_err());

        assert_eq!(
            parse_density_field_kernel("screened_poisson").unwrap(),
            DensityFieldKernel::ScreenedPoisson
        );
        assert_eq!(
            parse_density_field_kernel("single_gaussian").unwrap(),
            DensityFieldKernel::SingleGaussian
        );
        assert!(parse_density_field_kernel("invalid_kernel").is_err());
    }
}
