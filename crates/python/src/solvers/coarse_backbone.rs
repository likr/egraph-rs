//! Coarse Louvain area-reserved backbone initialization and coupled `PaCMAP-Graph` builder.

use egraph::{clustering as egraph_clustering, layout as egraph_layout};
use egraph_clustering::CoarseLouvainConfig;
use egraph_layout::{
    CoarseBackboneConfig, CoarseBackboneResult, CoarseBackboneStats, CoarseDistanceBackend,
    CoarseEdgeLengthMode, PacmapGraphConfig, SgdSolver,
    coarse_backbone_init as layout_coarse_backbone_init,
};
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, normalize_enum_key,
};
use crate::layout_buffers::PyLayout2d;

use super::sgd::PySgdSolver;
use super::sgd_core::SgdInner;

pub(super) fn parse_coarse_edge_length_mode(raw: &str) -> Result<CoarseEdgeLengthMode, String> {
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

pub(super) fn parse_coarse_distance_backend(raw: &str) -> Result<CoarseDistanceBackend, String> {
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
        phase_iterations: [phase1_iters.0, phase2_iters.0, phase3_iters.0],
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

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(coarse_backbone_init, m)?)?;
    Ok(())
}
