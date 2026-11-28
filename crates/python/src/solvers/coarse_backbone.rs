//! Coarse Louvain area-reserved backbone initialization (`CoarseBackbone`) and coupled `PaCMAP-Graph` builder.

use std::sync::Arc;

use egraph::core::{CsrGraph, LayoutGenerator};
use egraph::layout::{
    CoarseBackbone, CoarseBackboneConfig, PacmapGraphConfig,
    pacmap_graph_with_backbone as layout_pacmap_graph_with_backbone,
};
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::errors::core_error_to_py;
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictU64, StrictUsize};
use crate::layout_buffers::PyLayout2d;

use super::coarse_backbone_types::{PyCoarseBackboneResult, build_coarse_backbone_config};
#[cfg(test)]
pub(super) use super::coarse_backbone_types::{
    parse_coarse_distance_backend, parse_coarse_edge_length_mode,
};
use super::pacmap_graph::{PacmapGraphInner, PyPacmapGraphSolver};

/// Coarse Louvain edge-length-first area reservation backbone generator (`LayoutGenerator`).
#[pyclass(name = "CoarseBackbone", module = "egraph._core")]
pub struct PyCoarseBackbone {
    graph: Arc<CsrGraph<f64>>,
    config: CoarseBackboneConfig,
    dtype: ScalarDtype,
}

#[pymethods]
impl PyCoarseBackbone {
    #[new]
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
    fn new(
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
    ) -> PyResult<Self> {
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
        Ok(Self {
            graph: Arc::new(graph.inner().into_owned()),
            config: cfg,
            dtype,
        })
    }

    #[pyo3(signature = (), text_signature = "($self)")]
    fn generate(&self, py: Python<'_>) -> PyResult<PyCoarseBackboneResult> {
        let g = Arc::clone(&self.graph);
        let cfg = self.config.clone();
        match self.dtype {
            ScalarDtype::F64 => {
                let res = py
                    .detach(|| CoarseBackbone::<f64, f64>::from_config(&*g, cfg).generate())
                    .map_err(core_error_to_py)?;
                PyCoarseBackboneResult::from_f64(py, res)
            }
            ScalarDtype::F32 => {
                let res = py
                    .detach(|| CoarseBackbone::<f32, f64>::from_config(&*g, cfg).generate())
                    .map_err(core_error_to_py)?;
                PyCoarseBackboneResult::from_f32(py, res)
            }
        }
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        self.graph.num_nodes()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.dtype.as_str()
    }

    fn __repr__(&self) -> String {
        format!(
            "CoarseBackbone(num_nodes={}, dtype='{}')",
            self.num_nodes(),
            self.dtype()
        )
    }
}

/// Run coarse Louvain edge-length-first area reservation and non-SPD backbone initialization.
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
) -> PyResult<PyCoarseBackboneResult> {
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
    match dtype {
        ScalarDtype::F64 => {
            let res = py
                .detach(|| CoarseBackbone::<f64, f64>::from_config(&*g, cfg).generate())
                .map_err(core_error_to_py)?;
            PyCoarseBackboneResult::from_f64(py, res)
        }
        ScalarDtype::F32 => {
            let res = py
                .detach(|| CoarseBackbone::<f32, f64>::from_config(&*g, cfg).generate())
                .map_err(core_error_to_py)?;
            PyCoarseBackboneResult::from_f32(py, res)
        }
    }
}

/// Construct a 3-tier `PaCMAP-Graph` solver coupled with a coarse Louvain area-reserved backbone.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (graph, *, n_neighbors = StrictUsize(10), mn_ratio = StrictF64(0.5), fp_ratio = StrictF64(2.0), phase_iterations = (StrictUsize(100), StrictUsize(100), StrictUsize(250)), learning_rate = StrictF64(1.0), keep_all_1hop = false, max_1hop_per_node = None, degree_weight_beta = StrictF64(0.0), num_landmarks = StrictUsize(16), max_coarse_nodes = StrictUsize(10_000), resolution = StrictF64(1.0), resolution_decay = StrictF64(0.5), max_passes = StrictUsize(100), min_modularity_gain = StrictF64(1e-7), max_outer_rounds = StrictUsize(16), edge_length_mode = "conductance_inverse", distance_backend = "rhgh", alpha = StrictF64(0.75), c0 = StrictF64(0.5), kappa = StrictF64(1.0), gamma_w = StrictF64(0.5), gamma_m = StrictF64(0.5), mu_ov = StrictF64(1.0), m_gap = StrictF64(0.1), coarse_pivots = StrictUsize(64), coarse_neg_per_node = StrictUsize(16), coarse_t_max = StrictUsize(80), rhgh_gamma = None, rhgh_p = None, rcf_steps = StrictUsize(2), rcf_eta = StrictF64(0.2), jitter_scale = StrictF64(0.5), gamma_tether = StrictF64(0.5), target_std = Some(StrictF64(1.0)), robust_quantile = StrictF64(0.95), seed = StrictU64(42), dtype = ScalarDtype::F64),
    text_signature = "(graph, *, n_neighbors=10, mn_ratio=0.5, fp_ratio=2.0, phase_iterations=(100, 100, 250), learning_rate=1.0, keep_all_1hop=False, max_1hop_per_node=None, degree_weight_beta=0.0, num_landmarks=16, max_coarse_nodes=10000, resolution=1.0, resolution_decay=0.5, max_passes=100, min_modularity_gain=1e-7, max_outer_rounds=16, edge_length_mode='conductance_inverse', distance_backend='rhgh', alpha=0.75, c0=0.5, kappa=1.0, gamma_w=0.5, gamma_m=0.5, mu_ov=1.0, m_gap=0.1, coarse_pivots=64, coarse_neg_per_node=16, coarse_t_max=80, rhgh_gamma=None, rhgh_p=None, rcf_steps=2, rcf_eta=0.2, jitter_scale=0.5, gamma_tether=0.5, target_std=1.0, robust_quantile=0.95, seed=42, dtype='float64')"
)]
fn pacmap_graph_with_backbone(
    py: Python<'_>,
    graph: &PyCsrGraph,
    n_neighbors: StrictUsize,
    mn_ratio: StrictF64,
    fp_ratio: StrictF64,
    phase_iterations: (StrictUsize, StrictUsize, StrictUsize),
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
) -> PyResult<(PyPacmapGraphSolver, PyLayout2d, PyCoarseBackboneResult)> {
    let pm_cfg = PacmapGraphConfig::new()
        .with_n_neighbors(n_neighbors.0)
        .with_mn_ratio(mn_ratio.0)
        .with_fp_ratio(fp_ratio.0)
        .with_phase_iterations([
            phase_iterations.0.0,
            phase_iterations.1.0,
            phase_iterations.2.0,
        ])
        .with_learning_rate(learning_rate.0)
        .with_keep_all_1hop(keep_all_1hop)
        .with_max_1hop_per_node(max_1hop_per_node.map(|v| v.0))
        .with_degree_weight_beta(degree_weight_beta.0)
        .with_num_landmarks(num_landmarks.0)
        .with_seed(seed.0);
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
                .detach(|| layout_pacmap_graph_with_backbone::<f64, f64>(&*g, pm_cfg, &bb_cfg))
                .map_err(core_error_to_py)?;
            Ok((
                PyPacmapGraphSolver::from_inner(PacmapGraphInner::L2F64(solver)),
                PyLayout2d::from_f64_layout(&init_layout),
                PyCoarseBackboneResult::from_f64(py, bb)?,
            ))
        }
        ScalarDtype::F32 => {
            let (solver, init_layout, bb) = py
                .detach(|| layout_pacmap_graph_with_backbone::<f32, f64>(&*g, pm_cfg, &bb_cfg))
                .map_err(core_error_to_py)?;
            Ok((
                PyPacmapGraphSolver::from_inner(PacmapGraphInner::L2F32(solver)),
                PyLayout2d::from_f32_layout(&init_layout),
                PyCoarseBackboneResult::from_f32(py, bb)?,
            ))
        }
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    super::coarse_backbone_types::register(m)?;
    m.add_class::<PyCoarseBackbone>()?;
    m.add_function(wrap_pyfunction!(coarse_backbone_init, m)?)?;
    m.add_function(wrap_pyfunction!(pacmap_graph_with_backbone, m)?)?;
    Ok(())
}
