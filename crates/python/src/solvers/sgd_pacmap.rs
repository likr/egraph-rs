//! `SNAP-tFDP`, `PaCMAP-Graph`, and diagonal-preconditioned mini-batch `sgd2` constructors.

use egraph::layout as egraph_layout;
use egraph_layout::{PacmapGraphConfig, Sgd2Config, SgdSolver, SnapTFdpConfig};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::array_extract::extract_f64_2d;
use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py};
use crate::neighborhood::PyDogVaradhanLandmarks;

use super::sgd::PySgdSolver;
use super::sgd_core::SgdInner;

#[allow(clippy::too_many_arguments)]
pub(super) fn build_snap_t_fdp_solver(
    py: Python<'_>,
    graph: &PyCsrGraph,
    iterations: StrictUsize,
    negatives_per_edge: StrictUsize,
    learning_rate: StrictF64,
    alpha: StrictF64,
    repulsion_scale: StrictF64,
    rho_start: StrictF64,
    rho_end: StrictF64,
    rho_power: StrictF64,
    degree_beta: StrictF64,
    bounded_attraction: bool,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<PySgdSolver> {
    let cfg = SnapTFdpConfig {
        iterations: iterations.0,
        negatives_per_edge: negatives_per_edge.0,
        learning_rate: learning_rate.0,
        alpha: alpha.0,
        repulsion_scale: repulsion_scale.0,
        rho_start: rho_start.0,
        rho_end: rho_end.0,
        rho_power: rho_power.0,
        degree_beta: degree_beta.0,
        bounded_attraction,
        seed: seed.0,
    };
    cfg.validate().map_err(core_error_to_py)?;
    let g = graph.inner();
    let inner = py
        .detach(|| match dtype {
            ScalarDtype::F64 => SgdSolver::<f64>::snap_t_fdp(g, cfg).map(SgdInner::F64),
            ScalarDtype::F32 => SgdSolver::<f32>::snap_t_fdp(g, cfg).map(SgdInner::F32),
        })
        .map_err(core_error_to_py)?;
    Ok(PySgdSolver::from_inner(inner))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_pacmap_graph_solver(
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
    landmarks: Option<&Bound<'_, PyAny>>,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<PySgdSolver> {
    let cfg = PacmapGraphConfig {
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
    cfg.validate().map_err(core_error_to_py)?;
    let g = graph.inner();
    let inner = match landmarks {
        None => py
            .detach(|| match dtype {
                ScalarDtype::F64 => SgdSolver::<f64>::pacmap_graph(g, cfg).map(SgdInner::F64),
                ScalarDtype::F32 => SgdSolver::<f32>::pacmap_graph(g, cfg).map(SgdInner::F32),
            })
            .map_err(core_error_to_py)?,
        Some(obj) if obj.is_none() => py
            .detach(|| match dtype {
                ScalarDtype::F64 => SgdSolver::<f64>::pacmap_graph(g, cfg).map(SgdInner::F64),
                ScalarDtype::F32 => SgdSolver::<f32>::pacmap_graph(g, cfg).map(SgdInner::F32),
            })
            .map_err(core_error_to_py)?,
        Some(obj) => {
            if let Ok(lm_ref) = obj.extract::<PyRef<'_, PyDogVaradhanLandmarks>>() {
                let lm = lm_ref.inner();
                py.detach(|| match dtype {
                    ScalarDtype::F64 => {
                        SgdSolver::<f64>::pacmap_graph_with_landmarks(g, lm, cfg).map(SgdInner::F64)
                    }
                    ScalarDtype::F32 => {
                        SgdSolver::<f32>::pacmap_graph_with_landmarks(g, lm, cfg).map(SgdInner::F32)
                    }
                })
                .map_err(core_error_to_py)?
            } else {
                let (flat, nrows, ncols) = extract_f64_2d(obj, "landmarks")?;
                if nrows != g.num_nodes() {
                    return Err(PyValueError::new_err(format!(
                        "landmarks row count ({nrows}) must match graph.num_nodes ({})",
                        g.num_nodes()
                    )));
                }
                py.detach(|| match dtype {
                    ScalarDtype::F64 => {
                        SgdSolver::<f64>::pacmap_graph_with_landmark_distances(g, &flat, ncols, cfg)
                            .map(SgdInner::F64)
                    }
                    ScalarDtype::F32 => {
                        SgdSolver::<f32>::pacmap_graph_with_landmark_distances(g, &flat, ncols, cfg)
                            .map(SgdInner::F32)
                    }
                })
                .map_err(core_error_to_py)?
            }
        }
    };
    Ok(PySgdSolver::from_inner(inner))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_sgd2_solver(
    py: Python<'_>,
    graph: &PyCsrGraph,
    t_max: StrictUsize,
    eps: StrictF64,
    step_scale: StrictF64,
    batch_size: StrictUsize,
    damping: StrictF64,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<PySgdSolver> {
    let cfg = Sgd2Config {
        t_max: t_max.0,
        eps: eps.0,
        step_scale: step_scale.0,
        batch_size: batch_size.0,
        damping: damping.0,
        seed: seed.0,
    };
    cfg.validate().map_err(core_error_to_py)?;
    let g = graph.inner();
    let inner = py
        .detach(|| match dtype {
            ScalarDtype::F64 => SgdSolver::<f64>::sgd2(g, cfg).map(SgdInner::F64),
            ScalarDtype::F32 => SgdSolver::<f32>::sgd2(g, cfg).map(SgdInner::F32),
        })
        .map_err(core_error_to_py)?;
    Ok(PySgdSolver::from_inner(inner))
}
