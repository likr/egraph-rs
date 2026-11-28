//! Bounded Student-t force-directed placement (`TFdpSolver`) with Policy 1A target & Option 1 backend dispatch.

use egraph::core::{Layout2d, Layout3d, LayoutNd};
use egraph::layout::neighbor::Exact as NeighborExact;
use egraph::layout::{TFdpBackend2d, TFdpBackend3d, TFdpSolver};
use egraph::neighborhood::KnnGraph;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use super::fit_field_stats::PyFieldFftStats;
use super::layout_dispatch::{
    DualTargetSolver, SolverAction, execute_target_action, validate_target_space,
};
use super::t_fdp_builders::{ResolvedTFdpBackend, TFdpInner, build_t_fdp_config_and_backend};
pub(super) use super::t_fdp_builders::{TFdpBackendChoice, build_t_fdp_backend_2d};
use crate::csr_graph::PyCsrGraph;
use crate::errors::{core_error_to_py, dimension_mismatch_err, invalid_input_err};
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, extract_optional_weights_1d, extract_u32_1d,
    solver_status_str,
};
use crate::neighborhood::PyKnnGraph;

macro_rules! dispatch_tfdp_build {
    ($py:expr, $resolved:expr, $dtype:expr, |$b:ident, $solver_ty:ident| $call:expr) => {
        $py.detach(|| match ($resolved, $dtype) {
            (ResolvedTFdpBackend::L2($b), ScalarDtype::F64) => {
                type $solver_ty = TFdpSolver<Layout2d<f64>, TFdpBackend2d>;
                $call.map(TFdpInner::L2F64)
            }
            (ResolvedTFdpBackend::L2($b), ScalarDtype::F32) => {
                type $solver_ty = TFdpSolver<Layout2d<f32>, TFdpBackend2d>;
                $call.map(TFdpInner::L2F32)
            }
            (ResolvedTFdpBackend::L3($b), ScalarDtype::F64) => {
                type $solver_ty = TFdpSolver<Layout3d<f64>, TFdpBackend3d>;
                $call.map(TFdpInner::L3F64)
            }
            (ResolvedTFdpBackend::L3($b), ScalarDtype::F32) => {
                type $solver_ty = TFdpSolver<Layout3d<f32>, TFdpBackend3d>;
                $call.map(TFdpInner::L3F32)
            }
            (ResolvedTFdpBackend::Nd($b), ScalarDtype::F64) => {
                type $solver_ty = TFdpSolver<LayoutNd<f64>, NeighborExact>;
                $call.map(TFdpInner::NdF64)
            }
            (ResolvedTFdpBackend::Nd($b), ScalarDtype::F32) => {
                type $solver_ty = TFdpSolver<LayoutNd<f32>, NeighborExact>;
                $call.map(TFdpInner::NdF32)
            }
        })
        .map_err(core_error_to_py)
    };
}

/// Cooperative `t-FDP` bounded Student-t force-directed placement solver.
#[pyclass(name = "TFdpSolver", module = "egraph._core")]
pub struct PyTFdpSolver {
    inner: TFdpInner,
}

#[pymethods]
impl PyTFdpSolver {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            target = "layout2d",
            backend = "auto",
            iterations = StrictUsize(200),
            learning_rate = StrictF64(1.0),
            momentum = StrictF64(0.8),
            theta = StrictF64(0.5),
            alpha = StrictF64(1.0),
            gamma = StrictF64(1.0),
            rho_start = StrictF64(1.0),
            rho_end = StrictF64(0.1),
            rho_power = StrictF64(1.0),
            interpolation_order = StrictUsize(1),
            intervals = StrictUsize(25),
            min_grid_dim = StrictUsize(64),
            max_grid_dim = StrictUsize(512),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, target='layout2d', backend='auto', iterations=200, learning_rate=1.0, momentum=0.8, theta=0.5, alpha=1.0, gamma=1.0, rho_start=1.0, rho_end=0.1, rho_power=1.0, interpolation_order=1, intervals=25, min_grid_dim=64, max_grid_dim=512, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        target: &str,
        backend: &str,
        iterations: StrictUsize,
        learning_rate: StrictF64,
        momentum: StrictF64,
        theta: StrictF64,
        alpha: StrictF64,
        gamma: StrictF64,
        rho_start: StrictF64,
        rho_end: StrictF64,
        rho_power: StrictF64,
        interpolation_order: StrictUsize,
        intervals: StrictUsize,
        min_grid_dim: StrictUsize,
        max_grid_dim: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let (cfg, resolved) = build_t_fdp_config_and_backend(
            target,
            backend,
            iterations.0,
            learning_rate.0,
            momentum.0,
            theta.0,
            alpha.0,
            gamma.0,
            rho_start.0,
            rho_end.0,
            rho_power.0,
            interpolation_order.0,
            intervals.0,
            min_grid_dim.0,
            max_grid_dim.0,
            seed.0,
        )?;
        let g = graph.inner();
        let inner = dispatch_tfdp_build!(py, resolved, dtype, |b, S| {
            S::new_with_backend(&*g, cfg, b)
        })?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            knn_or_indptr,
            indices = None,
            weights = None,
            *,
            target = "layout2d",
            backend = "auto",
            iterations = StrictUsize(200),
            learning_rate = StrictF64(1.0),
            momentum = StrictF64(0.8),
            theta = StrictF64(0.5),
            alpha = StrictF64(1.0),
            gamma = StrictF64(1.0),
            rho_start = StrictF64(1.0),
            rho_end = StrictF64(0.1),
            rho_power = StrictF64(1.0),
            interpolation_order = StrictUsize(1),
            intervals = StrictUsize(25),
            min_grid_dim = StrictUsize(64),
            max_grid_dim = StrictUsize(512),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(knn_or_indptr, indices=None, weights=None, *, target='layout2d', backend='auto', iterations=200, learning_rate=1.0, momentum=0.8, theta=0.5, alpha=1.0, gamma=1.0, rho_start=1.0, rho_end=0.1, rho_power=1.0, interpolation_order=1, intervals=25, min_grid_dim=64, max_grid_dim=512, seed=42, dtype='float64')"
    )]
    fn from_knn(
        py: Python<'_>,
        knn_or_indptr: &Bound<'_, PyAny>,
        indices: Option<&Bound<'_, PyAny>>,
        weights: Option<&Bound<'_, PyAny>>,
        target: &str,
        backend: &str,
        iterations: StrictUsize,
        learning_rate: StrictF64,
        momentum: StrictF64,
        theta: StrictF64,
        alpha: StrictF64,
        gamma: StrictF64,
        rho_start: StrictF64,
        rho_end: StrictF64,
        rho_power: StrictF64,
        interpolation_order: StrictUsize,
        intervals: StrictUsize,
        min_grid_dim: StrictUsize,
        max_grid_dim: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let (cfg, resolved) = build_t_fdp_config_and_backend(
            target,
            backend,
            iterations.0,
            learning_rate.0,
            momentum.0,
            theta.0,
            alpha.0,
            gamma.0,
            rho_start.0,
            rho_end.0,
            rho_power.0,
            interpolation_order.0,
            intervals.0,
            min_grid_dim.0,
            max_grid_dim.0,
            seed.0,
        )?;
        if let Ok(knn_obj) = knn_or_indptr.cast::<PyKnnGraph>() {
            if indices.is_some_and(|o| !o.is_none()) || weights.is_some_and(|o| !o.is_none()) {
                return Err(PyTypeError::new_err(
                    "TFdpSolver.from_knn(knn, ...) does not accept indices or weights when given a KnnGraph",
                ));
            }
            let knn_borrowed = knn_obj.borrow();
            let knn_ref = knn_borrowed.inner();
            let inner = dispatch_tfdp_build!(py, resolved, dtype, |b, S| {
                S::from_knn_with_backend(knn_ref, cfg, b)
            })?;
            return Ok(Self { inner });
        }
        let Some(indices_obj) = indices.filter(|o| !o.is_none()) else {
            return Err(PyTypeError::new_err(
                "TFdpSolver.from_knn expects either a KnnGraph or (indptr, indices, weights=None)",
            ));
        };
        let indptr_vec = extract_u32_1d(knn_or_indptr, "indptr")?;
        if indptr_vec.is_empty() {
            return Err(invalid_input_err(
                "indptr must have length num_nodes + 1 (>= 1)",
            ));
        }
        let num_nodes = indptr_vec.len() - 1;
        let indices_vec = extract_u32_1d(indices_obj, "indices")?;
        if indices_vec.len() != (*indptr_vec.last().unwrap_or(&0) as usize) {
            return Err(dimension_mismatch_err(
                "indptr[-1] does not match indices length",
            ));
        }
        let distances_vec = match extract_optional_weights_1d(weights, "weights")? {
            Some(w) => w,
            None => vec![0.0_f64; indices_vec.len()],
        };
        let knn = KnnGraph::new_validated(num_nodes, indptr_vec, indices_vec, distances_vec)
            .map_err(core_error_to_py)?;
        let inner = dispatch_tfdp_build!(py, resolved, dtype, |b, S| {
            S::from_knn_with_backend(&knn, cfg, b)
        })?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn validate_target(&self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut none_lock = None;
        validate_target_space(&self.inner, &mut none_lock, py, layout)
    }

    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        let mut none_lock = None;
        execute_target_action(
            &mut self.inner,
            &mut none_lock,
            py,
            layout,
            SolverAction::Step,
        )
    }

    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyAny>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        let mut none_lock = None;
        execute_target_action(
            &mut self.inner,
            &mut none_lock,
            py,
            layout,
            SolverAction::RunSteps(steps.0),
        )
    }

    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        let mut none_lock = None;
        execute_target_action(
            &mut self.inner,
            &mut none_lock,
            py,
            layout,
            SolverAction::Run,
        )
    }

    fn last_fft_stats(&self) -> Option<PyFieldFftStats> {
        match &self.inner {
            TFdpInner::L2F64(s) => s.last_fft_stats().map(PyFieldFftStats::from_core),
            TFdpInner::L2F32(s) => s.last_fft_stats().map(PyFieldFftStats::from_core),
            _ => None,
        }
    }

    #[getter]
    fn target(&self) -> &'static str {
        self.inner.target_kind().as_str()
    }
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }
    #[getter]
    fn total_iterations(&self) -> Option<usize> {
        self.inner.solver_total_iterations()
    }
    #[getter]
    fn is_finished(&self) -> bool {
        self.inner.solver_status().is_finished()
    }
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }
    #[getter]
    fn num_nodes(&self) -> usize {
        match &self.inner {
            TFdpInner::L2F64(s) => s.num_nodes(),
            TFdpInner::L2F32(s) => s.num_nodes(),
            TFdpInner::L3F64(s) => s.num_nodes(),
            TFdpInner::L3F32(s) => s.num_nodes(),
            TFdpInner::NdF64(s) => s.num_nodes(),
            TFdpInner::NdF32(s) => s.num_nodes(),
        }
    }
    #[getter]
    fn current_rho(&self) -> f64 {
        match &self.inner {
            TFdpInner::L2F64(s) => s.current_rho(),
            TFdpInner::L2F32(s) => s.current_rho(),
            TFdpInner::L3F64(s) => s.current_rho(),
            TFdpInner::L3F32(s) => s.current_rho(),
            TFdpInner::NdF64(s) => s.current_rho(),
            TFdpInner::NdF32(s) => s.current_rho(),
        }
    }
    #[getter]
    fn num_edges(&self) -> usize {
        match &self.inner {
            TFdpInner::L2F64(s) => s.num_edges(),
            TFdpInner::L2F32(s) => s.num_edges(),
            TFdpInner::L3F64(s) => s.num_edges(),
            TFdpInner::L3F32(s) => s.num_edges(),
            TFdpInner::NdF64(s) => s.num_edges(),
            TFdpInner::NdF32(s) => s.num_edges(),
        }
    }
    #[getter]
    fn total_fft_2d_calls(&self) -> u64 {
        match &self.inner {
            TFdpInner::L2F64(s) => s.total_fft_2d_calls(),
            TFdpInner::L2F32(s) => s.total_fft_2d_calls(),
            _ => 0,
        }
    }
    fn __repr__(&self) -> String {
        format!(
            "TFdpSolver(num_nodes={}, target='{}', status='{}', iteration={}/{:?}, dtype='{}')",
            self.num_nodes(),
            self.target(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}
