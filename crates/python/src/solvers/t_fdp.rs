//! Bounded Student-t force-directed placement (`TFdpSolver`) and shared field layout inner helpers.

use egraph::{
    layout as egraph_layout, neighborhood as egraph_neighborhood, spatial as egraph_spatial,
};
use egraph_layout::{FieldLayoutSolver, FieldRepulsionBackend, RhoSchedule, TFdpConfig};
use egraph_neighborhood::KnnGraph;
use egraph_spatial::{IbFftConfig, InterpolationOrder};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::array_extract::extract_usize_1d;
use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, extract_optional_weights_1d,
    extract_u32_1d, normalize_enum_key, solver_status_str,
};
use crate::layout_buffers::PyLayout2d;
use crate::neighborhood::PyKnnGraph;
use crate::solver_common::{
    DualIterativeSolver, SolverAction, execute_solver_action, impl_dual_solver,
};

impl_dual_solver!(FieldLayoutInner, FieldLayoutSolver);

impl FieldLayoutInner {
    pub(super) fn num_nodes(&self) -> usize {
        match self {
            Self::F64(s) => s.num_nodes(),
            Self::F32(s) => s.num_nodes(),
        }
    }

    pub(super) fn current_rho(&self) -> f64 {
        match self {
            Self::F64(s) => s.current_rho(),
            Self::F32(s) => s.current_rho(),
        }
    }

    pub(super) fn num_edges(&self) -> usize {
        match self {
            Self::F64(s) => s.num_edges(),
            Self::F32(s) => s.num_edges(),
        }
    }

    pub(super) fn total_fft_2d_calls(&self) -> u64 {
        match self {
            Self::F64(s) => s.total_fft_2d_calls(),
            Self::F32(s) => s.total_fft_2d_calls(),
        }
    }

    pub(super) fn ab(&self) -> Option<(f64, f64)> {
        match self {
            Self::F64(s) => s.ab(),
            Self::F32(s) => s.ab(),
        }
    }
}

pub(super) fn parse_field_backend(raw: &str) -> Result<FieldRepulsionBackend, String> {
    match normalize_enum_key(raw).as_str() {
        "ibfft" | "fft" => Ok(FieldRepulsionBackend::IbFft),
        "barneshut" | "bh" => Ok(FieldRepulsionBackend::BarnesHut),
        "exact" => Ok(FieldRepulsionBackend::Exact),
        _ => Err(format!(
            "unknown field repulsion backend {raw:?}; expected 'ibfft', 'barnes_hut', or 'exact'"
        )),
    }
}

pub(super) fn build_ibfft_config(
    ibfft_order: usize,
    ibfft_intervals: usize,
    ibfft_min_grid_dim: usize,
    ibfft_max_grid_dim: usize,
) -> PyResult<IbFftConfig> {
    let order = InterpolationOrder::from_order(ibfft_order).map_err(core_error_to_py)?;
    Ok(IbFftConfig {
        order,
        intervals: ibfft_intervals,
        min_grid_dim: ibfft_min_grid_dim,
        max_grid_dim: ibfft_max_grid_dim,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_t_fdp_config(
    backend: &str,
    total_iterations: usize,
    learning_rate: f64,
    momentum: f64,
    theta: f64,
    alpha: f64,
    gamma: f64,
    rho_start: f64,
    rho_end: f64,
    rho_power: f64,
    ibfft_order: usize,
    ibfft_intervals: usize,
    ibfft_min_grid_dim: usize,
    ibfft_max_grid_dim: usize,
    seed: u64,
) -> PyResult<TFdpConfig> {
    let parsed_backend = parse_field_backend(backend).map_err(PyValueError::new_err)?;
    let ibfft = build_ibfft_config(
        ibfft_order,
        ibfft_intervals,
        ibfft_min_grid_dim,
        ibfft_max_grid_dim,
    )?;
    let cfg = TFdpConfig {
        backend: parsed_backend,
        total_iterations,
        learning_rate,
        momentum,
        theta,
        alpha,
        gamma,
        rho: RhoSchedule {
            start_weight: rho_start,
            end_weight: rho_end,
            power: rho_power,
        },
        ibfft,
        seed,
    };
    cfg.validate().map_err(core_error_to_py)?;
    Ok(cfg)
}

/// Cooperative `t-FDP` bounded Student-t force-directed placement solver.
///
/// Supports `"ibfft"` (default), `"barnes_hut"`, and `"exact"` repulsion backends in both
/// `"float64"` and `"float32"` precision, releasing the Python GIL during stepping.
#[pyclass(name = "TFdpSolver", module = "egraph._core")]
pub struct PyTFdpSolver {
    inner: FieldLayoutInner,
}

#[pymethods]
impl PyTFdpSolver {
    /// Construct a `TFdpSolver` directly from a `CsrGraph`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            backend = "ibfft",
            total_iterations = StrictUsize(200),
            learning_rate = StrictF64(1.0),
            momentum = StrictF64(0.8),
            theta = StrictF64(0.5),
            alpha = StrictF64(1.0),
            gamma = StrictF64(1.0),
            rho_start = StrictF64(1.0),
            rho_end = StrictF64(0.1),
            rho_power = StrictF64(1.0),
            ibfft_order = StrictUsize(1),
            ibfft_intervals = StrictUsize(25),
            ibfft_min_grid_dim = StrictUsize(64),
            ibfft_max_grid_dim = StrictUsize(512),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, backend='ibfft', total_iterations=200, learning_rate=1.0, momentum=0.8, theta=0.5, alpha=1.0, gamma=1.0, rho_start=1.0, rho_end=0.1, rho_power=1.0, ibfft_order=1, ibfft_intervals=25, ibfft_min_grid_dim=64, ibfft_max_grid_dim=512, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        backend: &str,
        total_iterations: StrictUsize,
        learning_rate: StrictF64,
        momentum: StrictF64,
        theta: StrictF64,
        alpha: StrictF64,
        gamma: StrictF64,
        rho_start: StrictF64,
        rho_end: StrictF64,
        rho_power: StrictF64,
        ibfft_order: StrictUsize,
        ibfft_intervals: StrictUsize,
        ibfft_min_grid_dim: StrictUsize,
        ibfft_max_grid_dim: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = build_t_fdp_config(
            backend,
            total_iterations.0,
            learning_rate.0,
            momentum.0,
            theta.0,
            alpha.0,
            gamma.0,
            rho_start.0,
            rho_end.0,
            rho_power.0,
            ibfft_order.0,
            ibfft_intervals.0,
            ibfft_min_grid_dim.0,
            ibfft_max_grid_dim.0,
            seed.0,
        )?;
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    FieldLayoutSolver::<f64>::t_fdp(g, cfg).map(FieldLayoutInner::F64)
                }
                ScalarDtype::F32 => {
                    FieldLayoutSolver::<f32>::t_fdp(g, cfg).map(FieldLayoutInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Construct a `TFdpSolver` from either a `KnnGraph` or CSR `(indptr, indices, weights=None)`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            knn_or_indptr,
            indices = None,
            weights = None,
            *,
            backend = "ibfft",
            total_iterations = StrictUsize(200),
            learning_rate = StrictF64(1.0),
            momentum = StrictF64(0.8),
            theta = StrictF64(0.5),
            alpha = StrictF64(1.0),
            gamma = StrictF64(1.0),
            rho_start = StrictF64(1.0),
            rho_end = StrictF64(0.1),
            rho_power = StrictF64(1.0),
            ibfft_order = StrictUsize(1),
            ibfft_intervals = StrictUsize(25),
            ibfft_min_grid_dim = StrictUsize(64),
            ibfft_max_grid_dim = StrictUsize(512),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(knn_or_indptr, indices=None, weights=None, *, backend='ibfft', total_iterations=200, learning_rate=1.0, momentum=0.8, theta=0.5, alpha=1.0, gamma=1.0, rho_start=1.0, rho_end=0.1, rho_power=1.0, ibfft_order=1, ibfft_intervals=25, ibfft_min_grid_dim=64, ibfft_max_grid_dim=512, seed=42, dtype='float64')"
    )]
    fn from_knn(
        py: Python<'_>,
        knn_or_indptr: &Bound<'_, PyAny>,
        indices: Option<&Bound<'_, PyAny>>,
        weights: Option<&Bound<'_, PyAny>>,
        backend: &str,
        total_iterations: StrictUsize,
        learning_rate: StrictF64,
        momentum: StrictF64,
        theta: StrictF64,
        alpha: StrictF64,
        gamma: StrictF64,
        rho_start: StrictF64,
        rho_end: StrictF64,
        rho_power: StrictF64,
        ibfft_order: StrictUsize,
        ibfft_intervals: StrictUsize,
        ibfft_min_grid_dim: StrictUsize,
        ibfft_max_grid_dim: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = build_t_fdp_config(
            backend,
            total_iterations.0,
            learning_rate.0,
            momentum.0,
            theta.0,
            alpha.0,
            gamma.0,
            rho_start.0,
            rho_end.0,
            rho_power.0,
            ibfft_order.0,
            ibfft_intervals.0,
            ibfft_min_grid_dim.0,
            ibfft_max_grid_dim.0,
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
            let inner = py
                .detach(|| match dtype {
                    ScalarDtype::F64 => FieldLayoutSolver::<f64>::t_fdp_from_knn(knn_ref, cfg)
                        .map(FieldLayoutInner::F64),
                    ScalarDtype::F32 => FieldLayoutSolver::<f32>::t_fdp_from_knn(knn_ref, cfg)
                        .map(FieldLayoutInner::F32),
                })
                .map_err(core_error_to_py)?;
            return Ok(Self { inner });
        }

        let Some(indices_obj) = indices.filter(|o| !o.is_none()) else {
            return Err(PyTypeError::new_err(
                "TFdpSolver.from_knn expects either a KnnGraph or (indptr, indices, weights=None)",
            ));
        };
        let indptr_vec = extract_usize_1d(knn_or_indptr, "indptr")?;
        if indptr_vec.is_empty() {
            return Err(PyValueError::new_err(
                "indptr must have length num_nodes + 1 (>= 1)",
            ));
        }
        let num_nodes = indptr_vec.len() - 1;
        let indices_vec = extract_u32_1d(indices_obj, "indices")?;
        let distances_vec = match extract_optional_weights_1d(weights, "weights")? {
            Some(w) => w,
            None => vec![0.0_f64; indices_vec.len()],
        };

        let inner = py
            .detach(|| {
                let knn =
                    KnnGraph::new_validated(num_nodes, indptr_vec, indices_vec, distances_vec)?;
                match dtype {
                    ScalarDtype::F64 => FieldLayoutSolver::<f64>::t_fdp_from_knn(&knn, cfg)
                        .map(FieldLayoutInner::F64),
                    ScalarDtype::F32 => FieldLayoutSolver::<f32>::t_fdp_from_knn(&knn, cfg)
                        .map(FieldLayoutInner::F32),
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute a single optimization step in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` optimization iterations in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run optimization to completion in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of iterations executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Total scheduled iterations.
    #[getter]
    fn total_iterations(&self) -> usize {
        self.inner.solver_total_iterations()
    }

    /// Whether the solver has reached `"converged"` or `"completed"`.
    #[getter]
    fn is_finished(&self) -> bool {
        self.inner.solver_status().is_finished()
    }

    /// Expected number of vertices in `Layout2d`.
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Current repulsion schedule multiplier `rho(t)`.
    #[getter]
    fn current_rho(&self) -> f64 {
        self.inner.current_rho()
    }

    /// Number of attractive edges in the `TFdpSolver`.
    #[getter]
    fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    /// Cumulative number of 2D FFT transforms executed across all steps.
    #[getter]
    fn total_fft_2d_calls(&self) -> u64 {
        self.inner.total_fft_2d_calls()
    }

    /// Return a concise string representation of the `TFdpSolver`.
    fn __repr__(&self) -> String {
        format!(
            "TFdpSolver(num_nodes={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}
