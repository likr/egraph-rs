//! Stress Majorization and Kamada-Kawai layout solver bindings.

use egraph::layout as egraph_layout;
use egraph_layout::{
    KamadaKawaiConfig, KamadaKawaiSolver, StressMajorizationConfig, StressMajorizationSolver,
};
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictUsize, core_error_to_py, solver_status_str,
};
use crate::layout_buffers::{Layout2dInner, PyLayout2d};
use crate::solver_common::{
    DualIterativeSolver, SolverAction, execute_solver_action, impl_dual_solver,
    layout_f32_to_f64_checked, layout_f64_to_f32_checked,
};

use super::distance_extract::extract_distance_matrix_2d;

impl_dual_solver!(StressMajorizationInner, StressMajorizationSolver);
impl_dual_solver!(KamadaKawaiInner, KamadaKawaiSolver);

/// Cooperative Gansner-Koren-North (2004) Stress Majorization layout solver backed by Jacobi-PCG.
///
/// Supports both `"float64"` and `"float32"` `Layout2d` and releases the Python GIL during stepping.
#[pyclass(name = "StressMajorizationSolver", module = "egraph._core")]
pub struct PyStressMajorizationSolver {
    inner: StressMajorizationInner,
}

#[pymethods]
impl PyStressMajorizationSolver {
    /// Construct a `StressMajorizationSolver` from a `CsrGraph` using all-pairs shortest paths.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, *, epsilon = StrictF64(1e-4), max_iterations = StrictUsize(100), pcg_tolerance = StrictF64(1e-6), pcg_max_iterations = None, min_norm = StrictF64(1e-4), dtype = ScalarDtype::F64),
        text_signature = "(graph, *, epsilon=1e-4, max_iterations=100, pcg_tolerance=1e-6, pcg_max_iterations=None, min_norm=1e-4, dtype='float64')"
    )]
    fn from_graph(
        py: Python<'_>,
        graph: &PyCsrGraph,
        epsilon: StrictF64,
        max_iterations: StrictUsize,
        pcg_tolerance: StrictF64,
        pcg_max_iterations: Option<StrictUsize>,
        min_norm: StrictF64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = StressMajorizationConfig {
            epsilon: epsilon.0,
            max_iterations: max_iterations.0,
            pcg_tolerance: pcg_tolerance.0,
            pcg_max_iterations: pcg_max_iterations.map(|v| v.0),
            min_norm: min_norm.0,
        };
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => StressMajorizationSolver::<f64>::from_graph(g, cfg)
                    .map(StressMajorizationInner::F64),
                ScalarDtype::F32 => StressMajorizationSolver::<f32>::from_graph(g, cfg)
                    .map(StressMajorizationInner::F32),
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Construct a `StressMajorizationSolver` from an `(N, N)` distance matrix (where `inf` marks unreachable pairs).
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (distances, *, epsilon = StrictF64(1e-4), max_iterations = StrictUsize(100), pcg_tolerance = StrictF64(1e-6), pcg_max_iterations = None, min_norm = StrictF64(1e-4), dtype = ScalarDtype::F64),
        text_signature = "(distances, *, epsilon=1e-4, max_iterations=100, pcg_tolerance=1e-6, pcg_max_iterations=None, min_norm=1e-4, dtype='float64')"
    )]
    fn from_distance_matrix(
        py: Python<'_>,
        distances: &Bound<'_, PyAny>,
        epsilon: StrictF64,
        max_iterations: StrictUsize,
        pcg_tolerance: StrictF64,
        pcg_max_iterations: Option<StrictUsize>,
        min_norm: StrictF64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let (dists_flat, n) = extract_distance_matrix_2d(distances)?;
        let cfg = StressMajorizationConfig {
            epsilon: epsilon.0,
            max_iterations: max_iterations.0,
            pcg_tolerance: pcg_tolerance.0,
            pcg_max_iterations: pcg_max_iterations.map(|v| v.0),
            min_norm: min_norm.0,
        };
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    StressMajorizationSolver::<f64>::from_distance_matrix(n, &dists_flat, cfg)
                        .map(StressMajorizationInner::F64)
                }
                ScalarDtype::F32 => {
                    StressMajorizationSolver::<f32>::from_distance_matrix(n, &dists_flat, cfg)
                        .map(StressMajorizationInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute one stress majorization step in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` stress majorization iterations in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run stress majorization to completion or convergence in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Evaluate the total weighted stress for `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn compute_stress(&self, py: Python<'_>, layout: &PyLayout2d) -> PyResult<f64> {
        py.detach(|| match (&self.inner, layout.inner()) {
            (StressMajorizationInner::F64(s), Layout2dInner::F64(buf)) => {
                s.compute_stress(&buf.to_layout2d())
            }
            (StressMajorizationInner::F32(s), Layout2dInner::F32(buf)) => {
                s.compute_stress(&buf.to_layout2d())
            }
            (StressMajorizationInner::F64(s), Layout2dInner::F32(buf)) => {
                let soa64 = layout_f32_to_f64_checked(&buf.to_layout2d())?;
                s.compute_stress(&soa64)
            }
            (StressMajorizationInner::F32(s), Layout2dInner::F64(buf)) => {
                let soa32 = layout_f64_to_f32_checked(&buf.to_layout2d())?;
                s.compute_stress(&soa32)
            }
        })
        .map_err(core_error_to_py)
    }

    /// Stress recorded after the most recent executed step, or `None` before stepping.
    #[getter]
    fn last_stress(&self) -> Option<f64> {
        match &self.inner {
            StressMajorizationInner::F64(s) => s.last_stress(),
            StressMajorizationInner::F32(s) => s.last_stress(),
        }
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

    /// Configured maximum number of iterations (`max_iterations`).
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
        match &self.inner {
            StressMajorizationInner::F64(s) => s.num_nodes(),
            StressMajorizationInner::F32(s) => s.num_nodes(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `StressMajorizationSolver`.
    fn __repr__(&self) -> String {
        format!(
            "StressMajorizationSolver(num_nodes={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}

/// Cooperative Kamada-Kawai (1989) spring layout solver with Hessian singularity fallback.
///
/// Supports both `"float64"` and `"float32"` `Layout2d` and releases the Python GIL during stepping.
#[pyclass(name = "KamadaKawaiSolver", module = "egraph._core")]
pub struct PyKamadaKawaiSolver {
    inner: KamadaKawaiInner,
}

#[pymethods]
impl PyKamadaKawaiSolver {
    /// Construct a `KamadaKawaiSolver` from a `CsrGraph` using all-pairs shortest paths.
    #[staticmethod]
    #[pyo3(
        signature = (graph, *, eps = StrictF64(1e-2), max_inner_iters = StrictUsize(50), max_outer_iters = None, dtype = ScalarDtype::F64),
        text_signature = "(graph, *, eps=1e-2, max_inner_iters=50, max_outer_iters=None, dtype='float64')"
    )]
    fn from_graph(
        py: Python<'_>,
        graph: &PyCsrGraph,
        eps: StrictF64,
        max_inner_iters: StrictUsize,
        max_outer_iters: Option<StrictUsize>,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = KamadaKawaiConfig {
            eps: eps.0,
            max_inner_iters: max_inner_iters.0,
            max_iterations: max_outer_iters.map(|v| v.0),
        };
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    KamadaKawaiSolver::<f64>::from_graph(g, cfg).map(KamadaKawaiInner::F64)
                }
                ScalarDtype::F32 => {
                    KamadaKawaiSolver::<f32>::from_graph(g, cfg).map(KamadaKawaiInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Construct a `KamadaKawaiSolver` from an `(N, N)` distance matrix (where `inf` marks unreachable pairs).
    #[staticmethod]
    #[pyo3(
        signature = (distances, *, eps = StrictF64(1e-2), max_inner_iters = StrictUsize(50), max_outer_iters = None, dtype = ScalarDtype::F64),
        text_signature = "(distances, *, eps=1e-2, max_inner_iters=50, max_outer_iters=None, dtype='float64')"
    )]
    fn from_distance_matrix(
        py: Python<'_>,
        distances: &Bound<'_, PyAny>,
        eps: StrictF64,
        max_inner_iters: StrictUsize,
        max_outer_iters: Option<StrictUsize>,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let (dists_flat, n) = extract_distance_matrix_2d(distances)?;
        let cfg = KamadaKawaiConfig {
            eps: eps.0,
            max_inner_iters: max_inner_iters.0,
            max_iterations: max_outer_iters.map(|v| v.0),
        };
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    KamadaKawaiSolver::<f64>::from_distance_matrix(n, &dists_flat, cfg)
                        .map(KamadaKawaiInner::F64)
                }
                ScalarDtype::F32 => {
                    KamadaKawaiSolver::<f32>::from_distance_matrix(n, &dists_flat, cfg)
                        .map(KamadaKawaiInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute one node-relaxation step in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` node-relaxation steps in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run Kamada-Kawai optimization to completion or convergence in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of outer steps executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Configured outer iteration cap (`max_outer_iters` or `(10 * n).max(100)`).
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
        match &self.inner {
            KamadaKawaiInner::F64(s) => s.num_nodes(),
            KamadaKawaiInner::F32(s) => s.num_nodes(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `KamadaKawaiSolver`.
    fn __repr__(&self) -> String {
        format!(
            "KamadaKawaiSolver(num_nodes={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}
