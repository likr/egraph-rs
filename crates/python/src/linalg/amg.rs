//! Algebraic Multigrid Preconditioned Conjugate Gradient (`AmgPcgSolver`) Python bindings.

use egraph::linalg::{AmgPcgSolver, AmgPcgWorkspace};
use numpy::PyArray1;
use pyo3::prelude::*;

use super::amg_helpers::{
    build_amg_pcg_config, create_amg_from_csr, create_amg_from_csr_matrix,
    create_amg_from_laplacian, create_amg_from_regularized_laplacian, solve_amg, solve_batch_amg,
    start_solve_amg,
};
use super::amg_results::{PyAmgBatchSolveResult, PyAmgSolveResult};
use super::matrix::PyCsrMatrix;
use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{StrictF64, StrictU64, StrictUsize, core_error_to_py, solver_status_str};

/// Hardened single-threaded Algebraic Multigrid Preconditioned Conjugate Gradient (`AmgPcgSolver`)
/// supporting symmetric CSR matrices, regularized/shifted graph Laplacians, batched RHS, and
/// cooperative stepping.
#[pyclass(name = "AmgPcgSolver", module = "egraph._core")]
pub struct PyAmgPcgSolver {
    inner: AmgPcgSolver,
    ws: AmgPcgWorkspace,
}

#[pymethods]
impl PyAmgPcgSolver {
    /// Construct an `AmgPcgSolver` directly from raw CSR buffers `(num_nodes, indptr, indices, values)`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (num_nodes, indptr, indices, values, *, rel_tol = StrictF64(1e-8), abs_tol = StrictF64(0.0), max_iterations = StrictUsize(500), theta = StrictF64(0.25), max_levels = StrictUsize(32), max_coarse_size = StrictUsize(64), pre_sweeps = StrictUsize(1), post_sweeps = StrictUsize(1), max_agg_size = StrictUsize(8), diag_eps = StrictF64(1e-14), eig_eps = StrictF64(1e-12), sym_tol = StrictF64(1e-10), seed = StrictU64(42), randomize_ties = false, near_nullspace = None),
        text_signature = "(num_nodes, indptr, indices, values, *, rel_tol=1e-8, abs_tol=0.0, max_iterations=500, theta=0.25, max_levels=32, max_coarse_size=64, pre_sweeps=1, post_sweeps=1, max_agg_size=8, diag_eps=1e-14, eig_eps=1e-12, sym_tol=1e-10, seed=42, randomize_ties=False, near_nullspace=None)"
    )]
    fn new(
        py: Python<'_>,
        num_nodes: StrictUsize,
        indptr: &Bound<'_, PyAny>,
        indices: &Bound<'_, PyAny>,
        values: &Bound<'_, PyAny>,
        rel_tol: StrictF64,
        abs_tol: StrictF64,
        max_iterations: StrictUsize,
        theta: StrictF64,
        max_levels: StrictUsize,
        max_coarse_size: StrictUsize,
        pre_sweeps: StrictUsize,
        post_sweeps: StrictUsize,
        max_agg_size: StrictUsize,
        diag_eps: StrictF64,
        eig_eps: StrictF64,
        sym_tol: StrictF64,
        seed: StrictU64,
        randomize_ties: bool,
        near_nullspace: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let cfg = build_amg_pcg_config(
            rel_tol,
            abs_tol,
            max_iterations,
            theta,
            max_levels,
            max_coarse_size,
            pre_sweeps,
            post_sweeps,
            max_agg_size,
            diag_eps,
            eig_eps,
            sym_tol,
            seed,
            randomize_ties,
        )?;
        let (inner, ws) =
            create_amg_from_csr(py, num_nodes, indptr, indices, values, cfg, near_nullspace)?;
        Ok(Self { inner, ws })
    }

    /// Construct an `AmgPcgSolver` directly from raw CSR buffers `(num_nodes, indptr, indices, values)`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (num_nodes, indptr, indices, values, *, rel_tol = StrictF64(1e-8), abs_tol = StrictF64(0.0), max_iterations = StrictUsize(500), theta = StrictF64(0.25), max_levels = StrictUsize(32), max_coarse_size = StrictUsize(64), pre_sweeps = StrictUsize(1), post_sweeps = StrictUsize(1), max_agg_size = StrictUsize(8), diag_eps = StrictF64(1e-14), eig_eps = StrictF64(1e-12), sym_tol = StrictF64(1e-10), seed = StrictU64(42), randomize_ties = false, near_nullspace = None),
        text_signature = "(num_nodes, indptr, indices, values, *, rel_tol=1e-8, abs_tol=0.0, max_iterations=500, theta=0.25, max_levels=32, max_coarse_size=64, pre_sweeps=1, post_sweeps=1, max_agg_size=8, diag_eps=1e-14, eig_eps=1e-12, sym_tol=1e-10, seed=42, randomize_ties=False, near_nullspace=None)"
    )]
    fn from_csr(
        py: Python<'_>,
        num_nodes: StrictUsize,
        indptr: &Bound<'_, PyAny>,
        indices: &Bound<'_, PyAny>,
        values: &Bound<'_, PyAny>,
        rel_tol: StrictF64,
        abs_tol: StrictF64,
        max_iterations: StrictUsize,
        theta: StrictF64,
        max_levels: StrictUsize,
        max_coarse_size: StrictUsize,
        pre_sweeps: StrictUsize,
        post_sweeps: StrictUsize,
        max_agg_size: StrictUsize,
        diag_eps: StrictF64,
        eig_eps: StrictF64,
        sym_tol: StrictF64,
        seed: StrictU64,
        randomize_ties: bool,
        near_nullspace: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Self::new(
            py,
            num_nodes,
            indptr,
            indices,
            values,
            rel_tol,
            abs_tol,
            max_iterations,
            theta,
            max_levels,
            max_coarse_size,
            pre_sweeps,
            post_sweeps,
            max_agg_size,
            diag_eps,
            eig_eps,
            sym_tol,
            seed,
            randomize_ties,
            near_nullspace,
        )
    }

    /// Construct an `AmgPcgSolver` directly from an existing `CsrMatrix`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (matrix, *, rel_tol = StrictF64(1e-8), abs_tol = StrictF64(0.0), max_iterations = StrictUsize(500), theta = StrictF64(0.25), max_levels = StrictUsize(32), max_coarse_size = StrictUsize(64), pre_sweeps = StrictUsize(1), post_sweeps = StrictUsize(1), max_agg_size = StrictUsize(8), diag_eps = StrictF64(1e-14), eig_eps = StrictF64(1e-12), sym_tol = StrictF64(1e-10), seed = StrictU64(42), randomize_ties = false, near_nullspace = None),
        text_signature = "(matrix, *, rel_tol=1e-8, abs_tol=0.0, max_iterations=500, theta=0.25, max_levels=32, max_coarse_size=64, pre_sweeps=1, post_sweeps=1, max_agg_size=8, diag_eps=1e-14, eig_eps=1e-12, sym_tol=1e-10, seed=42, randomize_ties=False, near_nullspace=None)"
    )]
    fn from_csr_matrix(
        py: Python<'_>,
        matrix: &PyCsrMatrix,
        rel_tol: StrictF64,
        abs_tol: StrictF64,
        max_iterations: StrictUsize,
        theta: StrictF64,
        max_levels: StrictUsize,
        max_coarse_size: StrictUsize,
        pre_sweeps: StrictUsize,
        post_sweeps: StrictUsize,
        max_agg_size: StrictUsize,
        diag_eps: StrictF64,
        eig_eps: StrictF64,
        sym_tol: StrictF64,
        seed: StrictU64,
        randomize_ties: bool,
        near_nullspace: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let cfg = build_amg_pcg_config(
            rel_tol,
            abs_tol,
            max_iterations,
            theta,
            max_levels,
            max_coarse_size,
            pre_sweeps,
            post_sweeps,
            max_agg_size,
            diag_eps,
            eig_eps,
            sym_tol,
            seed,
            randomize_ties,
        )?;
        let (inner, ws) = create_amg_from_csr_matrix(py, matrix, cfg, near_nullspace)?;
        Ok(Self { inner, ws })
    }

    /// Build an `AmgPcgSolver` for `I + gamma * L_sym` with exact `D^{1/2} 1` near-nullspace.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, gamma = StrictF64(1.0), *, rel_tol = StrictF64(1e-8), abs_tol = StrictF64(0.0), max_iterations = StrictUsize(500), theta = StrictF64(0.25), max_levels = StrictUsize(32), max_coarse_size = StrictUsize(64), pre_sweeps = StrictUsize(1), post_sweeps = StrictUsize(1), max_agg_size = StrictUsize(8), diag_eps = StrictF64(1e-14), eig_eps = StrictF64(1e-12), sym_tol = StrictF64(1e-10), seed = StrictU64(42), randomize_ties = false),
        text_signature = "(graph, gamma=1.0, *, rel_tol=1e-8, abs_tol=0.0, max_iterations=500, theta=0.25, max_levels=32, max_coarse_size=64, pre_sweeps=1, post_sweeps=1, max_agg_size=8, diag_eps=1e-14, eig_eps=1e-12, sym_tol=1e-10, seed=42, randomize_ties=False)"
    )]
    fn from_graph_regularized_laplacian(
        py: Python<'_>,
        graph: &PyCsrGraph,
        gamma: StrictF64,
        rel_tol: StrictF64,
        abs_tol: StrictF64,
        max_iterations: StrictUsize,
        theta: StrictF64,
        max_levels: StrictUsize,
        max_coarse_size: StrictUsize,
        pre_sweeps: StrictUsize,
        post_sweeps: StrictUsize,
        max_agg_size: StrictUsize,
        diag_eps: StrictF64,
        eig_eps: StrictF64,
        sym_tol: StrictF64,
        seed: StrictU64,
        randomize_ties: bool,
    ) -> PyResult<Self> {
        let cfg = build_amg_pcg_config(
            rel_tol,
            abs_tol,
            max_iterations,
            theta,
            max_levels,
            max_coarse_size,
            pre_sweeps,
            post_sweeps,
            max_agg_size,
            diag_eps,
            eig_eps,
            sym_tol,
            seed,
            randomize_ties,
        )?;
        let (inner, ws) = create_amg_from_regularized_laplacian(py, graph, gamma, cfg)?;
        Ok(Self { inner, ws })
    }

    /// Build an `AmgPcgSolver` for a shifted graph Laplacian (`kind="combinatorial"` or `"symmetric_normalized"`).
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, kind = "combinatorial", shift = StrictF64(0.0), *, rel_tol = StrictF64(1e-8), abs_tol = StrictF64(0.0), max_iterations = StrictUsize(500), theta = StrictF64(0.25), max_levels = StrictUsize(32), max_coarse_size = StrictUsize(64), pre_sweeps = StrictUsize(1), post_sweeps = StrictUsize(1), max_agg_size = StrictUsize(8), diag_eps = StrictF64(1e-14), eig_eps = StrictF64(1e-12), sym_tol = StrictF64(1e-10), seed = StrictU64(42), randomize_ties = false),
        text_signature = "(graph, kind='combinatorial', shift=0.0, *, rel_tol=1e-8, abs_tol=0.0, max_iterations=500, theta=0.25, max_levels=32, max_coarse_size=64, pre_sweeps=1, post_sweeps=1, max_agg_size=8, diag_eps=1e-14, eig_eps=1e-12, sym_tol=1e-10, seed=42, randomize_ties=False)"
    )]
    fn from_graph_laplacian(
        py: Python<'_>,
        graph: &PyCsrGraph,
        kind: &str,
        shift: StrictF64,
        rel_tol: StrictF64,
        abs_tol: StrictF64,
        max_iterations: StrictUsize,
        theta: StrictF64,
        max_levels: StrictUsize,
        max_coarse_size: StrictUsize,
        pre_sweeps: StrictUsize,
        post_sweeps: StrictUsize,
        max_agg_size: StrictUsize,
        diag_eps: StrictF64,
        eig_eps: StrictF64,
        sym_tol: StrictF64,
        seed: StrictU64,
        randomize_ties: bool,
    ) -> PyResult<Self> {
        let cfg = build_amg_pcg_config(
            rel_tol,
            abs_tol,
            max_iterations,
            theta,
            max_levels,
            max_coarse_size,
            pre_sweeps,
            post_sweeps,
            max_agg_size,
            diag_eps,
            eig_eps,
            sym_tol,
            seed,
            randomize_ties,
        )?;
        let (inner, ws) = create_amg_from_laplacian(py, graph, kind, shift, cfg)?;
        Ok(Self { inner, ws })
    }

    /// Solve `A x = b` synchronously and return `AmgSolveResult` while releasing the GIL.
    #[pyo3(signature = (b, x0 = None), text_signature = "($self, b, x0=None)")]
    fn solve(
        &mut self,
        py: Python<'_>,
        b: &Bound<'_, PyAny>,
        x0: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyAmgSolveResult> {
        solve_amg(&self.inner, &mut self.ws, py, b, x0)
    }

    /// Solve `A X = B` for a 2D `(num_nodes, ncols)` RHS matrix and return `AmgBatchSolveResult`.
    #[pyo3(signature = (b, x0 = None), text_signature = "($self, b, x0=None)")]
    fn solve_batch(
        &mut self,
        py: Python<'_>,
        b: &Bound<'_, PyAny>,
        x0: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyAmgBatchSolveResult> {
        solve_batch_amg(&self.inner, &mut self.ws, py, b, x0)
    }

    /// Initialize a cooperative PCG solve for `b` and optional `x0`, returning status string.
    #[pyo3(signature = (b, x0 = None), text_signature = "($self, b, x0=None)")]
    fn start_solve(
        &mut self,
        py: Python<'_>,
        b: &Bound<'_, PyAny>,
        x0: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<&'static str> {
        start_solve_amg(&self.inner, &mut self.ws, py, b, x0)
    }

    /// Execute a single cooperative PCG iteration while releasing the GIL.
    #[pyo3(signature = (), text_signature = "($self)")]
    fn step(&mut self, py: Python<'_>) -> PyResult<&'static str> {
        let inner = &self.inner;
        let ws = &mut self.ws;
        let status = py.detach(|| inner.step(ws)).map_err(core_error_to_py)?;
        Ok(solver_status_str(status))
    }

    /// Execute up to `steps` cooperative PCG iterations while releasing the GIL.
    #[pyo3(signature = (steps), text_signature = "($self, steps)")]
    fn run_steps(&mut self, py: Python<'_>, steps: StrictUsize) -> PyResult<&'static str> {
        let inner = &self.inner;
        let ws = &mut self.ws;
        let n_steps = steps.0;
        let status = py
            .detach(|| inner.run_steps(ws, n_steps))
            .map_err(core_error_to_py)?;
        Ok(solver_status_str(status))
    }

    /// Run the active cooperative PCG solve to completion while releasing the GIL.
    #[pyo3(signature = (), text_signature = "($self)")]
    fn run(&mut self, py: Python<'_>) -> PyResult<&'static str> {
        let inner = &self.inner;
        let ws = &mut self.ws;
        let status = py.detach(|| inner.run(ws)).map_err(core_error_to_py)?;
        Ok(solver_status_str(status))
    }

    /// Return a `float64` NumPy copy of the current solution vector `x`.
    fn solution<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.ws.x())
    }

    /// Matrix dimension `num_nodes`.
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Number of AMG hierarchy levels.
    #[getter]
    fn num_levels(&self) -> usize {
        self.inner.num_levels()
    }

    /// Coarsest-level matrix dimension.
    #[getter]
    fn coarsest_size(&self) -> usize {
        self.inner.coarsest_size()
    }

    /// AMG operator complexity `sum(nnz(A_l)) / nnz(A_0)`.
    #[getter]
    fn operator_complexity(&self) -> f64 {
        self.inner.operator_complexity()
    }

    /// Maximum CSR nonzero count across all AMG hierarchy levels.
    #[getter]
    fn max_level_nnz(&self) -> usize {
        self.inner.max_level_nnz()
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.ws.status())
    }

    /// Number of PCG iterations executed in the active solve.
    #[getter]
    fn iteration(&self) -> usize {
        self.ws.iteration()
    }

    /// Maximum scheduled PCG iterations for the active solve.
    #[getter]
    fn total_iterations(&self) -> usize {
        self.ws.total_iterations()
    }

    /// Whether the active solve has reached `"converged"` or `"completed"`.
    #[getter]
    fn is_finished(&self) -> bool {
        self.ws.is_finished()
    }

    /// Most recent Euclidean residual norm `||b - A x||_2`.
    #[getter]
    fn residual_norm(&self) -> f64 {
        self.ws.residual_norm()
    }

    /// Most recent relative Euclidean residual `||b - A x||_2 / ||b||_2`.
    #[getter]
    fn relative_residual(&self) -> f64 {
        self.ws.relative_residual()
    }

    /// Whether the active solve converged within tolerance.
    #[getter]
    fn converged(&self) -> bool {
        self.ws.converged()
    }

    /// Return a concise string representation of the `AmgPcgSolver`.
    fn __repr__(&self) -> String {
        format!(
            "AmgPcgSolver(num_nodes={}, num_levels={}, coarsest_size={}, operator_complexity={:.3}, status='{}', iteration={})",
            self.inner.num_nodes(),
            self.inner.num_levels(),
            self.inner.coarsest_size(),
            self.inner.operator_complexity(),
            self.status(),
            self.ws.iteration(),
        )
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAmgPcgSolver>()?;
    Ok(())
}
