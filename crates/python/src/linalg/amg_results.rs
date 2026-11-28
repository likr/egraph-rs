//! Frozen result classes for `AmgPcgSolver.solve` and `AmgPcgSolver.solve_batch`.

use egraph::linalg::{AmgPcgBatchResult, AmgPcgSolveResult};
use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::errors::invalid_input_err;
use crate::ffi_convert::solver_status_str;

const AMG_SOLVE_KEYS: [&str; 6] = [
    "solution",
    "iterations",
    "residual_norm",
    "relative_residual",
    "converged",
    "status",
];

const AMG_BATCH_KEYS: [&str; 11] = [
    "solutions",
    "total_iterations",
    "max_iterations_used",
    "min_iterations_used",
    "max_residual_norm",
    "max_relative_residual",
    "all_converged",
    "iterations_per_col",
    "residual_norms",
    "relative_residuals",
    "converged_per_col",
];

/// Frozen result returned by `AmgPcgSolver.solve`.
#[pyclass(
    name = "AmgSolveResult",
    module = "egraph._core",
    frozen,
    from_py_object
)]
#[derive(Debug, Clone)]
pub struct PyAmgSolveResult {
    solution_vec: Vec<f64>,
    result: AmgPcgSolveResult,
}

impl PyAmgSolveResult {
    pub(super) fn new(solution_vec: Vec<f64>, result: AmgPcgSolveResult) -> Self {
        Self {
            solution_vec,
            result,
        }
    }

    fn lookup(&self, py: Python<'_>, key: &str) -> PyResult<Option<Py<PyAny>>> {
        match key {
            "solution" => Ok(Some(self.solution(py).into_any().unbind())),
            "iterations" => Ok(Some(self.result.iterations.into_py_any(py)?)),
            "residual_norm" => Ok(Some(self.result.residual_norm.into_py_any(py)?)),
            "relative_residual" => Ok(Some(self.result.relative_residual.into_py_any(py)?)),
            "converged" => Ok(Some(self.result.converged.into_py_any(py)?)),
            "status" => Ok(Some(solver_status_str(self.result.status).into_py_any(py)?)),
            _ => Ok(None),
        }
    }
}

#[pymethods]
impl PyAmgSolveResult {
    /// Solution vector `x` as a `float64` NumPy array.
    #[getter]
    fn solution<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.solution_vec)
    }

    /// Number of PCG iterations performed.
    #[getter]
    fn iterations(&self) -> usize {
        self.result.iterations
    }

    /// Final Euclidean residual norm `||b - A x||_2`.
    #[getter]
    fn residual_norm(&self) -> f64 {
        self.result.residual_norm
    }

    /// Final relative residual norm `||b - A x||_2 / ||b||_2`.
    #[getter]
    fn relative_residual(&self) -> f64 {
        self.result.relative_residual
    }

    /// Whether the solve satisfied the convergence criteria.
    #[getter]
    fn converged(&self) -> bool {
        self.result.converged
    }

    /// Solver status string (`"converged"`, `"completed"`, or `"failed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.result.status)
    }

    /// Return all result fields as a Python dictionary.
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        for k in AMG_SOLVE_KEYS {
            if let Some(v) = self.lookup(py, k)? {
                dict.set_item(k, v)?;
            }
        }
        Ok(dict.unbind())
    }

    /// Return ordered field keys for dictionary compatibility.
    fn keys(&self) -> Vec<&'static str> {
        AMG_SOLVE_KEYS.to_vec()
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        self.lookup(py, key)?
            .ok_or_else(|| PyKeyError::new_err(key.to_owned()))
    }

    fn __contains__(&self, key: &str) -> bool {
        AMG_SOLVE_KEYS.contains(&key)
    }

    fn __repr__(&self) -> String {
        format!(
            "AmgSolveResult(n={}, iterations={}, relative_residual={:.3e}, converged={}, status='{}')",
            self.solution_vec.len(),
            self.result.iterations,
            self.result.relative_residual,
            self.result.converged,
            self.status(),
        )
    }
}

/// Frozen result returned by `AmgPcgSolver.solve_batch`.
#[pyclass(
    name = "AmgBatchSolveResult",
    module = "egraph._core",
    frozen,
    from_py_object
)]
#[derive(Debug, Clone)]
pub struct PyAmgBatchSolveResult {
    solutions_flat: Vec<f64>,
    nrows: usize,
    ncols: usize,
    result: AmgPcgBatchResult,
}

impl PyAmgBatchSolveResult {
    pub(super) fn new(
        solutions_flat: Vec<f64>,
        nrows: usize,
        ncols: usize,
        result: AmgPcgBatchResult,
    ) -> Self {
        Self {
            solutions_flat,
            nrows,
            ncols,
            result,
        }
    }

    fn lookup(&self, py: Python<'_>, key: &str) -> PyResult<Option<Py<PyAny>>> {
        match key {
            "solutions" => Ok(Some(self.solutions(py)?.into_any().unbind())),
            "total_iterations" => Ok(Some(self.result.total_iterations.into_py_any(py)?)),
            "max_iterations_used" => Ok(Some(self.result.max_iterations_used.into_py_any(py)?)),
            "min_iterations_used" => Ok(Some(self.result.min_iterations_used.into_py_any(py)?)),
            "max_residual_norm" => Ok(Some(self.result.max_residual_norm.into_py_any(py)?)),
            "max_relative_residual" => Ok(Some(self.result.max_relative_residual.into_py_any(py)?)),
            "all_converged" => Ok(Some(self.result.all_converged.into_py_any(py)?)),
            "iterations_per_col" => Ok(Some(
                self.result.iterations_per_col.clone().into_py_any(py)?,
            )),
            "residual_norms" => Ok(Some(self.result.residual_norms.clone().into_py_any(py)?)),
            "relative_residuals" => Ok(Some(
                self.result.relative_residuals.clone().into_py_any(py)?,
            )),
            "converged_per_col" => Ok(Some(self.result.converged_per_col.clone().into_py_any(py)?)),
            _ => Ok(None),
        }
    }
}

#[pymethods]
impl PyAmgBatchSolveResult {
    /// Solution matrix `X` of shape `(num_nodes, ncols)` as a `float64` NumPy array.
    #[getter]
    fn solutions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let arr = Array2::from_shape_vec((self.nrows, self.ncols), self.solutions_flat.clone())
            .map_err(|e| invalid_input_err(e.to_string()))?;
        Ok(PyArray2::from_owned_array(py, arr))
    }

    /// Total PCG iterations summed across all right-hand sides.
    #[getter]
    fn total_iterations(&self) -> usize {
        self.result.total_iterations
    }

    /// Maximum PCG iterations used across columns.
    #[getter]
    fn max_iterations_used(&self) -> usize {
        self.result.max_iterations_used
    }

    /// Minimum PCG iterations used across columns.
    #[getter]
    fn min_iterations_used(&self) -> usize {
        self.result.min_iterations_used
    }

    /// Maximum Euclidean residual norm across columns.
    #[getter]
    fn max_residual_norm(&self) -> f64 {
        self.result.max_residual_norm
    }

    /// Maximum relative residual norm across columns.
    #[getter]
    fn max_relative_residual(&self) -> f64 {
        self.result.max_relative_residual
    }

    /// Whether every right-hand side column converged.
    #[getter]
    fn all_converged(&self) -> bool {
        self.result.all_converged
    }

    /// Iteration count per right-hand side column.
    #[getter]
    fn iterations_per_col(&self) -> Vec<usize> {
        self.result.iterations_per_col.clone()
    }

    /// Euclidean residual norm per right-hand side column.
    #[getter]
    fn residual_norms(&self) -> Vec<f64> {
        self.result.residual_norms.clone()
    }

    /// Relative residual norm per right-hand side column.
    #[getter]
    fn relative_residuals(&self) -> Vec<f64> {
        self.result.relative_residuals.clone()
    }

    /// Convergence flag per right-hand side column.
    #[getter]
    fn converged_per_col(&self) -> Vec<bool> {
        self.result.converged_per_col.clone()
    }

    /// Return all batch result fields as a Python dictionary.
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        for k in AMG_BATCH_KEYS {
            if let Some(v) = self.lookup(py, k)? {
                dict.set_item(k, v)?;
            }
        }
        Ok(dict.unbind())
    }

    /// Return ordered field keys for dictionary compatibility.
    fn keys(&self) -> Vec<&'static str> {
        AMG_BATCH_KEYS.to_vec()
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        self.lookup(py, key)?
            .ok_or_else(|| PyKeyError::new_err(key.to_owned()))
    }

    fn __contains__(&self, key: &str) -> bool {
        AMG_BATCH_KEYS.contains(&key)
    }

    fn __repr__(&self) -> String {
        format!(
            "AmgBatchSolveResult(shape=({}, {}), total_iterations={}, max_relative_residual={:.3e}, all_converged={})",
            self.nrows,
            self.ncols,
            self.result.total_iterations,
            self.result.max_relative_residual,
            self.result.all_converged,
        )
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAmgSolveResult>()?;
    m.add_class::<PyAmgBatchSolveResult>()?;
    Ok(())
}
