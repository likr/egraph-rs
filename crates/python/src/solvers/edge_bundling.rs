//! Force-directed edge bundling (`FdebSolver`) bindings.

use egraph::{core as egraph_core, layout as egraph_layout};
use egraph_layout::{FdebConfig, FdebSolver};
use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictUsize, core_error_to_py, solver_status_str,
};
use crate::solver_common::{DualIterativeSolver, SolverAction, impl_dual_solver};

use super::layout_dispatch::{execute_euclid_action, impl_dual_nd_solver};

impl_dual_solver!(FdebInner, FdebSolver);
impl_dual_nd_solver!(FdebInner, FdebSolver);

impl FdebInner {
    fn num_nodes(&self) -> usize {
        match self {
            Self::F64(s) => s.num_nodes(),
            Self::F32(s) => s.num_nodes(),
        }
    }

    fn num_edges(&self) -> usize {
        match self {
            Self::F64(s) => s.num_edges(),
            Self::F32(s) => s.num_edges(),
        }
    }

    fn point_dims(&self) -> usize {
        match self {
            Self::F64(s) => s.point_dims(),
            Self::F32(s) => s.point_dims(),
        }
    }
}

/// Cooperative Force-Directed Edge Bundling (`FDEB`) solver with flat polyline outputs.
///
/// Constructed from a `CsrGraph` and lazily initialized on first `step(layout)` / `run(layout)`
/// against either `Layout2d` or `LayoutNd`, supporting `"float64"` and `"float32"` precision
/// while releasing the Python GIL.
#[pyclass(name = "FdebSolver", module = "egraph._core")]
pub struct PyFdebSolver {
    inner: FdebInner,
}

#[pymethods]
impl PyFdebSolver {
    /// Construct an `FdebSolver` from `graph` (polylines initialize lazily on first step).
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            cycles = StrictUsize(6),
            s0 = StrictF64(0.1),
            steps_per_cycle = StrictUsize(90),
            s_step = StrictF64(0.5),
            i_step = StrictF64(0.6666666666666666),
            spring_k = StrictF64(0.1),
            minimum_edge_compatibility = StrictF64(0.6),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, cycles=6, s0=0.1, steps_per_cycle=90, s_step=0.5, i_step=0.6666666666666666, spring_k=0.1, minimum_edge_compatibility=0.6, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        cycles: StrictUsize,
        s0: StrictF64,
        steps_per_cycle: StrictUsize,
        s_step: StrictF64,
        i_step: StrictF64,
        spring_k: StrictF64,
        minimum_edge_compatibility: StrictF64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = FdebConfig {
            cycles: cycles.0,
            s0: s0.0,
            steps_per_cycle: steps_per_cycle.0,
            s_step: s_step.0,
            i_step: i_step.0,
            spring_k: spring_k.0,
            minimum_edge_compatibility: minimum_edge_compatibility.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => FdebSolver::<f64>::new(&*g, cfg).map(FdebInner::F64),
                ScalarDtype::F32 => FdebSolver::<f32>::new(&*g, cfg).map(FdebInner::F32),
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute a single FDEB force iteration against `layout` (`Layout2d | LayoutNd`).
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        execute_euclid_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` FDEB force iterations against `layout` (`Layout2d | LayoutNd`).
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyAny>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_euclid_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run all remaining FDEB subdivision cycles against `layout` (`Layout2d | LayoutNd`).
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        execute_euclid_action(&mut self.inner, py, layout, SolverAction::Run)
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

    /// Total scheduled iterations across all subdivision cycles.
    #[getter]
    fn total_iterations(&self) -> usize {
        self.inner.solver_total_iterations()
    }

    /// Whether the solver has reached `"converged"` or `"completed"`.
    #[getter]
    fn is_finished(&self) -> bool {
        self.inner.solver_status().is_finished()
    }

    /// Expected number of graph vertices.
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Number of edges being bundled.
    #[getter]
    fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    /// Active coordinate dimensionality of bundled edge polylines (`0` before first `step`).
    #[getter]
    fn point_dims(&self) -> usize {
        self.inner.point_dims()
    }

    /// Scalar coordinate precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a `uint32` NumPy copy of CSR-style polyline offsets of length `num_edges + 1`.
    fn edge_offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        match &self.inner {
            FdebInner::F64(s) => PyArray1::from_slice(py, s.edge_offsets()),
            FdebInner::F32(s) => PyArray1::from_slice(py, s.edge_offsets()),
        }
    }

    /// Return a `(K, point_dims)` NumPy array of all edge polyline points matching `self.dtype`.
    fn points<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            FdebInner::F64(s) => {
                let d = s.point_dims();
                if d == 0 {
                    let arr = Array2::<f64>::zeros((0, 0));
                    return Ok(PyArray2::from_owned_array(py, arr).into_any());
                }
                let flat = s.to_interleaved_points_nd();
                let k = flat.len() / d;
                let arr = Array2::from_shape_vec((k, d), flat)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyArray2::from_owned_array(py, arr).into_any())
            }
            FdebInner::F32(s) => {
                let d = s.point_dims();
                if d == 0 {
                    let arr = Array2::<f32>::zeros((0, 0));
                    return Ok(PyArray2::from_owned_array(py, arr).into_any());
                }
                let flat = s.to_interleaved_points_nd();
                let k = flat.len() / d;
                let arr = Array2::from_shape_vec((k, d), flat)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyArray2::from_owned_array(py, arr).into_any())
            }
        }
    }

    /// Extract a single coordinate `axis` (`0..point_dims`) across all bundled edge control points.
    #[pyo3(signature = (axis), text_signature = "($self, axis)")]
    fn points_axis<'py>(&self, py: Python<'py>, axis: StrictUsize) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            FdebInner::F64(s) => {
                let vec = s.points_axis(axis.0).map_err(core_error_to_py)?;
                Ok(PyArray1::from_vec(py, vec).into_any())
            }
            FdebInner::F32(s) => {
                let vec = s.points_axis(axis.0).map_err(core_error_to_py)?;
                Ok(PyArray1::from_vec(py, vec).into_any())
            }
        }
    }

    /// Extract the `z` coordinate (`axis=2`) across all bundled edge control points.
    #[pyo3(signature = (), text_signature = "($self)")]
    fn points_z<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            FdebInner::F64(s) => {
                let vec = s.points_z().map_err(core_error_to_py)?;
                Ok(PyArray1::from_vec(py, vec).into_any())
            }
            FdebInner::F32(s) => {
                let vec = s.points_z().map_err(core_error_to_py)?;
                Ok(PyArray1::from_vec(py, vec).into_any())
            }
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "FdebSolver(num_edges={}, point_dims={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_edges(),
            self.point_dims(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}

#[allow(dead_code)]
fn _check_core_err(_: egraph_core::CoreError) {}
