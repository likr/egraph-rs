//! Force-directed edge bundling (`FdebSolver`) bindings.

use egraph::{core as egraph_core, layout as egraph_layout};
use egraph_core::SolverStatus;
use egraph_layout::{FdebConfig, FdebSolver};
use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictUsize, core_error_to_py, solver_status_str,
};
use crate::layout_buffers::{Layout2dInner, PyLayout2d};
use crate::solver_common::{SolverLayout, extract_solver_layout};

enum FdebInner {
    F64(FdebSolver<f64>),
    F32(FdebSolver<f32>),
}

impl FdebInner {
    fn dtype(&self) -> ScalarDtype {
        match self {
            Self::F64(_) => ScalarDtype::F64,
            Self::F32(_) => ScalarDtype::F32,
        }
    }

    fn status(&self) -> SolverStatus {
        match self {
            Self::F64(s) => s.status(),
            Self::F32(s) => s.status(),
        }
    }

    fn iteration(&self) -> usize {
        match self {
            Self::F64(s) => s.iteration(),
            Self::F32(s) => s.iteration(),
        }
    }

    fn total_iterations(&self) -> usize {
        match self {
            Self::F64(s) => s.total_iterations(),
            Self::F32(s) => s.total_iterations(),
        }
    }

    fn num_edges(&self) -> usize {
        match self {
            Self::F64(s) => s.num_edges(),
            Self::F32(s) => s.num_edges(),
        }
    }
}

/// Cooperative Force-Directed Edge Bundling (`FDEB`) solver with flat polyline outputs.
///
/// Takes `CsrGraph` and `Layout2d` by reference, supports `"float64"` and `"float32"`,
/// and releases the Python GIL during `step()`, `run_steps()`, and `run()`.
#[pyclass(name = "FdebSolver", module = "egraph._core")]
pub struct PyFdebSolver {
    inner: FdebInner,
}

#[pymethods]
impl PyFdebSolver {
    /// Construct an `FdebSolver` from `graph` and `layout` without taking ownership of `layout`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            layout,
            *,
            cycles = StrictUsize(6),
            s0 = StrictF64(0.1),
            i0 = StrictUsize(90),
            s_step = StrictF64(0.5),
            i_step = StrictF64(2.0 / 3.0),
            spring_k = StrictF64(0.1),
            minimum_edge_compatibility = StrictF64(0.6),
            dtype = None
        ),
        text_signature = "(graph, layout, *, cycles=6, s0=0.1, i0=90, s_step=0.5, i_step=2.0/3.0, spring_k=0.1, minimum_edge_compatibility=0.6, dtype=None)"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        layout: &Bound<'_, PyLayout2d>,
        cycles: StrictUsize,
        s0: StrictF64,
        i0: StrictUsize,
        s_step: StrictF64,
        i_step: StrictF64,
        spring_k: StrictF64,
        minimum_edge_compatibility: StrictF64,
        dtype: Option<ScalarDtype>,
    ) -> PyResult<Self> {
        let cfg = FdebConfig {
            cycles: cycles.0,
            s0: s0.0,
            i0: i0.0,
            s_step: s_step.0,
            i_step: i_step.0,
            spring_k: spring_k.0,
            minimum_edge_compatibility: minimum_edge_compatibility.0,
        };
        cfg.validate().map_err(core_error_to_py)?;

        let resolved_dtype = match dtype {
            Some(dt) => dt,
            None => match layout.try_borrow()?.inner() {
                Layout2dInner::F64(_) => ScalarDtype::F64,
                Layout2dInner::F32(_) => ScalarDtype::F32,
            },
        };
        let soa = extract_solver_layout(layout, resolved_dtype)?;
        let g = graph.inner();
        let inner = py
            .detach(|| match soa {
                SolverLayout::F64(l64) => FdebSolver::<f64>::new(g, &l64, cfg).map(FdebInner::F64),
                SolverLayout::F32(l32) => FdebSolver::<f32>::new(g, &l32, cfg).map(FdebInner::F32),
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute a single FDEB force iteration while releasing the GIL.
    #[pyo3(signature = (), text_signature = "($self)")]
    fn step(&mut self, py: Python<'_>) -> PyResult<&'static str> {
        let status = py
            .detach(|| match &mut self.inner {
                FdebInner::F64(s) => s.step(),
                FdebInner::F32(s) => s.step(),
            })
            .map_err(core_error_to_py)?;
        Ok(solver_status_str(status))
    }

    /// Execute up to `steps` FDEB force iterations while releasing the GIL.
    #[pyo3(signature = (steps), text_signature = "($self, steps)")]
    fn run_steps(&mut self, py: Python<'_>, steps: StrictUsize) -> PyResult<&'static str> {
        let n_steps = steps.0;
        let status = py
            .detach(|| match &mut self.inner {
                FdebInner::F64(s) => s.run_steps(n_steps),
                FdebInner::F32(s) => s.run_steps(n_steps),
            })
            .map_err(core_error_to_py)?;
        Ok(solver_status_str(status))
    }

    /// Run all remaining FDEB subdivision cycles to completion while releasing the GIL.
    #[pyo3(signature = (), text_signature = "($self)")]
    fn run(&mut self, py: Python<'_>) -> PyResult<&'static str> {
        let status = py
            .detach(|| match &mut self.inner {
                FdebInner::F64(s) => s.run(),
                FdebInner::F32(s) => s.run(),
            })
            .map_err(core_error_to_py)?;
        Ok(solver_status_str(status))
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.status())
    }

    /// Number of iterations executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Total scheduled iterations across all subdivision cycles.
    #[getter]
    fn total_iterations(&self) -> usize {
        self.inner.total_iterations()
    }

    /// Whether the solver has reached `"converged"` or `"completed"`.
    #[getter]
    fn is_finished(&self) -> bool {
        self.inner.status().is_finished()
    }

    /// Number of edges being bundled.
    #[getter]
    fn num_edges(&self) -> usize {
        self.inner.num_edges()
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

    /// Return a `(K, 2)` NumPy array of all edge polyline points matching `self.dtype`.
    fn points<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            FdebInner::F64(s) => {
                let k = s.points_x().len();
                let interleaved = s.to_interleaved_points();
                let arr = Array2::from_shape_vec((k, 2), interleaved)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyArray2::from_owned_array(py, arr).into_any())
            }
            FdebInner::F32(s) => {
                let k = s.points_x().len();
                let interleaved = s.to_interleaved_points();
                let arr = Array2::from_shape_vec((k, 2), interleaved)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyArray2::from_owned_array(py, arr).into_any())
            }
        }
    }

    /// Return a concise string representation of the `FdebSolver`.
    fn __repr__(&self) -> String {
        format!(
            "FdebSolver(num_edges={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_edges(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}
