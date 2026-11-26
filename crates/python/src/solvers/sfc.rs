//! Space-filling curve (`SfcLayoutSolver`) layout solver bindings.

use egraph::layout as egraph_layout;
use egraph_layout::{SfcConfig, SfcCurveKind, SfcLayoutSolver};
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, normalize_enum_key,
    solver_status_str,
};
use crate::layout_buffers::PyLayout2d;
use crate::solver_common::{
    DualIterativeSolver, SolverAction, execute_solver_action, impl_dual_solver,
};

pub(super) fn parse_sfc_curve_kind(raw: &str) -> Result<SfcCurveKind, String> {
    match normalize_enum_key(raw).as_str() {
        "hilbert" => Ok(SfcCurveKind::Hilbert),
        "morton" | "zorder" => Ok(SfcCurveKind::Morton),
        _ => Err(format!(
            "unknown SFC curve {raw:?}; expected 'hilbert' or 'morton'"
        )),
    }
}

impl_dual_solver!(SfcLayoutInner, SfcLayoutSolver);

/// Cooperative community-ordered Space-Filling Curve (`SFC`) layout solver.
///
/// Precomputes a hierarchical Louvain community vertex ordering and maps ranks injectively
/// onto a `2^k x 2^k` Hilbert or Morton (`"hilbert"` / `"morton"`) grid inside
/// `[-scale / 2, scale / 2]^2`, guaranteeing zero vertex co-location in both `"float64"`
/// and `"float32"` (`total_iterations == 1`) while releasing the Python GIL.
///
/// Note (`muelder2008_space_filling` `MISSING_PDF`): Muelder & Ma (2008) is marked
/// `MISSING_PDF` in the project literature archive, so exact clustering, ordering, and
/// post-refinement parameters from that paper are not claimed; this solver provides a
/// canonical Louvain community-ordered Hilbert/Morton space-filling curve baseline.
#[pyclass(name = "SfcLayoutSolver", module = "egraph._core")]
pub struct PySfcLayoutSolver {
    inner: SfcLayoutInner,
}

#[pymethods]
impl PySfcLayoutSolver {
    /// Construct an `SfcLayoutSolver` from a `CsrGraph`.
    #[new]
    #[pyo3(
        signature = (graph, *, curve = "hilbert", scale = StrictF64(100.0), resolution_order = None, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, *, curve='hilbert', scale=100.0, resolution_order=None, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        curve: &str,
        scale: StrictF64,
        resolution_order: Option<StrictUsize>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let parsed_curve = parse_sfc_curve_kind(curve).map_err(PyValueError::new_err)?;
        let res_order = match resolution_order {
            Some(ord) => Some(u32::try_from(ord.0).map_err(|_| {
                PyValueError::new_err(format!(
                    "resolution_order must be in [1, 15], got {}",
                    ord.0
                ))
            })?),
            None => None,
        };
        let cfg = SfcConfig {
            curve: parsed_curve,
            scale: scale.0,
            resolution_order: res_order,
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => SfcLayoutSolver::<f64>::new(g, cfg).map(SfcLayoutInner::F64),
                ScalarDtype::F32 => SfcLayoutSolver::<f32>::new(g, cfg).map(SfcLayoutInner::F32),
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Write the precomputed space-filling curve coordinates into `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` space-filling curve steps in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run SFC coordinate placement to completion in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Return a `uint32` NumPy copy of the community-ordered vertex permutation along the curve.
    #[pyo3(signature = (), text_signature = "($self)")]
    fn vertex_order<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        let slice = match &self.inner {
            SfcLayoutInner::F64(s) => s.vertex_order(),
            SfcLayoutInner::F32(s) => s.vertex_order(),
        };
        PyArray1::from_slice(py, slice)
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of steps executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Total number of SFC placement steps (`1`).
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
            SfcLayoutInner::F64(s) => s.num_nodes(),
            SfcLayoutInner::F32(s) => s.num_nodes(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `SfcLayoutSolver`.
    fn __repr__(&self) -> String {
        format!(
            "SfcLayoutSolver(num_nodes={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}
