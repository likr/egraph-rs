//! Proximity and scan-line node overlap removal (`PrismOverlapRemoval`, `PfsPrimeOverlapRemoval`).

use egraph::layout as egraph_layout;
use egraph_layout::{PfsPrimeConfig, PfsPrimeOverlapRemoval, PrismConfig, PrismOverlapRemoval};
use pyo3::prelude::*;

use crate::array_extract::extract_f64_1d;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, solver_status_str,
};
use crate::layout_buffers::PyLayout2d;
use crate::solver_common::{
    DualIterativeSolver, SolverAction, execute_solver_action, impl_dual_solver,
    vec_f64_to_f32_checked,
};

impl_dual_solver!(PrismInner, PrismOverlapRemoval);
impl_dual_solver!(PfsPrimeInner, PfsPrimeOverlapRemoval);

/// Cooperative Gansner-Hu (2010) PRISM proximity-stress node overlap removal solver.
///
/// Uses `Delaunay2d` proximity triangulation and sparse CSR Symmetric SSOR-PCG stress majorization
/// in both `"float64"` and `"float32"` precision while releasing the Python GIL.
#[pyclass(name = "PrismOverlapRemoval", module = "egraph._core")]
pub struct PyPrismOverlapRemoval {
    inner: PrismInner,
}

#[pymethods]
impl PyPrismOverlapRemoval {
    /// Create a `PrismOverlapRemoval` solver from per-node `half_widths` and `half_heights`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            half_widths,
            half_heights,
            *,
            max_iterations = StrictUsize(50),
            max_scale_factor = StrictF64(1.5),
            cg_max_iters = StrictUsize(80),
            cg_tolerance = StrictF64(1e-3),
            padding = StrictF64(0.0),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(half_widths, half_heights, *, max_iterations=50, max_scale_factor=1.5, cg_max_iters=80, cg_tolerance=1e-3, padding=0.0, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        half_widths: &Bound<'_, PyAny>,
        half_heights: &Bound<'_, PyAny>,
        max_iterations: StrictUsize,
        max_scale_factor: StrictF64,
        cg_max_iters: StrictUsize,
        cg_tolerance: StrictF64,
        padding: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = PrismConfig {
            max_iterations: max_iterations.0,
            max_scale_factor: max_scale_factor.0,
            cg_max_iters: cg_max_iters.0,
            cg_tolerance: cg_tolerance.0,
            padding: padding.0,
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let hw_f64 = extract_f64_1d(half_widths, "half_widths")?;
        let hh_f64 = extract_f64_1d(half_heights, "half_heights")?;
        let inner = match dtype {
            ScalarDtype::F64 => py
                .detach(|| {
                    PrismOverlapRemoval::<f64>::new(hw_f64, hh_f64, cfg).map(PrismInner::F64)
                })
                .map_err(core_error_to_py)?,
            ScalarDtype::F32 => {
                let hw_f32 = vec_f64_to_f32_checked(&hw_f64, "half_widths")?;
                let hh_f32 = vec_f64_to_f32_checked(&hh_f64, "half_heights")?;
                py.detach(|| {
                    PrismOverlapRemoval::<f32>::new(hw_f32, hh_f32, cfg).map(PrismInner::F32)
                })
                .map_err(core_error_to_py)?
            }
        };
        Ok(Self { inner })
    }

    /// Create a `PrismOverlapRemoval` solver for square/circular nodes with equal half-widths and half-heights.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            radii,
            *,
            max_iterations = StrictUsize(50),
            max_scale_factor = StrictF64(1.5),
            cg_max_iters = StrictUsize(80),
            cg_tolerance = StrictF64(1e-3),
            padding = StrictF64(0.0),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(radii, *, max_iterations=50, max_scale_factor=1.5, cg_max_iters=80, cg_tolerance=1e-3, padding=0.0, seed=42, dtype='float64')"
    )]
    fn from_radii(
        py: Python<'_>,
        radii: &Bound<'_, PyAny>,
        max_iterations: StrictUsize,
        max_scale_factor: StrictF64,
        cg_max_iters: StrictUsize,
        cg_tolerance: StrictF64,
        padding: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = PrismConfig {
            max_iterations: max_iterations.0,
            max_scale_factor: max_scale_factor.0,
            cg_max_iters: cg_max_iters.0,
            cg_tolerance: cg_tolerance.0,
            padding: padding.0,
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let r_f64 = extract_f64_1d(radii, "radii")?;
        let inner = match dtype {
            ScalarDtype::F64 => py
                .detach(|| PrismOverlapRemoval::<f64>::from_radii(r_f64, cfg).map(PrismInner::F64))
                .map_err(core_error_to_py)?,
            ScalarDtype::F32 => {
                let r_f32 = vec_f64_to_f32_checked(&r_f64, "radii")?;
                py.detach(|| {
                    PrismOverlapRemoval::<f32>::from_radii(r_f32, cfg).map(PrismInner::F32)
                })
                .map_err(core_error_to_py)?
            }
        };
        Ok(Self { inner })
    }

    /// Execute one PRISM overlap removal round in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` PRISM overlap removal rounds in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run PRISM overlap removal until zero overlaps remain or `max_iterations` is reached.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Number of overlapping rectangle pairs remaining after the most recent step.
    #[getter]
    fn last_overlap_count(&self) -> usize {
        match &self.inner {
            PrismInner::F64(s) => s.last_overlap_count(),
            PrismInner::F32(s) => s.last_overlap_count(),
        }
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of PRISM rounds executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Configured maximum number of PRISM rounds.
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
            PrismInner::F64(s) => s.num_nodes(),
            PrismInner::F32(s) => s.num_nodes(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `PrismOverlapRemoval` solver.
    fn __repr__(&self) -> String {
        format!(
            "PrismOverlapRemoval(num_nodes={}, last_overlap_count={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.last_overlap_count(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}

/// Cooperative Nachmanson et al. / Chen et al. (2020) `PFS'` dual-axis scan-line node overlap removal solver.
///
/// Removes all axis-aligned rectangle overlaps in one deterministic pass (`total_iterations == 1`)
/// while strictly preserving orthogonal ordering (`oo_nni == 0.0`). Supports `"float64"` and
/// `"float32"` `Layout2d` while releasing the Python GIL.
#[pyclass(name = "PfsPrimeOverlapRemoval", module = "egraph._core")]
pub struct PyPfsPrimeOverlapRemoval {
    inner: PfsPrimeInner,
}

#[pymethods]
impl PyPfsPrimeOverlapRemoval {
    /// Create a `PfsPrimeOverlapRemoval` solver from per-node `half_widths` and `half_heights`.
    #[new]
    #[pyo3(
        signature = (half_widths, half_heights, *, padding = StrictF64(0.0), seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(half_widths, half_heights, *, padding=0.0, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        half_widths: &Bound<'_, PyAny>,
        half_heights: &Bound<'_, PyAny>,
        padding: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = PfsPrimeConfig {
            padding: padding.0,
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let hw_f64 = extract_f64_1d(half_widths, "half_widths")?;
        let hh_f64 = extract_f64_1d(half_heights, "half_heights")?;
        let inner = match dtype {
            ScalarDtype::F64 => py
                .detach(|| {
                    PfsPrimeOverlapRemoval::<f64>::new(hw_f64, hh_f64, cfg).map(PfsPrimeInner::F64)
                })
                .map_err(core_error_to_py)?,
            ScalarDtype::F32 => {
                let hw_f32 = vec_f64_to_f32_checked(&hw_f64, "half_widths")?;
                let hh_f32 = vec_f64_to_f32_checked(&hh_f64, "half_heights")?;
                py.detach(|| {
                    PfsPrimeOverlapRemoval::<f32>::new(hw_f32, hh_f32, cfg).map(PfsPrimeInner::F32)
                })
                .map_err(core_error_to_py)?
            }
        };
        Ok(Self { inner })
    }

    /// Create a `PfsPrimeOverlapRemoval` solver for square/circular nodes with equal half-widths and half-heights.
    #[staticmethod]
    #[pyo3(
        signature = (radii, *, padding = StrictF64(0.0), seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(radii, *, padding=0.0, seed=42, dtype='float64')"
    )]
    fn from_radii(
        py: Python<'_>,
        radii: &Bound<'_, PyAny>,
        padding: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = PfsPrimeConfig {
            padding: padding.0,
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let r_f64 = extract_f64_1d(radii, "radii")?;
        let inner = match dtype {
            ScalarDtype::F64 => py
                .detach(|| {
                    PfsPrimeOverlapRemoval::<f64>::from_radii(r_f64, cfg).map(PfsPrimeInner::F64)
                })
                .map_err(core_error_to_py)?,
            ScalarDtype::F32 => {
                let r_f32 = vec_f64_to_f32_checked(&r_f64, "radii")?;
                py.detach(|| {
                    PfsPrimeOverlapRemoval::<f32>::from_radii(r_f32, cfg).map(PfsPrimeInner::F32)
                })
                .map_err(core_error_to_py)?
            }
        };
        Ok(Self { inner })
    }

    /// Execute the single-pass `PFS'` overlap removal step in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` `PFS'` steps in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run `PFS'` overlap removal to completion in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Number of overlapping rectangle pairs remaining after the most recent step.
    #[getter]
    fn last_overlap_count(&self) -> usize {
        match &self.inner {
            PfsPrimeInner::F64(s) => s.last_overlap_count(),
            PfsPrimeInner::F32(s) => s.last_overlap_count(),
        }
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

    /// Total number of `PFS'` steps (`1`).
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
            PfsPrimeInner::F64(s) => s.num_nodes(),
            PfsPrimeInner::F32(s) => s.num_nodes(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `PfsPrimeOverlapRemoval` solver.
    fn __repr__(&self) -> String {
        format!(
            "PfsPrimeOverlapRemoval(num_nodes={}, last_overlap_count={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.last_overlap_count(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}
