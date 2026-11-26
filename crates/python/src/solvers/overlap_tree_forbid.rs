//! Tree and SGD-based node overlap removal (`GTreeOverlapRemoval`, `ForbidOverlapRemoval`).

use egraph::layout as egraph_layout;
use egraph_layout::{
    ForbidConfig, ForbidOverlapRemoval, ForbidVariant, GTreeConfig, GTreeOverlapRemoval,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::array_extract::extract_f64_1d;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, normalize_enum_key,
    solver_status_str,
};
use crate::layout_buffers::PyLayout2d;
use crate::solver_common::{
    DualIterativeSolver, SolverAction, execute_solver_action, impl_dual_solver,
    vec_f64_to_f32_checked,
};

pub(super) fn parse_forbid_variant(raw: &str) -> Result<ForbidVariant, String> {
    match normalize_enum_key(raw).as_str() {
        "forbid" => Ok(ForbidVariant::Forbid),
        "forbidprime" | "forbid'" | "forbidp" => Ok(ForbidVariant::ForbidPrime),
        "sparseforbid" | "sparse" => Ok(ForbidVariant::SparseForbid),
        _ => Err(format!(
            "unknown FORBID variant {raw:?}; expected 'forbid', 'forbid_prime', or 'sparse_forbid'"
        )),
    }
}

impl_dual_solver!(GTreeInner, GTreeOverlapRemoval);
impl_dual_solver!(ForbidInner, ForbidOverlapRemoval);

/// Cooperative Nachmanson et al. (2016) GTree proximity-MST node overlap removal solver.
///
/// Supports both `"float64"` and `"float32"` `Layout2d` while releasing the Python GIL.
#[pyclass(name = "GTreeOverlapRemoval", module = "egraph._core")]
pub struct PyGTreeOverlapRemoval {
    inner: GTreeInner,
}

#[pymethods]
impl PyGTreeOverlapRemoval {
    /// Create a `GTreeOverlapRemoval` solver from per-node `half_widths` and `half_heights`.
    ///
    /// When `compact_t_max` is provided, it must satisfy `compact_t_max > 1.0` (`None` disables
    /// compact per-step expansion capping). Values `<= 1.0` raise `ValueError`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            half_widths,
            half_heights,
            *,
            max_iterations = StrictUsize(20),
            padding = StrictF64(0.0),
            compact_t_max = None,
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(half_widths, half_heights, *, max_iterations=20, padding=0.0, compact_t_max=None, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        half_widths: &Bound<'_, PyAny>,
        half_heights: &Bound<'_, PyAny>,
        max_iterations: StrictUsize,
        padding: StrictF64,
        compact_t_max: Option<StrictF64>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = GTreeConfig {
            max_iterations: max_iterations.0,
            padding: padding.0,
            compact_t_max: compact_t_max.map(|v| v.0),
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let hw_f64 = extract_f64_1d(half_widths, "half_widths")?;
        let hh_f64 = extract_f64_1d(half_heights, "half_heights")?;
        let inner = match dtype {
            ScalarDtype::F64 => py
                .detach(|| {
                    GTreeOverlapRemoval::<f64>::new(hw_f64, hh_f64, cfg).map(GTreeInner::F64)
                })
                .map_err(core_error_to_py)?,
            ScalarDtype::F32 => {
                let hw_f32 = vec_f64_to_f32_checked(&hw_f64, "half_widths")?;
                let hh_f32 = vec_f64_to_f32_checked(&hh_f64, "half_heights")?;
                py.detach(|| {
                    GTreeOverlapRemoval::<f32>::new(hw_f32, hh_f32, cfg).map(GTreeInner::F32)
                })
                .map_err(core_error_to_py)?
            }
        };
        Ok(Self { inner })
    }

    /// Create a `GTreeOverlapRemoval` solver for square/circular nodes with equal half-widths and half-heights.
    ///
    /// When `compact_t_max` is provided, it must satisfy `compact_t_max > 1.0` (`None` disables
    /// compact per-step expansion capping). Values `<= 1.0` raise `ValueError`.
    #[staticmethod]
    #[pyo3(
        signature = (
            radii,
            *,
            max_iterations = StrictUsize(20),
            padding = StrictF64(0.0),
            compact_t_max = None,
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(radii, *, max_iterations=20, padding=0.0, compact_t_max=None, seed=42, dtype='float64')"
    )]
    fn from_radii(
        py: Python<'_>,
        radii: &Bound<'_, PyAny>,
        max_iterations: StrictUsize,
        padding: StrictF64,
        compact_t_max: Option<StrictF64>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = GTreeConfig {
            max_iterations: max_iterations.0,
            padding: padding.0,
            compact_t_max: compact_t_max.map(|v| v.0),
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let r_f64 = extract_f64_1d(radii, "radii")?;
        let inner = match dtype {
            ScalarDtype::F64 => py
                .detach(|| GTreeOverlapRemoval::<f64>::from_radii(r_f64, cfg).map(GTreeInner::F64))
                .map_err(core_error_to_py)?,
            ScalarDtype::F32 => {
                let r_f32 = vec_f64_to_f32_checked(&r_f64, "radii")?;
                py.detach(|| {
                    GTreeOverlapRemoval::<f32>::from_radii(r_f32, cfg).map(GTreeInner::F32)
                })
                .map_err(core_error_to_py)?
            }
        };
        Ok(Self { inner })
    }

    /// Execute one GTree overlap removal round in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` GTree overlap removal rounds in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run GTree overlap removal to completion in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Number of overlapping rectangle pairs remaining after the most recent step.
    #[getter]
    fn last_overlap_count(&self) -> usize {
        match &self.inner {
            GTreeInner::F64(s) => s.last_overlap_count(),
            GTreeInner::F32(s) => s.last_overlap_count(),
        }
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of GTree rounds executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Configured maximum number of GTree rounds.
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
            GTreeInner::F64(s) => s.num_nodes(),
            GTreeInner::F32(s) => s.num_nodes(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `GTreeOverlapRemoval` solver.
    fn __repr__(&self) -> String {
        format!(
            "GTreeOverlapRemoval(num_nodes={}, last_overlap_count={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.last_overlap_count(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}

/// Cooperative Giovannangeli et al. (2022) FORBID stochastic gradient descent overlap removal solver.
///
/// Supports `"forbid"`, `"forbid_prime"`, and `"sparse_forbid"` variants across both
/// `"float64"` and `"float32"` `Layout2d` while releasing the Python GIL.
#[pyclass(name = "ForbidOverlapRemoval", module = "egraph._core")]
pub struct PyForbidOverlapRemoval {
    inner: ForbidInner,
}

#[pymethods]
impl PyForbidOverlapRemoval {
    /// Create a `ForbidOverlapRemoval` solver from per-node `half_widths` and `half_heights`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            half_widths,
            half_heights,
            *,
            variant = "forbid",
            max_scale_iters = StrictUsize(20),
            sgd_iterations = StrictUsize(30),
            sparse_k = StrictUsize(30),
            overlap_weight_k = StrictF64(1.0),
            scale_step = StrictF64(0.1),
            padding = StrictF64(0.0),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(half_widths, half_heights, *, variant='forbid', max_scale_iters=20, sgd_iterations=30, sparse_k=30, overlap_weight_k=1.0, scale_step=0.1, padding=0.0, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        half_widths: &Bound<'_, PyAny>,
        half_heights: &Bound<'_, PyAny>,
        variant: &str,
        max_scale_iters: StrictUsize,
        sgd_iterations: StrictUsize,
        sparse_k: StrictUsize,
        overlap_weight_k: StrictF64,
        scale_step: StrictF64,
        padding: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let parsed_variant = parse_forbid_variant(variant).map_err(PyValueError::new_err)?;
        let cfg = ForbidConfig {
            variant: parsed_variant,
            max_scale_iters: max_scale_iters.0,
            sgd_iterations: sgd_iterations.0,
            sparse_k: sparse_k.0,
            overlap_weight_k: overlap_weight_k.0,
            scale_step: scale_step.0,
            padding: padding.0,
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let hw_f64 = extract_f64_1d(half_widths, "half_widths")?;
        let hh_f64 = extract_f64_1d(half_heights, "half_heights")?;
        let inner = match dtype {
            ScalarDtype::F64 => py
                .detach(|| {
                    ForbidOverlapRemoval::<f64>::new(hw_f64, hh_f64, cfg).map(ForbidInner::F64)
                })
                .map_err(core_error_to_py)?,
            ScalarDtype::F32 => {
                let hw_f32 = vec_f64_to_f32_checked(&hw_f64, "half_widths")?;
                let hh_f32 = vec_f64_to_f32_checked(&hh_f64, "half_heights")?;
                py.detach(|| {
                    ForbidOverlapRemoval::<f32>::new(hw_f32, hh_f32, cfg).map(ForbidInner::F32)
                })
                .map_err(core_error_to_py)?
            }
        };
        Ok(Self { inner })
    }

    /// Create a `ForbidOverlapRemoval` solver for square/circular nodes with equal half-widths and half-heights.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            radii,
            *,
            variant = "forbid",
            max_scale_iters = StrictUsize(20),
            sgd_iterations = StrictUsize(30),
            sparse_k = StrictUsize(30),
            overlap_weight_k = StrictF64(1.0),
            scale_step = StrictF64(0.1),
            padding = StrictF64(0.0),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(radii, *, variant='forbid', max_scale_iters=20, sgd_iterations=30, sparse_k=30, overlap_weight_k=1.0, scale_step=0.1, padding=0.0, seed=42, dtype='float64')"
    )]
    fn from_radii(
        py: Python<'_>,
        radii: &Bound<'_, PyAny>,
        variant: &str,
        max_scale_iters: StrictUsize,
        sgd_iterations: StrictUsize,
        sparse_k: StrictUsize,
        overlap_weight_k: StrictF64,
        scale_step: StrictF64,
        padding: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let parsed_variant = parse_forbid_variant(variant).map_err(PyValueError::new_err)?;
        let cfg = ForbidConfig {
            variant: parsed_variant,
            max_scale_iters: max_scale_iters.0,
            sgd_iterations: sgd_iterations.0,
            sparse_k: sparse_k.0,
            overlap_weight_k: overlap_weight_k.0,
            scale_step: scale_step.0,
            padding: padding.0,
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let r_f64 = extract_f64_1d(radii, "radii")?;
        let inner = match dtype {
            ScalarDtype::F64 => py
                .detach(|| {
                    ForbidOverlapRemoval::<f64>::from_radii(r_f64, cfg).map(ForbidInner::F64)
                })
                .map_err(core_error_to_py)?,
            ScalarDtype::F32 => {
                let r_f32 = vec_f64_to_f32_checked(&r_f64, "radii")?;
                py.detach(|| {
                    ForbidOverlapRemoval::<f32>::from_radii(r_f32, cfg).map(ForbidInner::F32)
                })
                .map_err(core_error_to_py)?
            }
        };
        Ok(Self { inner })
    }

    /// Execute one FORBID binary-search scaling iteration in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` FORBID scaling iterations in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run FORBID overlap removal to completion in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Number of overlapping rectangle pairs remaining after the most recent step.
    #[getter]
    fn last_overlap_count(&self) -> usize {
        match &self.inner {
            ForbidInner::F64(s) => s.last_overlap_count(),
            ForbidInner::F32(s) => s.last_overlap_count(),
        }
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of FORBID scaling iterations executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Configured maximum number of FORBID scaling iterations.
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
            ForbidInner::F64(s) => s.num_nodes(),
            ForbidInner::F32(s) => s.num_nodes(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `ForbidOverlapRemoval` solver.
    fn __repr__(&self) -> String {
        format!(
            "ForbidOverlapRemoval(num_nodes={}, last_overlap_count={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.last_overlap_count(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}
