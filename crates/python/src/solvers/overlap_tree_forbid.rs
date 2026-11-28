//! Tree and SGD-based node overlap removal (`GTreeOverlapRemoval`, `ForbidOverlapRemoval`).

use egraph::core::NodeExtents2d;
use egraph::layout::{
    ForbidConfig, ForbidOverlapRemoval, ForbidVariant, GTreeConfig, GTreeOverlapRemoval,
};
use pyo3::prelude::*;

use crate::array_extract::extract_f64_1d;
use crate::errors::{core_error_to_py, unsupported_operation_err};
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, normalize_enum_key, solver_status_str,
};
use crate::impl_dual_solver;
use crate::layout_buffers::PyLayout2d;
use crate::solver_common::{
    DualIterativeSolver, SolverAction, execute_solver_action, format_optional_usize,
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
#[pyclass(name = "GTreeOverlapRemoval", module = "egraph._core")]
pub struct PyGTreeOverlapRemoval {
    inner: GTreeInner,
}

#[pymethods]
impl PyGTreeOverlapRemoval {
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
        let cfg = GTreeConfig::new()
            .with_max_iterations(max_iterations.0)
            .with_padding(padding.0)
            .with_compact_t_max(compact_t_max.map(|v| v.0))
            .with_seed(seed.0);
        cfg.validate().map_err(core_error_to_py)?;
        let hw_f64 = extract_f64_1d(half_widths, "half_widths")?;
        let hh_f64 = extract_f64_1d(half_heights, "half_heights")?;
        let inner = match dtype {
            ScalarDtype::F64 => py
                .detach(|| {
                    let ext = NodeExtents2d::new(&hw_f64, &hh_f64)?;
                    GTreeOverlapRemoval::<f64>::new(ext, cfg).map(GTreeInner::F64)
                })
                .map_err(core_error_to_py)?,
            ScalarDtype::F32 => {
                let hw_f32 = vec_f64_to_f32_checked(&hw_f64, "half_widths")?;
                let hh_f32 = vec_f64_to_f32_checked(&hh_f64, "half_heights")?;
                py.detach(|| {
                    let ext = NodeExtents2d::new(&hw_f32, &hh_f32)?;
                    GTreeOverlapRemoval::<f32>::new(ext, cfg).map(GTreeInner::F32)
                })
                .map_err(core_error_to_py)?
            }
        };
        Ok(Self { inner })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
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
        let cfg = GTreeConfig::new()
            .with_max_iterations(max_iterations.0)
            .with_padding(padding.0)
            .with_compact_t_max(compact_t_max.map(|v| v.0))
            .with_seed(seed.0);
        cfg.validate().map_err(core_error_to_py)?;
        let r_f64 = extract_f64_1d(radii, "radii")?;
        let inner = match dtype {
            ScalarDtype::F64 => py
                .detach(|| GTreeOverlapRemoval::<f64>::from_radii(&r_f64, cfg).map(GTreeInner::F64))
                .map_err(core_error_to_py)?,
            ScalarDtype::F32 => {
                let r_f32 = vec_f64_to_f32_checked(&r_f64, "radii")?;
                py.detach(|| {
                    GTreeOverlapRemoval::<f32>::from_radii(&r_f32, cfg).map(GTreeInner::F32)
                })
                .map_err(core_error_to_py)?
            }
        };
        Ok(Self { inner })
    }

    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    #[getter]
    fn last_overlap_count(&self) -> usize {
        match &self.inner {
            GTreeInner::F64(s) => s.last_overlap_count(),
            GTreeInner::F32(s) => s.last_overlap_count(),
        }
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
    fn num_nodes(&self) -> usize {
        match &self.inner {
            GTreeInner::F64(s) => s.num_nodes(),
            GTreeInner::F32(s) => s.num_nodes(),
        }
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    fn __repr__(&self) -> String {
        format!(
            "GTreeOverlapRemoval(num_nodes={}, last_overlap_count={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.last_overlap_count(),
            self.status(),
            self.iteration(),
            format_optional_usize(self.total_iterations()),
            self.dtype()
        )
    }
}

/// Cooperative Giovannangeli et al. (2022) FORBID stochastic gradient descent overlap removal solver.
#[pyclass(name = "ForbidOverlapRemoval", module = "egraph._core")]
pub struct PyForbidOverlapRemoval {
    inner: ForbidInner,
}

#[pymethods]
impl PyForbidOverlapRemoval {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            half_widths,
            half_heights,
            *,
            variant = "sparse_forbid",
            max_scale_iters = StrictUsize(20),
            sgd_iterations = StrictUsize(30),
            sparse_k = StrictUsize(30),
            overlap_weight_k = StrictF64(1.0),
            scale_step = StrictF64(0.1),
            padding = StrictF64(0.0),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(half_widths, half_heights, *, variant='sparse_forbid', max_scale_iters=20, sgd_iterations=30, sparse_k=30, overlap_weight_k=1.0, scale_step=0.1, padding=0.0, seed=42, dtype='float64')"
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
        let parsed_variant = parse_forbid_variant(variant).map_err(unsupported_operation_err)?;
        let cfg = ForbidConfig::new()
            .with_variant(parsed_variant)
            .with_max_scale_iters(max_scale_iters.0)
            .with_sgd_iterations(sgd_iterations.0)
            .with_sparse_k(sparse_k.0)
            .with_overlap_weight_k(overlap_weight_k.0)
            .with_scale_step(scale_step.0)
            .with_padding(padding.0)
            .with_seed(seed.0);
        cfg.validate().map_err(core_error_to_py)?;
        let hw_f64 = extract_f64_1d(half_widths, "half_widths")?;
        let hh_f64 = extract_f64_1d(half_heights, "half_heights")?;
        let inner = match dtype {
            ScalarDtype::F64 => py
                .detach(|| {
                    let ext = NodeExtents2d::new(&hw_f64, &hh_f64)?;
                    ForbidOverlapRemoval::<f64>::new(ext, cfg).map(ForbidInner::F64)
                })
                .map_err(core_error_to_py)?,
            ScalarDtype::F32 => {
                let hw_f32 = vec_f64_to_f32_checked(&hw_f64, "half_widths")?;
                let hh_f32 = vec_f64_to_f32_checked(&hh_f64, "half_heights")?;
                py.detach(|| {
                    let ext = NodeExtents2d::new(&hw_f32, &hh_f32)?;
                    ForbidOverlapRemoval::<f32>::new(ext, cfg).map(ForbidInner::F32)
                })
                .map_err(core_error_to_py)?
            }
        };
        Ok(Self { inner })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            radii,
            *,
            variant = "sparse_forbid",
            max_scale_iters = StrictUsize(20),
            sgd_iterations = StrictUsize(30),
            sparse_k = StrictUsize(30),
            overlap_weight_k = StrictF64(1.0),
            scale_step = StrictF64(0.1),
            padding = StrictF64(0.0),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(radii, *, variant='sparse_forbid', max_scale_iters=20, sgd_iterations=30, sparse_k=30, overlap_weight_k=1.0, scale_step=0.1, padding=0.0, seed=42, dtype='float64')"
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
        let parsed_variant = parse_forbid_variant(variant).map_err(unsupported_operation_err)?;
        let cfg = ForbidConfig::new()
            .with_variant(parsed_variant)
            .with_max_scale_iters(max_scale_iters.0)
            .with_sgd_iterations(sgd_iterations.0)
            .with_sparse_k(sparse_k.0)
            .with_overlap_weight_k(overlap_weight_k.0)
            .with_scale_step(scale_step.0)
            .with_padding(padding.0)
            .with_seed(seed.0);
        cfg.validate().map_err(core_error_to_py)?;
        let r_f64 = extract_f64_1d(radii, "radii")?;
        let inner = match dtype {
            ScalarDtype::F64 => py
                .detach(|| {
                    ForbidOverlapRemoval::<f64>::from_radii(&r_f64, cfg).map(ForbidInner::F64)
                })
                .map_err(core_error_to_py)?,
            ScalarDtype::F32 => {
                let r_f32 = vec_f64_to_f32_checked(&r_f64, "radii")?;
                py.detach(|| {
                    ForbidOverlapRemoval::<f32>::from_radii(&r_f32, cfg).map(ForbidInner::F32)
                })
                .map_err(core_error_to_py)?
            }
        };
        Ok(Self { inner })
    }

    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    #[getter]
    fn last_overlap_count(&self) -> usize {
        match &self.inner {
            ForbidInner::F64(s) => s.last_overlap_count(),
            ForbidInner::F32(s) => s.last_overlap_count(),
        }
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
    fn num_nodes(&self) -> usize {
        match &self.inner {
            ForbidInner::F64(s) => s.num_nodes(),
            ForbidInner::F32(s) => s.num_nodes(),
        }
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    fn __repr__(&self) -> String {
        format!(
            "ForbidOverlapRemoval(num_nodes={}, last_overlap_count={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.last_overlap_count(),
            self.status(),
            self.iteration(),
            format_optional_usize(self.total_iterations()),
            self.dtype()
        )
    }
}
