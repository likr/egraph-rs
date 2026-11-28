//! Proximity (`PrismOverlapRemoval`) and scan-line (`PfsPrimeOverlapRemoval`) node overlap removal.

use egraph::core::{CoreError, LayoutTransform, NodeExtents2d};
use egraph::layout::{PfsPrimeConfig, PfsPrimeOverlapRemoval, PrismConfig, PrismOverlapRemoval};
use pyo3::prelude::*;

use crate::array_extract::extract_f64_1d;
use crate::errors::core_error_to_py;
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictU64, StrictUsize, solver_status_str};
use crate::impl_dual_solver;
use crate::layout_buffers::PyLayout2d;
use crate::solver_common::{
    DualIterativeSolver, SolverAction, SolverLayout, execute_solver_action, extract_solver_layout,
    format_optional_usize, vec_f64_to_f32_checked, write_back_solver_layout,
};

impl_dual_solver!(PrismInner, PrismOverlapRemoval);

/// Cooperative Gansner-Hu (2010) PRISM proximity-stress node overlap removal solver.
#[pyclass(name = "PrismOverlapRemoval", module = "egraph._core")]
pub struct PyPrismOverlapRemoval {
    inner: PrismInner,
}

#[pymethods]
impl PyPrismOverlapRemoval {
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
        let cfg = PrismConfig::new()
            .with_max_iterations(max_iterations.0)
            .with_max_scale_factor(max_scale_factor.0)
            .with_cg_max_iters(cg_max_iters.0)
            .with_cg_tolerance(cg_tolerance.0)
            .with_padding(padding.0)
            .with_seed(seed.0);
        cfg.validate().map_err(core_error_to_py)?;
        let hw_f64 = extract_f64_1d(half_widths, "half_widths")?;
        let hh_f64 = extract_f64_1d(half_heights, "half_heights")?;
        let inner = match dtype {
            ScalarDtype::F64 => py
                .detach(|| {
                    let ext = NodeExtents2d::new(&hw_f64, &hh_f64)?;
                    PrismOverlapRemoval::<f64>::new(ext, cfg).map(PrismInner::F64)
                })
                .map_err(core_error_to_py)?,
            ScalarDtype::F32 => {
                let hw_f32 = vec_f64_to_f32_checked(&hw_f64, "half_widths")?;
                let hh_f32 = vec_f64_to_f32_checked(&hh_f64, "half_heights")?;
                py.detach(|| {
                    let ext = NodeExtents2d::new(&hw_f32, &hh_f32)?;
                    PrismOverlapRemoval::<f32>::new(ext, cfg).map(PrismInner::F32)
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
        let cfg = PrismConfig::new()
            .with_max_iterations(max_iterations.0)
            .with_max_scale_factor(max_scale_factor.0)
            .with_cg_max_iters(cg_max_iters.0)
            .with_cg_tolerance(cg_tolerance.0)
            .with_padding(padding.0)
            .with_seed(seed.0);
        cfg.validate().map_err(core_error_to_py)?;
        let r_f64 = extract_f64_1d(radii, "radii")?;
        let inner = match dtype {
            ScalarDtype::F64 => py
                .detach(|| PrismOverlapRemoval::<f64>::from_radii(&r_f64, cfg).map(PrismInner::F64))
                .map_err(core_error_to_py)?,
            ScalarDtype::F32 => {
                let r_f32 = vec_f64_to_f32_checked(&r_f64, "radii")?;
                py.detach(|| {
                    PrismOverlapRemoval::<f32>::from_radii(&r_f32, cfg).map(PrismInner::F32)
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
            PrismInner::F64(s) => s.last_overlap_count(),
            PrismInner::F32(s) => s.last_overlap_count(),
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
            PrismInner::F64(s) => s.num_nodes(),
            PrismInner::F32(s) => s.num_nodes(),
        }
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    fn __repr__(&self) -> String {
        format!(
            "PrismOverlapRemoval(num_nodes={}, last_overlap_count={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.last_overlap_count(),
            self.status(),
            self.iteration(),
            format_optional_usize(self.total_iterations()),
            self.dtype()
        )
    }
}

enum PfsPrimeInner {
    F64(PfsPrimeOverlapRemoval<f64>),
    F32(PfsPrimeOverlapRemoval<f32>),
}

/// Nachmanson et al. / Chen et al. (2020) `PFS'` dual-axis scan-line node overlap removal transform (`LayoutTransform`).
#[pyclass(name = "PfsPrimeOverlapRemoval", module = "egraph._core")]
pub struct PyPfsPrimeOverlapRemoval {
    inner: PfsPrimeInner,
}

#[pymethods]
impl PyPfsPrimeOverlapRemoval {
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
        let cfg = PfsPrimeConfig::new()
            .with_padding(padding.0)
            .with_seed(seed.0);
        cfg.validate().map_err(core_error_to_py)?;
        let hw_f64 = extract_f64_1d(half_widths, "half_widths")?;
        let hh_f64 = extract_f64_1d(half_heights, "half_heights")?;
        let inner = match dtype {
            ScalarDtype::F64 => py
                .detach(|| {
                    let ext = NodeExtents2d::new(&hw_f64, &hh_f64)?;
                    PfsPrimeOverlapRemoval::<f64>::new(ext, cfg).map(PfsPrimeInner::F64)
                })
                .map_err(core_error_to_py)?,
            ScalarDtype::F32 => {
                let hw_f32 = vec_f64_to_f32_checked(&hw_f64, "half_widths")?;
                let hh_f32 = vec_f64_to_f32_checked(&hh_f64, "half_heights")?;
                py.detach(|| {
                    let ext = NodeExtents2d::new(&hw_f32, &hh_f32)?;
                    PfsPrimeOverlapRemoval::<f32>::new(ext, cfg).map(PfsPrimeInner::F32)
                })
                .map_err(core_error_to_py)?
            }
        };
        Ok(Self { inner })
    }

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
        let cfg = PfsPrimeConfig::new()
            .with_padding(padding.0)
            .with_seed(seed.0);
        cfg.validate().map_err(core_error_to_py)?;
        let r_f64 = extract_f64_1d(radii, "radii")?;
        let inner = match dtype {
            ScalarDtype::F64 => py
                .detach(|| {
                    PfsPrimeOverlapRemoval::<f64>::from_radii(&r_f64, cfg).map(PfsPrimeInner::F64)
                })
                .map_err(core_error_to_py)?,
            ScalarDtype::F32 => {
                let r_f32 = vec_f64_to_f32_checked(&r_f64, "radii")?;
                py.detach(|| {
                    PfsPrimeOverlapRemoval::<f32>::from_radii(&r_f32, cfg).map(PfsPrimeInner::F32)
                })
                .map_err(core_error_to_py)?
            }
        };
        Ok(Self { inner })
    }

    /// Apply single-pass `PFS'` overlap removal in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn apply(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<()> {
        let dtype = match &self.inner {
            PfsPrimeInner::F64(_) => ScalarDtype::F64,
            PfsPrimeInner::F32(_) => ScalarDtype::F32,
        };
        let mut soa = extract_solver_layout(layout, dtype)?;
        py.detach(|| match (&mut self.inner, &mut soa) {
            (PfsPrimeInner::F64(s), SolverLayout::F64(l)) => s.apply(l),
            (PfsPrimeInner::F32(s), SolverLayout::F32(l)) => s.apply(l),
            _ => Err(CoreError::invalid_input("internal layout dtype mismatch")),
        })
        .map_err(core_error_to_py)?;
        write_back_solver_layout(layout, &soa)
    }

    #[getter]
    fn last_overlap_count(&self) -> usize {
        match &self.inner {
            PfsPrimeInner::F64(s) => s.last_overlap_count(),
            PfsPrimeInner::F32(s) => s.last_overlap_count(),
        }
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        match &self.inner {
            PfsPrimeInner::F64(s) => s.num_nodes(),
            PfsPrimeInner::F32(s) => s.num_nodes(),
        }
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        match &self.inner {
            PfsPrimeInner::F64(_) => "float64",
            PfsPrimeInner::F32(_) => "float32",
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PfsPrimeOverlapRemoval(num_nodes={}, last_overlap_count={}, dtype='{}')",
            self.num_nodes(),
            self.last_overlap_count(),
            self.dtype()
        )
    }
}
