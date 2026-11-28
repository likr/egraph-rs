//! Force-directed edge bundling (`FdebSolver`) bindings over UAF-safe `EdgePolylines`.

use egraph::core::{IterativeSolver, Layout2d, Layout3d, LayoutNd, SolverStatus};
use egraph::edge_routing::{FdebConfig, FdebSolver};
use pyo3::prelude::*;

use crate::edge_polylines::{PolylineTarget, PyEdgePolylines};
use crate::errors::{core_error_to_py, dimension_mismatch_err};
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictUsize, solver_status_str};
use crate::solver_common::SolverAction;

enum FdebInner {
    L2F64(FdebSolver<Layout2d<f64>>),
    L2F32(FdebSolver<Layout2d<f32>>),
    L3F64(FdebSolver<Layout3d<f64>>),
    L3F32(FdebSolver<Layout3d<f32>>),
    NdF64(FdebSolver<LayoutNd<f64>>),
    NdF32(FdebSolver<LayoutNd<f32>>),
}

impl FdebInner {
    fn target_kind(&self) -> PolylineTarget {
        match self {
            Self::L2F64(_) | Self::L2F32(_) => PolylineTarget::Layout2d,
            Self::L3F64(_) | Self::L3F32(_) => PolylineTarget::Layout3d,
            Self::NdF64(_) | Self::NdF32(_) => PolylineTarget::LayoutNd,
        }
    }

    fn dtype(&self) -> ScalarDtype {
        match self {
            Self::L2F64(_) | Self::L3F64(_) | Self::NdF64(_) => ScalarDtype::F64,
            Self::L2F32(_) | Self::L3F32(_) | Self::NdF32(_) => ScalarDtype::F32,
        }
    }

    fn num_edges(&self) -> usize {
        match self {
            Self::L2F64(s) => s.num_edges(),
            Self::L2F32(s) => s.num_edges(),
            Self::L3F64(s) => s.num_edges(),
            Self::L3F32(s) => s.num_edges(),
            Self::NdF64(s) => s.num_edges(),
            Self::NdF32(s) => s.num_edges(),
        }
    }

    fn current_cycle(&self) -> usize {
        match self {
            Self::L2F64(s) => s.current_cycle(),
            Self::L2F32(s) => s.current_cycle(),
            Self::L3F64(s) => s.current_cycle(),
            Self::L3F32(s) => s.current_cycle(),
            Self::NdF64(s) => s.current_cycle(),
            Self::NdF32(s) => s.current_cycle(),
        }
    }

    fn pts_per_edge(&self) -> usize {
        match self {
            Self::L2F64(s) => s.pts_per_edge(),
            Self::L2F32(s) => s.pts_per_edge(),
            Self::L3F64(s) => s.pts_per_edge(),
            Self::L3F32(s) => s.pts_per_edge(),
            Self::NdF64(s) => s.pts_per_edge(),
            Self::NdF32(s) => s.pts_per_edge(),
        }
    }

    fn status(&self) -> SolverStatus {
        match self {
            Self::L2F64(s) => s.status(),
            Self::L2F32(s) => s.status(),
            Self::L3F64(s) => s.status(),
            Self::L3F32(s) => s.status(),
            Self::NdF64(s) => s.status(),
            Self::NdF32(s) => s.status(),
        }
    }

    fn iteration(&self) -> usize {
        match self {
            Self::L2F64(s) => s.iteration(),
            Self::L2F32(s) => s.iteration(),
            Self::L3F64(s) => s.iteration(),
            Self::L3F32(s) => s.iteration(),
            Self::NdF64(s) => s.iteration(),
            Self::NdF32(s) => s.iteration(),
        }
    }

    fn total_iterations(&self) -> Option<usize> {
        match self {
            Self::L2F64(s) => s.total_iterations(),
            Self::L2F32(s) => s.total_iterations(),
            Self::L3F64(s) => s.total_iterations(),
            Self::L3F32(s) => s.total_iterations(),
            Self::NdF64(s) => s.total_iterations(),
            Self::NdF32(s) => s.total_iterations(),
        }
    }

    fn execute_action(
        &mut self,
        py: Python<'_>,
        polylines: &Bound<'_, PyEdgePolylines>,
        action: SolverAction,
    ) -> PyResult<&'static str> {
        let (p_target, p_dtype, dims) = {
            let b = polylines.try_borrow()?;
            (b.target_kind(), b.dtype_kind(), b.dims_count())
        };
        if p_target != self.target_kind() || p_dtype != self.dtype() {
            return Err(dimension_mismatch_err(format!(
                "FdebSolver expected target='{}' dtype='{}', got target='{}' dtype='{}'",
                self.target_kind().as_str(),
                self.dtype().as_str(),
                p_target.as_str(),
                p_dtype.as_str()
            )));
        }

        macro_rules! run_target {
            ($solver:expr, $core_poly:expr, $to_flat:expr, $write_fn:ident, $dims_val:expr) => {{
                let mut poly = $core_poly;
                let status = py
                    .detach(|| match action {
                        SolverAction::Step => IterativeSolver::step($solver, &mut poly),
                        SolverAction::RunSteps(steps) => {
                            IterativeSolver::run_steps($solver, &mut poly, steps)
                        }
                        SolverAction::Run => IterativeSolver::run($solver, &mut poly),
                    })
                    .map_err(core_error_to_py)?;
                let (offs, pts) = poly.into_parts();
                let flat = $to_flat(&pts);
                let mut borrowed_mut = polylines.try_borrow_mut()?;
                borrowed_mut.$write_fn(py, offs, flat, $dims_val)?;
                Ok(solver_status_str(status))
            }};
        }

        let b = polylines.try_borrow()?;
        match self {
            Self::L2F64(s) => {
                let core_poly = b.to_core_2d_f64(py)?;
                drop(b);
                run_target!(
                    s,
                    core_poly,
                    |p: &Layout2d<f64>| p.to_interleaved(),
                    write_back_f64,
                    2
                )
            }
            Self::L2F32(s) => {
                let core_poly = b.to_core_2d_f32(py)?;
                drop(b);
                run_target!(
                    s,
                    core_poly,
                    |p: &Layout2d<f32>| p.to_interleaved(),
                    write_back_f32,
                    2
                )
            }
            Self::L3F64(s) => {
                let core_poly = b.to_core_3d_f64(py)?;
                drop(b);
                run_target!(
                    s,
                    core_poly,
                    |p: &Layout3d<f64>| p.to_interleaved(),
                    write_back_f64,
                    3
                )
            }
            Self::L3F32(s) => {
                let core_poly = b.to_core_3d_f32(py)?;
                drop(b);
                run_target!(
                    s,
                    core_poly,
                    |p: &Layout3d<f32>| p.to_interleaved(),
                    write_back_f32,
                    3
                )
            }
            Self::NdF64(s) => {
                let core_poly = b.to_core_nd_f64(py)?;
                drop(b);
                run_target!(
                    s,
                    core_poly,
                    |p: &LayoutNd<f64>| p.coords().to_vec(),
                    write_back_f64,
                    dims
                )
            }
            Self::NdF32(s) => {
                let core_poly = b.to_core_nd_f32(py)?;
                drop(b);
                run_target!(
                    s,
                    core_poly,
                    |p: &LayoutNd<f32>| p.coords().to_vec(),
                    write_back_f32,
                    dims
                )
            }
        }
    }
}

/// Cooperative Force-Directed Edge Bundling (`FDEB`) solver mutating an `EdgePolylines` buffer.
#[pyclass(name = "FdebSolver", module = "egraph._core")]
pub struct PyFdebSolver {
    inner: FdebInner,
    point_dims: usize,
}

#[pymethods]
impl PyFdebSolver {
    /// Construct an `FdebSolver` from an initial `EdgePolylines` buffer.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            polylines,
            *,
            cycles = StrictUsize(6),
            s0 = StrictF64(0.1),
            steps_per_cycle = StrictUsize(90),
            s_step = StrictF64(0.5),
            i_step = StrictF64(0.6666666666666666),
            spring_k = StrictF64(0.1),
            minimum_edge_compatibility = StrictF64(0.6)
        ),
        text_signature = "(polylines, *, cycles=6, s0=0.1, steps_per_cycle=90, s_step=0.5, i_step=0.6666666666666666, spring_k=0.1, minimum_edge_compatibility=0.6)"
    )]
    fn new(
        py: Python<'_>,
        polylines: &PyEdgePolylines,
        cycles: StrictUsize,
        s0: StrictF64,
        steps_per_cycle: StrictUsize,
        s_step: StrictF64,
        i_step: StrictF64,
        spring_k: StrictF64,
        minimum_edge_compatibility: StrictF64,
    ) -> PyResult<Self> {
        let cfg = FdebConfig::new()
            .with_cycles(cycles.0)
            .with_s0(s0.0)
            .with_steps_per_cycle(steps_per_cycle.0)
            .with_s_step(s_step.0)
            .with_i_step(i_step.0)
            .with_spring_k(spring_k.0)
            .with_minimum_edge_compatibility(minimum_edge_compatibility.0);
        cfg.validate().map_err(core_error_to_py)?;
        let dims = polylines.dims_count();
        let inner = match (polylines.target_kind(), polylines.dtype_kind()) {
            (PolylineTarget::Layout2d, ScalarDtype::F64) => {
                let core_poly = polylines.to_core_2d_f64(py)?;
                py.detach(|| FdebSolver::new(&core_poly, cfg).map(FdebInner::L2F64))
                    .map_err(core_error_to_py)?
            }
            (PolylineTarget::Layout2d, ScalarDtype::F32) => {
                let core_poly = polylines.to_core_2d_f32(py)?;
                py.detach(|| FdebSolver::new(&core_poly, cfg).map(FdebInner::L2F32))
                    .map_err(core_error_to_py)?
            }
            (PolylineTarget::Layout3d, ScalarDtype::F64) => {
                let core_poly = polylines.to_core_3d_f64(py)?;
                py.detach(|| FdebSolver::new(&core_poly, cfg).map(FdebInner::L3F64))
                    .map_err(core_error_to_py)?
            }
            (PolylineTarget::Layout3d, ScalarDtype::F32) => {
                let core_poly = polylines.to_core_3d_f32(py)?;
                py.detach(|| FdebSolver::new(&core_poly, cfg).map(FdebInner::L3F32))
                    .map_err(core_error_to_py)?
            }
            (PolylineTarget::LayoutNd, ScalarDtype::F64) => {
                let core_poly = polylines.to_core_nd_f64(py)?;
                py.detach(|| FdebSolver::new(&core_poly, cfg).map(FdebInner::NdF64))
                    .map_err(core_error_to_py)?
            }
            (PolylineTarget::LayoutNd, ScalarDtype::F32) => {
                let core_poly = polylines.to_core_nd_f32(py)?;
                py.detach(|| FdebSolver::new(&core_poly, cfg).map(FdebInner::NdF32))
                    .map_err(core_error_to_py)?
            }
        };
        Ok(Self {
            inner,
            point_dims: dims,
        })
    }

    #[pyo3(signature = (polylines), text_signature = "($self, polylines)")]
    fn validate_target(&self, py: Python<'_>, polylines: &PyEdgePolylines) -> PyResult<()> {
        if polylines.target_kind() != self.inner.target_kind()
            || polylines.dtype_kind() != self.inner.dtype()
        {
            return Err(dimension_mismatch_err(format!(
                "FdebSolver expected target='{}' dtype='{}', got target='{}' dtype='{}'",
                self.inner.target_kind().as_str(),
                self.inner.dtype().as_str(),
                polylines.target_kind().as_str(),
                polylines.dtype_kind().as_str()
            )));
        }
        match &self.inner {
            FdebInner::L2F64(s) => {
                let p = polylines.to_core_2d_f64(py)?;
                py.detach(|| s.validate_target(&p))
                    .map_err(core_error_to_py)
            }
            FdebInner::L2F32(s) => {
                let p = polylines.to_core_2d_f32(py)?;
                py.detach(|| s.validate_target(&p))
                    .map_err(core_error_to_py)
            }
            FdebInner::L3F64(s) => {
                let p = polylines.to_core_3d_f64(py)?;
                py.detach(|| s.validate_target(&p))
                    .map_err(core_error_to_py)
            }
            FdebInner::L3F32(s) => {
                let p = polylines.to_core_3d_f32(py)?;
                py.detach(|| s.validate_target(&p))
                    .map_err(core_error_to_py)
            }
            FdebInner::NdF64(s) => {
                let p = polylines.to_core_nd_f64(py)?;
                py.detach(|| s.validate_target(&p))
                    .map_err(core_error_to_py)
            }
            FdebInner::NdF32(s) => {
                let p = polylines.to_core_nd_f32(py)?;
                py.detach(|| s.validate_target(&p))
                    .map_err(core_error_to_py)
            }
        }
    }

    #[pyo3(signature = (polylines), text_signature = "($self, polylines)")]
    fn step(
        &mut self,
        py: Python<'_>,
        polylines: &Bound<'_, PyEdgePolylines>,
    ) -> PyResult<&'static str> {
        self.inner.execute_action(py, polylines, SolverAction::Step)
    }

    #[pyo3(signature = (polylines, steps), text_signature = "($self, polylines, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        polylines: &Bound<'_, PyEdgePolylines>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        self.inner
            .execute_action(py, polylines, SolverAction::RunSteps(steps.0))
    }

    #[pyo3(signature = (polylines), text_signature = "($self, polylines)")]
    fn run(
        &mut self,
        py: Python<'_>,
        polylines: &Bound<'_, PyEdgePolylines>,
    ) -> PyResult<&'static str> {
        self.inner.execute_action(py, polylines, SolverAction::Run)
    }

    #[getter]
    fn target(&self) -> &'static str {
        self.inner.target_kind().as_str()
    }
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }
    #[getter]
    fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }
    #[getter]
    fn point_dims(&self) -> usize {
        self.point_dims
    }
    #[getter]
    fn current_cycle(&self) -> usize {
        self.inner.current_cycle()
    }
    #[getter]
    fn pts_per_edge(&self) -> usize {
        self.inner.pts_per_edge()
    }
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.status())
    }
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.iteration()
    }
    #[getter]
    fn total_iterations(&self) -> Option<usize> {
        self.inner.total_iterations()
    }
    #[getter]
    fn is_finished(&self) -> bool {
        self.inner.status().is_finished()
    }

    fn __repr__(&self) -> String {
        format!(
            "FdebSolver(num_edges={}, point_dims={}, target='{}', status='{}', iteration={}/{:?}, dtype='{}')",
            self.num_edges(),
            self.point_dims(),
            self.target(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}
