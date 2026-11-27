//! Polymorphic Euclidean (`Layout2d | LayoutNd`) and Manifold (`ManifoldLayout2d`) solver dispatch.

use egraph::core::{CoreError, CoreResult, LayoutNd, ManifoldLayout2d, SolverStatus};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::ffi_convert::{ScalarDtype, core_error_to_py, solver_status_str};
use crate::layout_buffers::PyLayout2d;
use crate::layout_nd::{LayoutNdInner, PyLayoutNd};
use crate::manifold::{
    ExtractedManifold, PyManifoldLayout2d, extract_manifold_for_solver, write_back_manifold,
};
use crate::solver_common::{DualIterativeSolver, SolverAction, execute_solver_action};

pub(crate) enum SolverLayoutNd {
    F64(LayoutNd<f64>),
    F32(LayoutNd<f32>),
}

fn layout_nd_f32_to_f64_checked(nd32: &LayoutNd<f32>) -> CoreResult<LayoutNd<f64>> {
    let coords: Vec<f64> = nd32.coords().iter().map(|&v| f64::from(v)).collect();
    LayoutNd::from_flat(nd32.num_nodes(), nd32.dims(), coords)
}

fn layout_nd_f64_to_f32_checked(nd64: &LayoutNd<f64>) -> CoreResult<LayoutNd<f32>> {
    let mut coords = Vec::with_capacity(nd64.coords().len());
    for &v in nd64.coords() {
        if !v.is_finite() {
            return Err(CoreError::non_finite("LayoutNd::coords"));
        }
        let vf = v as f32;
        if !vf.is_finite() {
            return Err(CoreError::invalid_input(format!(
                "LayoutNd coordinate {v} overflows float32"
            )));
        }
        coords.push(vf);
    }
    LayoutNd::from_flat(nd64.num_nodes(), nd64.dims(), coords)
}

pub(crate) fn extract_solver_layout_nd(
    layout: &Bound<'_, PyLayoutNd>,
    target_dtype: ScalarDtype,
) -> PyResult<SolverLayoutNd> {
    let borrowed = layout.try_borrow()?;
    match (borrowed.inner(), target_dtype) {
        (LayoutNdInner::F64(nd), ScalarDtype::F64) => Ok(SolverLayoutNd::F64(nd.clone())),
        (LayoutNdInner::F32(nd), ScalarDtype::F32) => Ok(SolverLayoutNd::F32(nd.clone())),
        (LayoutNdInner::F32(nd), ScalarDtype::F64) => layout_nd_f32_to_f64_checked(nd)
            .map(SolverLayoutNd::F64)
            .map_err(core_error_to_py),
        (LayoutNdInner::F64(nd), ScalarDtype::F32) => layout_nd_f64_to_f32_checked(nd)
            .map(SolverLayoutNd::F32)
            .map_err(core_error_to_py),
    }
}

pub(crate) fn write_back_solver_layout_nd(
    layout: &Bound<'_, PyLayoutNd>,
    nd: &SolverLayoutNd,
) -> PyResult<()> {
    let mut borrowed = layout.try_borrow_mut()?;
    match (borrowed.inner_mut(), nd) {
        (LayoutNdInner::F64(dst), SolverLayoutNd::F64(src)) => {
            if dst.num_nodes() != src.num_nodes() || dst.dims() != src.dims() {
                return Err(PyValueError::new_err(format!(
                    "LayoutNd shape mismatch: expected ({}, {}), got ({}, {})",
                    src.num_nodes(),
                    src.dims(),
                    dst.num_nodes(),
                    dst.dims()
                )));
            }
            dst.coords_mut().copy_from_slice(src.coords());
            Ok(())
        }
        (LayoutNdInner::F32(dst), SolverLayoutNd::F32(src)) => {
            if dst.num_nodes() != src.num_nodes() || dst.dims() != src.dims() {
                return Err(PyValueError::new_err(format!(
                    "LayoutNd shape mismatch: expected ({}, {}), got ({}, {})",
                    src.num_nodes(),
                    src.dims(),
                    dst.num_nodes(),
                    dst.dims()
                )));
            }
            dst.coords_mut().copy_from_slice(src.coords());
            Ok(())
        }
        (LayoutNdInner::F32(dst), SolverLayoutNd::F64(src)) => {
            let converted = layout_nd_f64_to_f32_checked(src).map_err(core_error_to_py)?;
            if dst.num_nodes() != converted.num_nodes() || dst.dims() != converted.dims() {
                return Err(PyValueError::new_err("LayoutNd shape mismatch"));
            }
            dst.coords_mut().copy_from_slice(converted.coords());
            Ok(())
        }
        (LayoutNdInner::F64(dst), SolverLayoutNd::F32(src)) => {
            let converted = layout_nd_f32_to_f64_checked(src).map_err(core_error_to_py)?;
            if dst.num_nodes() != converted.num_nodes() || dst.dims() != converted.dims() {
                return Err(PyValueError::new_err("LayoutNd shape mismatch"));
            }
            dst.coords_mut().copy_from_slice(converted.coords());
            Ok(())
        }
    }
}

pub(crate) trait DualNdIterativeSolver: DualIterativeSolver {
    fn step_nd(&mut self, layout: &mut SolverLayoutNd) -> CoreResult<SolverStatus>;
    fn run_steps_nd(
        &mut self,
        layout: &mut SolverLayoutNd,
        steps: usize,
    ) -> CoreResult<SolverStatus>;
    fn run_nd(&mut self, layout: &mut SolverLayoutNd) -> CoreResult<SolverStatus>;
}

pub(crate) trait DualManifoldIterativeSolver: DualIterativeSolver {
    fn step_manifold_layout(
        &mut self,
        layout: &mut ManifoldLayout2d<f64>,
    ) -> CoreResult<SolverStatus>;
    fn step_manifold_layout_f32(
        &mut self,
        layout: &mut ManifoldLayout2d<f32>,
    ) -> CoreResult<SolverStatus>;
    fn run_steps_manifold_layout(
        &mut self,
        layout: &mut ManifoldLayout2d<f64>,
        steps: usize,
    ) -> CoreResult<SolverStatus>;
    fn run_steps_manifold_layout_f32(
        &mut self,
        layout: &mut ManifoldLayout2d<f32>,
        steps: usize,
    ) -> CoreResult<SolverStatus>;
    fn run_manifold_layout(
        &mut self,
        layout: &mut ManifoldLayout2d<f64>,
    ) -> CoreResult<SolverStatus>;
    fn run_manifold_layout_f32(
        &mut self,
        layout: &mut ManifoldLayout2d<f32>,
    ) -> CoreResult<SolverStatus>;
}

macro_rules! impl_dual_nd_solver {
    ($enum_name:ident, $solver_ty:ident) => {
        impl $crate::solvers::layout_dispatch::DualNdIterativeSolver for $enum_name {
            fn step_nd(
                &mut self,
                layout: &mut $crate::solvers::layout_dispatch::SolverLayoutNd,
            ) -> egraph::core::CoreResult<egraph::core::SolverStatus> {
                match (self, layout) {
                    (Self::F64(s), $crate::solvers::layout_dispatch::SolverLayoutNd::F64(l)) => {
                        egraph::core::IterativeSolver::step(s, l)
                    }
                    (Self::F32(s), $crate::solvers::layout_dispatch::SolverLayoutNd::F32(l)) => {
                        egraph::core::IterativeSolver::step(s, l)
                    }
                    _ => Err(egraph::core::CoreError::invalid_input(
                        "internal solver N-D layout dtype mismatch",
                    )),
                }
            }

            fn run_steps_nd(
                &mut self,
                layout: &mut $crate::solvers::layout_dispatch::SolverLayoutNd,
                steps: usize,
            ) -> egraph::core::CoreResult<egraph::core::SolverStatus> {
                match (self, layout) {
                    (Self::F64(s), $crate::solvers::layout_dispatch::SolverLayoutNd::F64(l)) => {
                        egraph::core::IterativeSolver::run_steps(s, l, steps)
                    }
                    (Self::F32(s), $crate::solvers::layout_dispatch::SolverLayoutNd::F32(l)) => {
                        egraph::core::IterativeSolver::run_steps(s, l, steps)
                    }
                    _ => Err(egraph::core::CoreError::invalid_input(
                        "internal solver N-D layout dtype mismatch",
                    )),
                }
            }

            fn run_nd(
                &mut self,
                layout: &mut $crate::solvers::layout_dispatch::SolverLayoutNd,
            ) -> egraph::core::CoreResult<egraph::core::SolverStatus> {
                match (self, layout) {
                    (Self::F64(s), $crate::solvers::layout_dispatch::SolverLayoutNd::F64(l)) => {
                        egraph::core::IterativeSolver::run(s, l)
                    }
                    (Self::F32(s), $crate::solvers::layout_dispatch::SolverLayoutNd::F32(l)) => {
                        egraph::core::IterativeSolver::run(s, l)
                    }
                    _ => Err(egraph::core::CoreError::invalid_input(
                        "internal solver N-D layout dtype mismatch",
                    )),
                }
            }
        }
    };
}

macro_rules! impl_dual_manifold_solver {
    ($enum_name:ident, $solver_ty:ident) => {
        impl $crate::solvers::layout_dispatch::DualManifoldIterativeSolver for $enum_name {
            fn step_manifold_layout(
                &mut self,
                layout: &mut egraph::core::ManifoldLayout2d<f64>,
            ) -> egraph::core::CoreResult<egraph::core::SolverStatus> {
                match self {
                    Self::F64(s) => egraph::core::IterativeSolver::step(s, layout),
                    Self::F32(_) => Err(egraph::core::CoreError::invalid_input(
                        "internal solver manifold dtype mismatch",
                    )),
                }
            }

            fn step_manifold_layout_f32(
                &mut self,
                layout: &mut egraph::core::ManifoldLayout2d<f32>,
            ) -> egraph::core::CoreResult<egraph::core::SolverStatus> {
                match self {
                    Self::F32(s) => egraph::core::IterativeSolver::step(s, layout),
                    Self::F64(_) => Err(egraph::core::CoreError::invalid_input(
                        "internal solver manifold dtype mismatch",
                    )),
                }
            }

            fn run_steps_manifold_layout(
                &mut self,
                layout: &mut egraph::core::ManifoldLayout2d<f64>,
                steps: usize,
            ) -> egraph::core::CoreResult<egraph::core::SolverStatus> {
                match self {
                    Self::F64(s) => egraph::core::IterativeSolver::run_steps(s, layout, steps),
                    Self::F32(_) => Err(egraph::core::CoreError::invalid_input(
                        "internal solver manifold dtype mismatch",
                    )),
                }
            }

            fn run_steps_manifold_layout_f32(
                &mut self,
                layout: &mut egraph::core::ManifoldLayout2d<f32>,
                steps: usize,
            ) -> egraph::core::CoreResult<egraph::core::SolverStatus> {
                match self {
                    Self::F32(s) => egraph::core::IterativeSolver::run_steps(s, layout, steps),
                    Self::F64(_) => Err(egraph::core::CoreError::invalid_input(
                        "internal solver manifold dtype mismatch",
                    )),
                }
            }

            fn run_manifold_layout(
                &mut self,
                layout: &mut egraph::core::ManifoldLayout2d<f64>,
            ) -> egraph::core::CoreResult<egraph::core::SolverStatus> {
                match self {
                    Self::F64(s) => egraph::core::IterativeSolver::run(s, layout),
                    Self::F32(_) => Err(egraph::core::CoreError::invalid_input(
                        "internal solver manifold dtype mismatch",
                    )),
                }
            }

            fn run_manifold_layout_f32(
                &mut self,
                layout: &mut egraph::core::ManifoldLayout2d<f32>,
            ) -> egraph::core::CoreResult<egraph::core::SolverStatus> {
                match self {
                    Self::F32(s) => egraph::core::IterativeSolver::run(s, layout),
                    Self::F64(_) => Err(egraph::core::CoreError::invalid_input(
                        "internal solver manifold dtype mismatch",
                    )),
                }
            }
        }
    };
}

pub(crate) use {impl_dual_manifold_solver, impl_dual_nd_solver};

fn execute_nd_action<I: DualNdIterativeSolver>(
    inner: &mut I,
    py: Python<'_>,
    layout: &Bound<'_, PyLayoutNd>,
    action: SolverAction,
) -> PyResult<&'static str> {
    let target_dtype = inner.dtype();
    let mut nd = extract_solver_layout_nd(layout, target_dtype)?;
    let status = py
        .detach(|| match action {
            SolverAction::Step => inner.step_nd(&mut nd),
            SolverAction::RunSteps(steps) => inner.run_steps_nd(&mut nd, steps),
            SolverAction::Run => inner.run_nd(&mut nd),
        })
        .map_err(core_error_to_py)?;
    write_back_solver_layout_nd(layout, &nd)?;
    Ok(solver_status_str(status))
}

pub(crate) fn execute_manifold_layout_action<I: DualManifoldIterativeSolver>(
    inner: &mut I,
    py: Python<'_>,
    layout: &Bound<'_, PyManifoldLayout2d>,
    action: SolverAction,
) -> PyResult<&'static str> {
    let target_dtype = inner.dtype();
    let mut extracted = extract_manifold_for_solver(layout, target_dtype)?;
    let status = py
        .detach(|| match &mut extracted {
            ExtractedManifold::F64(m) => match action {
                SolverAction::Step => inner.step_manifold_layout(m),
                SolverAction::RunSteps(steps) => inner.run_steps_manifold_layout(m, steps),
                SolverAction::Run => inner.run_manifold_layout(m),
            },
            ExtractedManifold::F32(m) => match action {
                SolverAction::Step => inner.step_manifold_layout_f32(m),
                SolverAction::RunSteps(steps) => inner.run_steps_manifold_layout_f32(m, steps),
                SolverAction::Run => inner.run_manifold_layout_f32(m),
            },
        })
        .map_err(core_error_to_py)?;
    write_back_manifold(layout, extracted)?;
    Ok(solver_status_str(status))
}

pub(crate) fn execute_euclid_action<I: DualNdIterativeSolver>(
    inner: &mut I,
    py: Python<'_>,
    layout: &Bound<'_, PyAny>,
    action: SolverAction,
) -> PyResult<&'static str> {
    if let Ok(l2) = layout.cast::<PyLayout2d>() {
        return execute_solver_action(inner, py, l2, action);
    }
    if let Ok(lnd) = layout.cast::<PyLayoutNd>() {
        return execute_nd_action(inner, py, lnd, action);
    }
    Err(PyTypeError::new_err(
        "layout must be a Layout2d or LayoutNd instance",
    ))
}

pub(crate) fn execute_any_layout_action<I: DualNdIterativeSolver + DualManifoldIterativeSolver>(
    inner: &mut I,
    py: Python<'_>,
    layout: &Bound<'_, PyAny>,
    action: SolverAction,
) -> PyResult<&'static str> {
    if let Ok(l2) = layout.cast::<PyLayout2d>() {
        return execute_solver_action(inner, py, l2, action);
    }
    if let Ok(lnd) = layout.cast::<PyLayoutNd>() {
        return execute_nd_action(inner, py, lnd, action);
    }
    if let Ok(lm) = layout.cast::<PyManifoldLayout2d>() {
        return execute_manifold_layout_action(inner, py, lm, action);
    }
    Err(PyTypeError::new_err(
        "layout must be a Layout2d, LayoutNd, or ManifoldLayout2d instance",
    ))
}
