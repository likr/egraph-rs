//! Shared dual-precision (`f64`/`f32`) cooperative iterative solver helpers across 2D, 3D, N-D, and manifolds.

use egraph::core::{CoreError, CoreResult, Layout2d, Layout3d, SolverStatus};
use pyo3::prelude::*;

use crate::errors::{
    core_error_to_py, dimension_mismatch_err, ffi_error_to_py, non_finite_value_err,
};
use crate::ffi_convert::{ScalarDtype, solver_status_str};
use crate::layout_buffers::{Layout2dInner, PyLayout2d};
use crate::layout3d::{Layout3dInner, PyLayout3d};

pub(crate) fn vec_f64_to_f32_checked(values: &[f64], ctx: &str) -> PyResult<Vec<f32>> {
    let mut out = Vec::with_capacity(values.len());
    for &v in values {
        if !v.is_finite() {
            return Err(non_finite_value_err(format!("non-finite value in {ctx}")));
        }
        let vf = v as f32;
        if !vf.is_finite() {
            return Err(non_finite_value_err(format!(
                "{ctx} entry {v} overflows float32"
            )));
        }
        out.push(vf);
    }
    Ok(out)
}

pub(crate) enum SolverLayout {
    F64(Layout2d<f64>),
    F32(Layout2d<f32>),
}

pub(crate) enum SolverLayout3d {
    F64(Layout3d<f64>),
    F32(Layout3d<f32>),
}

pub(crate) fn layout_f32_to_f64_checked(soa32: &Layout2d<f32>) -> CoreResult<Layout2d<f64>> {
    let x: Vec<f64> = soa32.x().iter().map(|&v| f64::from(v)).collect();
    let y: Vec<f64> = soa32.y().iter().map(|&v| f64::from(v)).collect();
    Layout2d::from_xy(x, y)
}

pub(crate) fn layout_f64_to_f32_checked(soa64: &Layout2d<f64>) -> CoreResult<Layout2d<f32>> {
    let mut x = Vec::with_capacity(soa64.len());
    let mut y = Vec::with_capacity(soa64.len());
    for (&px, &py) in soa64.x().iter().zip(soa64.y().iter()) {
        if !px.is_finite() || !py.is_finite() {
            return Err(CoreError::non_finite("Layout2d"));
        }
        let fx = px as f32;
        let fy = py as f32;
        if !fx.is_finite() || !fy.is_finite() {
            return Err(CoreError::non_finite("Layout2d float32 overflow"));
        }
        x.push(fx);
        y.push(fy);
    }
    Layout2d::from_xy(x, y)
}

pub(crate) fn layout3d_f32_to_f64_checked(soa32: &Layout3d<f32>) -> CoreResult<Layout3d<f64>> {
    let x: Vec<f64> = soa32.x().iter().map(|&v| f64::from(v)).collect();
    let y: Vec<f64> = soa32.y().iter().map(|&v| f64::from(v)).collect();
    let z: Vec<f64> = soa32.z().iter().map(|&v| f64::from(v)).collect();
    Layout3d::from_xyz(x, y, z)
}

pub(crate) fn layout3d_f64_to_f32_checked(soa64: &Layout3d<f64>) -> CoreResult<Layout3d<f32>> {
    let mut x = Vec::with_capacity(soa64.len());
    let mut y = Vec::with_capacity(soa64.len());
    let mut z = Vec::with_capacity(soa64.len());
    for ((&px, &py), &pz) in soa64.x().iter().zip(soa64.y().iter()).zip(soa64.z().iter()) {
        if !px.is_finite() || !py.is_finite() || !pz.is_finite() {
            return Err(CoreError::non_finite("Layout3d"));
        }
        let (fx, fy, fz) = (px as f32, py as f32, pz as f32);
        if !fx.is_finite() || !fy.is_finite() || !fz.is_finite() {
            return Err(CoreError::non_finite("Layout3d float32 overflow"));
        }
        x.push(fx);
        y.push(fy);
        z.push(fz);
    }
    Layout3d::from_xyz(x, y, z)
}

pub(crate) fn extract_solver_layout(
    layout: &Bound<'_, PyLayout2d>,
    target_dtype: ScalarDtype,
) -> PyResult<SolverLayout> {
    let borrowed = layout.try_borrow()?;
    match (borrowed.inner(), target_dtype) {
        (Layout2dInner::F64(buf), ScalarDtype::F64) => Ok(SolverLayout::F64(buf.to_layout2d())),
        (Layout2dInner::F32(buf), ScalarDtype::F32) => Ok(SolverLayout::F32(buf.to_layout2d())),
        (Layout2dInner::F32(buf), ScalarDtype::F64) => {
            layout_f32_to_f64_checked(&buf.to_layout2d())
                .map(SolverLayout::F64)
                .map_err(core_error_to_py)
        }
        (Layout2dInner::F64(buf), ScalarDtype::F32) => {
            layout_f64_to_f32_checked(&buf.to_layout2d())
                .map(SolverLayout::F32)
                .map_err(core_error_to_py)
        }
    }
}

pub(crate) fn write_back_solver_layout(
    layout: &Bound<'_, PyLayout2d>,
    soa: &SolverLayout,
) -> PyResult<()> {
    let mut borrowed = layout.try_borrow_mut()?;
    match (borrowed.inner_mut(), soa) {
        (Layout2dInner::F64(buf), SolverLayout::F64(l64)) => {
            buf.copy_from_layout2d(l64).map_err(ffi_error_to_py)
        }
        (Layout2dInner::F32(buf), SolverLayout::F32(l32)) => {
            buf.copy_from_layout2d(l32).map_err(ffi_error_to_py)
        }
        (Layout2dInner::F32(buf), SolverLayout::F64(l64)) => {
            let n = l64.len();
            if buf.len() != n {
                return Err(dimension_mismatch_err(format!(
                    "layout length mismatch: expected {n}, got {}",
                    buf.len()
                )));
            }
            for (i, [px, py_coord]) in buf
                .as_mut_slice()
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .enumerate()
            {
                *px = l64.x()[i] as f32;
                *py_coord = l64.y()[i] as f32;
            }
            Ok(())
        }
        (Layout2dInner::F64(buf), SolverLayout::F32(l32)) => {
            let n = l32.len();
            if buf.len() != n {
                return Err(dimension_mismatch_err(format!(
                    "layout length mismatch: expected {n}, got {}",
                    buf.len()
                )));
            }
            for (i, [px, py_coord]) in buf
                .as_mut_slice()
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .enumerate()
            {
                *px = f64::from(l32.x()[i]);
                *py_coord = f64::from(l32.y()[i]);
            }
            Ok(())
        }
    }
}

pub(crate) fn extract_solver_layout3d(
    layout: &Bound<'_, PyLayout3d>,
    target_dtype: ScalarDtype,
) -> PyResult<SolverLayout3d> {
    let borrowed = layout.try_borrow()?;
    match (borrowed.inner(), target_dtype) {
        (Layout3dInner::F64(buf), ScalarDtype::F64) => Ok(SolverLayout3d::F64(buf.to_layout3d())),
        (Layout3dInner::F32(buf), ScalarDtype::F32) => Ok(SolverLayout3d::F32(buf.to_layout3d())),
        (Layout3dInner::F32(buf), ScalarDtype::F64) => {
            layout3d_f32_to_f64_checked(&buf.to_layout3d())
                .map(SolverLayout3d::F64)
                .map_err(core_error_to_py)
        }
        (Layout3dInner::F64(buf), ScalarDtype::F32) => {
            layout3d_f64_to_f32_checked(&buf.to_layout3d())
                .map(SolverLayout3d::F32)
                .map_err(core_error_to_py)
        }
    }
}

pub(crate) fn write_back_solver_layout3d(
    layout: &Bound<'_, PyLayout3d>,
    soa: &SolverLayout3d,
) -> PyResult<()> {
    let mut borrowed = layout.try_borrow_mut()?;
    match (borrowed.inner_mut(), soa) {
        (Layout3dInner::F64(buf), SolverLayout3d::F64(l64)) => {
            buf.copy_from_layout3d(l64).map_err(ffi_error_to_py)
        }
        (Layout3dInner::F32(buf), SolverLayout3d::F32(l32)) => {
            buf.copy_from_layout3d(l32).map_err(ffi_error_to_py)
        }
        (Layout3dInner::F32(buf), SolverLayout3d::F64(l64)) => {
            let n = l64.len();
            if buf.len() != n {
                return Err(dimension_mismatch_err(format!(
                    "layout length mismatch: expected {n}, got {}",
                    buf.len()
                )));
            }
            for (i, [px, py_coord, pz]) in buf
                .as_mut_slice()
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .enumerate()
            {
                *px = l64.x()[i] as f32;
                *py_coord = l64.y()[i] as f32;
                *pz = l64.z()[i] as f32;
            }
            Ok(())
        }
        (Layout3dInner::F64(buf), SolverLayout3d::F32(l32)) => {
            let n = l32.len();
            if buf.len() != n {
                return Err(dimension_mismatch_err(format!(
                    "layout length mismatch: expected {n}, got {}",
                    buf.len()
                )));
            }
            for (i, [px, py_coord, pz]) in buf
                .as_mut_slice()
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .enumerate()
            {
                *px = f64::from(l32.x()[i]);
                *py_coord = f64::from(l32.y()[i]);
                *pz = f64::from(l32.z()[i]);
            }
            Ok(())
        }
    }
}

pub(crate) trait DualIterativeSolver: Send {
    fn dtype(&self) -> ScalarDtype;
    fn step_layout(&mut self, layout: &mut SolverLayout) -> CoreResult<SolverStatus>;
    fn run_steps_layout(
        &mut self,
        layout: &mut SolverLayout,
        steps: usize,
    ) -> CoreResult<SolverStatus>;
    fn run_layout(&mut self, layout: &mut SolverLayout) -> CoreResult<SolverStatus>;
    fn solver_status(&self) -> SolverStatus;
    fn solver_iteration(&self) -> usize;
    fn solver_total_iterations(&self) -> Option<usize>;
}

/// Generate a dual-precision (`f64`/`f32`) `Layout2d` solver wrapper implementing `DualIterativeSolver`.
#[macro_export]
macro_rules! impl_dual_solver {
    ($enum_name:ident, $solver_ty:ident) => {
        pub(crate) enum $enum_name {
            F64($solver_ty<f64>),
            F32($solver_ty<f32>),
        }

        impl $crate::solver_common::DualIterativeSolver for $enum_name {
            fn dtype(&self) -> $crate::ffi_convert::ScalarDtype {
                match self {
                    Self::F64(_) => $crate::ffi_convert::ScalarDtype::F64,
                    Self::F32(_) => $crate::ffi_convert::ScalarDtype::F32,
                }
            }

            fn step_layout(
                &mut self,
                layout: &mut $crate::solver_common::SolverLayout,
            ) -> egraph::core::CoreResult<egraph::core::SolverStatus> {
                match (self, layout) {
                    (Self::F64(s), $crate::solver_common::SolverLayout::F64(l)) => {
                        egraph::core::IterativeSolver::step(s, l)
                    }
                    (Self::F32(s), $crate::solver_common::SolverLayout::F32(l)) => {
                        egraph::core::IterativeSolver::step(s, l)
                    }
                    _ => Err(egraph::core::CoreError::invalid_input(
                        "internal solver layout dtype mismatch",
                    )),
                }
            }

            fn run_steps_layout(
                &mut self,
                layout: &mut $crate::solver_common::SolverLayout,
                steps: usize,
            ) -> egraph::core::CoreResult<egraph::core::SolverStatus> {
                match (self, layout) {
                    (Self::F64(s), $crate::solver_common::SolverLayout::F64(l)) => {
                        egraph::core::IterativeSolver::run_steps(s, l, steps)
                    }
                    (Self::F32(s), $crate::solver_common::SolverLayout::F32(l)) => {
                        egraph::core::IterativeSolver::run_steps(s, l, steps)
                    }
                    _ => Err(egraph::core::CoreError::invalid_input(
                        "internal solver layout dtype mismatch",
                    )),
                }
            }

            fn run_layout(
                &mut self,
                layout: &mut $crate::solver_common::SolverLayout,
            ) -> egraph::core::CoreResult<egraph::core::SolverStatus> {
                match (self, layout) {
                    (Self::F64(s), $crate::solver_common::SolverLayout::F64(l)) => {
                        egraph::core::IterativeSolver::run(s, l)
                    }
                    (Self::F32(s), $crate::solver_common::SolverLayout::F32(l)) => {
                        egraph::core::IterativeSolver::run(s, l)
                    }
                    _ => Err(egraph::core::CoreError::invalid_input(
                        "internal solver layout dtype mismatch",
                    )),
                }
            }

            fn solver_status(&self) -> egraph::core::SolverStatus {
                match self {
                    Self::F64(s) => egraph::core::IterativeSolver::status(s),
                    Self::F32(s) => egraph::core::IterativeSolver::status(s),
                }
            }

            fn solver_iteration(&self) -> usize {
                match self {
                    Self::F64(s) => egraph::core::IterativeSolver::iteration(s),
                    Self::F32(s) => egraph::core::IterativeSolver::iteration(s),
                }
            }

            fn solver_total_iterations(&self) -> Option<usize> {
                match self {
                    Self::F64(s) => egraph::core::IterativeSolver::total_iterations(s),
                    Self::F32(s) => egraph::core::IterativeSolver::total_iterations(s),
                }
            }
        }
    };
}

pub(crate) use crate::impl_dual_solver;

pub(crate) enum SolverAction {
    Step,
    RunSteps(usize),
    Run,
}

pub(crate) fn execute_solver_action<I: DualIterativeSolver>(
    inner: &mut I,
    py: Python<'_>,
    layout: &Bound<'_, PyLayout2d>,
    action: SolverAction,
) -> PyResult<&'static str> {
    let target_dtype = inner.dtype();
    let mut soa = extract_solver_layout(layout, target_dtype)?;
    let status = py
        .detach(|| match action {
            SolverAction::Step => inner.step_layout(&mut soa),
            SolverAction::RunSteps(steps) => inner.run_steps_layout(&mut soa, steps),
            SolverAction::Run => inner.run_layout(&mut soa),
        })
        .map_err(core_error_to_py)?;
    write_back_solver_layout(layout, &soa)?;
    Ok(solver_status_str(status))
}

#[inline]
pub(crate) fn format_optional_usize(val: Option<usize>) -> String {
    match val {
        Some(v) => v.to_string(),
        None => "None".to_string(),
    }
}
