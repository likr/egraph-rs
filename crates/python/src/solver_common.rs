//! Shared dual-precision (`f64`/`f32`) cooperative iterative solver helpers.

use egraph::core::{CoreResult, Layout2d, SolverStatus};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::ffi_convert::{ScalarDtype, core_error_to_py, ffi_error_to_py, solver_status_str};
use crate::layout_buffers::{Layout2dInner, PyLayout2d};

pub(crate) fn vec_f64_to_f32_checked(values: &[f64], ctx: &str) -> PyResult<Vec<f32>> {
    let mut out = Vec::with_capacity(values.len());
    for &v in values {
        if !v.is_finite() {
            return Err(PyValueError::new_err(format!("non-finite value in {ctx}")));
        }
        let vf = v as f32;
        if !vf.is_finite() {
            return Err(PyValueError::new_err(format!(
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

pub(crate) fn layout_f32_to_f64_checked(soa32: &Layout2d<f32>) -> CoreResult<Layout2d<f64>> {
    let x: Vec<f64> = soa32.x().iter().map(|&v| f64::from(v)).collect();
    let y: Vec<f64> = soa32.y().iter().map(|&v| f64::from(v)).collect();
    Layout2d::from_xy(x, y)
}

pub(crate) fn layout_f64_to_f32_checked(soa64: &Layout2d<f64>) -> CoreResult<Layout2d<f32>> {
    let x: Vec<f32> = soa64.x().iter().map(|&v| v as f32).collect();
    let y: Vec<f32> = soa64.y().iter().map(|&v| v as f32).collect();
    Layout2d::from_xy(x, y)
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
                return Err(PyValueError::new_err(format!(
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
                return Err(PyValueError::new_err(format!(
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
    fn solver_total_iterations(&self) -> usize;
}

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
                    Self::F64(s) => <$solver_ty<f64> as egraph::core::IterativeSolver<
                        f64,
                        egraph::core::Layout2d<f64>,
                    >>::status(s),
                    Self::F32(s) => <$solver_ty<f32> as egraph::core::IterativeSolver<
                        f32,
                        egraph::core::Layout2d<f32>,
                    >>::status(s),
                }
            }

            fn solver_iteration(&self) -> usize {
                match self {
                    Self::F64(s) => <$solver_ty<f64> as egraph::core::IterativeSolver<
                        f64,
                        egraph::core::Layout2d<f64>,
                    >>::iteration(s),
                    Self::F32(s) => <$solver_ty<f32> as egraph::core::IterativeSolver<
                        f32,
                        egraph::core::Layout2d<f32>,
                    >>::iteration(s),
                }
            }

            fn solver_total_iterations(&self) -> usize {
                match self {
                    Self::F64(s) => <$solver_ty<f64> as egraph::core::IterativeSolver<
                        f64,
                        egraph::core::Layout2d<f64>,
                    >>::total_iterations(s),
                    Self::F32(s) => <$solver_ty<f32> as egraph::core::IterativeSolver<
                        f32,
                        egraph::core::Layout2d<f32>,
                    >>::total_iterations(s),
                }
            }
        }
    };
}

pub(crate) use impl_dual_solver;

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
