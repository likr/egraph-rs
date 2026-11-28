//! Policy 1A target-space dispatch (`layout2d | layout3d | layout_nd | manifold2d`) and `ManifoldKind` locking.

use egraph::core::{CoreResult, ManifoldKind, SolverStatus};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

pub(crate) use super::target_layout::{
    TargetLayout, extract_target_layout, write_back_target_layout,
};
use crate::errors::{core_error_to_py, unsupported_operation_err};
use crate::ffi_convert::{ScalarDtype, normalize_enum_key, solver_status_str};
use crate::manifold::manifold_kind_str;
pub(crate) use crate::solver_common::SolverAction;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SolverTargetKind {
    Layout2d,
    Layout3d,
    LayoutNd,
    Manifold2d,
}

impl SolverTargetKind {
    pub(crate) fn from_str(raw: &str) -> PyResult<Self> {
        match normalize_enum_key(raw).as_str() {
            "layout2d" => Ok(Self::Layout2d),
            "layout3d" => Ok(Self::Layout3d),
            "layoutnd" => Ok(Self::LayoutNd),
            "manifold2d" => Ok(Self::Manifold2d),
            _ => Err(unsupported_operation_err(format!(
                "unsupported target {raw:?}; expected 'layout2d', 'layout3d', 'layout_nd', or 'manifold2d'"
            ))),
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Layout2d => "layout2d",
            Self::Layout3d => "layout3d",
            Self::LayoutNd => "layout_nd",
            Self::Manifold2d => "manifold2d",
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for SolverTargetKind {
    type Error = PyErr;

    fn extract(ob: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let raw = ob.extract::<String>().map_err(|_| {
            PyTypeError::new_err(
                "target must be 'layout2d', 'layout3d', 'layout_nd', or 'manifold2d'",
            )
        })?;
        Self::from_str(&raw)
    }
}

pub(crate) trait DualTargetSolver: Send + Sync {
    fn target_kind(&self) -> SolverTargetKind;
    fn dtype(&self) -> ScalarDtype;
    fn solver_status(&self) -> SolverStatus;
    fn solver_iteration(&self) -> usize;
    fn solver_total_iterations(&self) -> Option<usize>;
    fn validate_raw(&self, layout: &TargetLayout) -> CoreResult<()>;
    fn step_raw(&mut self, layout: &mut TargetLayout) -> CoreResult<SolverStatus>;
    fn run_steps_raw(
        &mut self,
        layout: &mut TargetLayout,
        steps: usize,
    ) -> CoreResult<SolverStatus>;
    fn run_raw(&mut self, layout: &mut TargetLayout) -> CoreResult<SolverStatus>;
}

pub(crate) fn validate_target_space<S: DualTargetSolver>(
    inner: &S,
    locked_manifold: &mut Option<ManifoldKind>,
    py: Python<'_>,
    layout: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let extracted = extract_target_layout(layout, inner.target_kind(), inner.dtype())?;
    if let (Some(prev), Some(curr)) = (*locked_manifold, extracted.manifold_kind())
        && prev != curr
    {
        return Err(unsupported_operation_err(format!(
            "manifold solver is locked to '{}', got '{}'",
            manifold_kind_str(prev),
            manifold_kind_str(curr)
        )));
    }
    py.detach(|| inner.validate_raw(&extracted))
        .map_err(core_error_to_py)?;
    if locked_manifold.is_none()
        && let Some(curr) = extracted.manifold_kind()
    {
        *locked_manifold = Some(curr);
    }
    Ok(())
}

pub(crate) fn execute_target_action<S: DualTargetSolver>(
    inner: &mut S,
    locked_manifold: &mut Option<ManifoldKind>,
    py: Python<'_>,
    layout: &Bound<'_, PyAny>,
    action: SolverAction,
) -> PyResult<&'static str> {
    let mut extracted = extract_target_layout(layout, inner.target_kind(), inner.dtype())?;
    if let (Some(prev), Some(curr)) = (*locked_manifold, extracted.manifold_kind())
        && prev != curr
    {
        return Err(unsupported_operation_err(format!(
            "manifold solver is locked to '{}', got '{}'",
            manifold_kind_str(prev),
            manifold_kind_str(curr)
        )));
    }
    let kind_opt = extracted.manifold_kind();
    let status = py
        .detach(|| match action {
            SolverAction::Step => inner.step_raw(&mut extracted),
            SolverAction::RunSteps(steps) => inner.run_steps_raw(&mut extracted, steps),
            SolverAction::Run => inner.run_raw(&mut extracted),
        })
        .map_err(core_error_to_py)?;
    if locked_manifold.is_none()
        && let Some(curr) = kind_opt
    {
        *locked_manifold = Some(curr);
    }
    write_back_target_layout(layout, extracted)?;
    Ok(solver_status_str(status))
}

macro_rules! impl_riemannian_target_solver {
    ($enum_name:ident, $solver_ty:ident) => {
        pub(crate) enum $enum_name {
            L2F64($solver_ty<egraph::core::Layout2d<f64>>),
            L2F32($solver_ty<egraph::core::Layout2d<f32>>),
            L3F64($solver_ty<egraph::core::Layout3d<f64>>),
            L3F32($solver_ty<egraph::core::Layout3d<f32>>),
            NdF64($solver_ty<egraph::core::LayoutNd<f64>>),
            NdF32($solver_ty<egraph::core::LayoutNd<f32>>),
            M2F64($solver_ty<egraph::core::ManifoldLayout2d<f64>>),
            M2F32($solver_ty<egraph::core::ManifoldLayout2d<f32>>),
        }

        impl $crate::solvers::layout_dispatch::DualTargetSolver for $enum_name {
            fn target_kind(&self) -> $crate::solvers::layout_dispatch::SolverTargetKind {
                use $crate::solvers::layout_dispatch::SolverTargetKind;
                match self {
                    Self::L2F64(_) | Self::L2F32(_) => SolverTargetKind::Layout2d,
                    Self::L3F64(_) | Self::L3F32(_) => SolverTargetKind::Layout3d,
                    Self::NdF64(_) | Self::NdF32(_) => SolverTargetKind::LayoutNd,
                    Self::M2F64(_) | Self::M2F32(_) => SolverTargetKind::Manifold2d,
                }
            }

            fn dtype(&self) -> $crate::ffi_convert::ScalarDtype {
                use $crate::ffi_convert::ScalarDtype;
                match self {
                    Self::L2F64(_) | Self::L3F64(_) | Self::NdF64(_) | Self::M2F64(_) => {
                        ScalarDtype::F64
                    }
                    Self::L2F32(_) | Self::L3F32(_) | Self::NdF32(_) | Self::M2F32(_) => {
                        ScalarDtype::F32
                    }
                }
            }

            fn solver_status(&self) -> egraph::core::SolverStatus {
                use egraph::core::IterativeSolver;
                match self {
                    Self::L2F64(s) => IterativeSolver::status(s),
                    Self::L2F32(s) => IterativeSolver::status(s),
                    Self::L3F64(s) => IterativeSolver::status(s),
                    Self::L3F32(s) => IterativeSolver::status(s),
                    Self::NdF64(s) => IterativeSolver::status(s),
                    Self::NdF32(s) => IterativeSolver::status(s),
                    Self::M2F64(s) => IterativeSolver::status(s),
                    Self::M2F32(s) => IterativeSolver::status(s),
                }
            }

            fn solver_iteration(&self) -> usize {
                use egraph::core::IterativeSolver;
                match self {
                    Self::L2F64(s) => IterativeSolver::iteration(s),
                    Self::L2F32(s) => IterativeSolver::iteration(s),
                    Self::L3F64(s) => IterativeSolver::iteration(s),
                    Self::L3F32(s) => IterativeSolver::iteration(s),
                    Self::NdF64(s) => IterativeSolver::iteration(s),
                    Self::NdF32(s) => IterativeSolver::iteration(s),
                    Self::M2F64(s) => IterativeSolver::iteration(s),
                    Self::M2F32(s) => IterativeSolver::iteration(s),
                }
            }

            fn solver_total_iterations(&self) -> Option<usize> {
                use egraph::core::IterativeSolver;
                match self {
                    Self::L2F64(s) => IterativeSolver::total_iterations(s),
                    Self::L2F32(s) => IterativeSolver::total_iterations(s),
                    Self::L3F64(s) => IterativeSolver::total_iterations(s),
                    Self::L3F32(s) => IterativeSolver::total_iterations(s),
                    Self::NdF64(s) => IterativeSolver::total_iterations(s),
                    Self::NdF32(s) => IterativeSolver::total_iterations(s),
                    Self::M2F64(s) => IterativeSolver::total_iterations(s),
                    Self::M2F32(s) => IterativeSolver::total_iterations(s),
                }
            }

            fn validate_raw(
                &self,
                layout: &$crate::solvers::layout_dispatch::TargetLayout,
            ) -> egraph::core::CoreResult<()> {
                use egraph::core::IterativeSolver;
                use $crate::solvers::layout_dispatch::TargetLayout;
                match (self, layout) {
                    (Self::L2F64(s), TargetLayout::L2F64(l)) => {
                        IterativeSolver::validate_target(s, l)
                    }
                    (Self::L2F32(s), TargetLayout::L2F32(l)) => {
                        IterativeSolver::validate_target(s, l)
                    }
                    (Self::L3F64(s), TargetLayout::L3F64(l)) => {
                        IterativeSolver::validate_target(s, l)
                    }
                    (Self::L3F32(s), TargetLayout::L3F32(l)) => {
                        IterativeSolver::validate_target(s, l)
                    }
                    (Self::NdF64(s), TargetLayout::NdF64(l)) => {
                        IterativeSolver::validate_target(s, l)
                    }
                    (Self::NdF32(s), TargetLayout::NdF32(l)) => {
                        IterativeSolver::validate_target(s, l)
                    }
                    (Self::M2F64(s), TargetLayout::M2F64(l)) => {
                        IterativeSolver::validate_target(s, l)
                    }
                    (Self::M2F32(s), TargetLayout::M2F32(l)) => {
                        IterativeSolver::validate_target(s, l)
                    }
                    _ => Err(egraph::core::CoreError::invalid_input(
                        "internal solver target/dtype mismatch",
                    )),
                }
            }

            fn step_raw(
                &mut self,
                layout: &mut $crate::solvers::layout_dispatch::TargetLayout,
            ) -> egraph::core::CoreResult<egraph::core::SolverStatus> {
                use egraph::core::IterativeSolver;
                use $crate::solvers::layout_dispatch::TargetLayout;
                match (self, layout) {
                    (Self::L2F64(s), TargetLayout::L2F64(l)) => IterativeSolver::step(s, l),
                    (Self::L2F32(s), TargetLayout::L2F32(l)) => IterativeSolver::step(s, l),
                    (Self::L3F64(s), TargetLayout::L3F64(l)) => IterativeSolver::step(s, l),
                    (Self::L3F32(s), TargetLayout::L3F32(l)) => IterativeSolver::step(s, l),
                    (Self::NdF64(s), TargetLayout::NdF64(l)) => IterativeSolver::step(s, l),
                    (Self::NdF32(s), TargetLayout::NdF32(l)) => IterativeSolver::step(s, l),
                    (Self::M2F64(s), TargetLayout::M2F64(l)) => IterativeSolver::step(s, l),
                    (Self::M2F32(s), TargetLayout::M2F32(l)) => IterativeSolver::step(s, l),
                    _ => Err(egraph::core::CoreError::invalid_input(
                        "internal solver target/dtype mismatch",
                    )),
                }
            }

            fn run_steps_raw(
                &mut self,
                layout: &mut $crate::solvers::layout_dispatch::TargetLayout,
                steps: usize,
            ) -> egraph::core::CoreResult<egraph::core::SolverStatus> {
                use egraph::core::IterativeSolver;
                use $crate::solvers::layout_dispatch::TargetLayout;
                match (self, layout) {
                    (Self::L2F64(s), TargetLayout::L2F64(l)) => {
                        IterativeSolver::run_steps(s, l, steps)
                    }
                    (Self::L2F32(s), TargetLayout::L2F32(l)) => {
                        IterativeSolver::run_steps(s, l, steps)
                    }
                    (Self::L3F64(s), TargetLayout::L3F64(l)) => {
                        IterativeSolver::run_steps(s, l, steps)
                    }
                    (Self::L3F32(s), TargetLayout::L3F32(l)) => {
                        IterativeSolver::run_steps(s, l, steps)
                    }
                    (Self::NdF64(s), TargetLayout::NdF64(l)) => {
                        IterativeSolver::run_steps(s, l, steps)
                    }
                    (Self::NdF32(s), TargetLayout::NdF32(l)) => {
                        IterativeSolver::run_steps(s, l, steps)
                    }
                    (Self::M2F64(s), TargetLayout::M2F64(l)) => {
                        IterativeSolver::run_steps(s, l, steps)
                    }
                    (Self::M2F32(s), TargetLayout::M2F32(l)) => {
                        IterativeSolver::run_steps(s, l, steps)
                    }
                    _ => Err(egraph::core::CoreError::invalid_input(
                        "internal solver target/dtype mismatch",
                    )),
                }
            }

            fn run_raw(
                &mut self,
                layout: &mut $crate::solvers::layout_dispatch::TargetLayout,
            ) -> egraph::core::CoreResult<egraph::core::SolverStatus> {
                use egraph::core::IterativeSolver;
                use $crate::solvers::layout_dispatch::TargetLayout;
                match (self, layout) {
                    (Self::L2F64(s), TargetLayout::L2F64(l)) => IterativeSolver::run(s, l),
                    (Self::L2F32(s), TargetLayout::L2F32(l)) => IterativeSolver::run(s, l),
                    (Self::L3F64(s), TargetLayout::L3F64(l)) => IterativeSolver::run(s, l),
                    (Self::L3F32(s), TargetLayout::L3F32(l)) => IterativeSolver::run(s, l),
                    (Self::NdF64(s), TargetLayout::NdF64(l)) => IterativeSolver::run(s, l),
                    (Self::NdF32(s), TargetLayout::NdF32(l)) => IterativeSolver::run(s, l),
                    (Self::M2F64(s), TargetLayout::M2F64(l)) => IterativeSolver::run(s, l),
                    (Self::M2F32(s), TargetLayout::M2F32(l)) => IterativeSolver::run(s, l),
                    _ => Err(egraph::core::CoreError::invalid_input(
                        "internal solver target/dtype mismatch",
                    )),
                }
            }
        }
    };
}

pub(crate) use impl_riemannian_target_solver;

macro_rules! dispatch_riemannian_target {
    (@f64 $enum_name:ident, $target:expr, $solver_ty:ident :: $method:ident ( $($arg:expr),* $(,)? )) => {{
        match $target {
            $crate::solvers::layout_dispatch::SolverTargetKind::Layout2d => {
                $solver_ty::<egraph::core::Layout2d<f64>>::$method($($arg),*).map($enum_name::L2F64)
            }
            $crate::solvers::layout_dispatch::SolverTargetKind::Layout3d => {
                $solver_ty::<egraph::core::Layout3d<f64>>::$method($($arg),*).map($enum_name::L3F64)
            }
            $crate::solvers::layout_dispatch::SolverTargetKind::LayoutNd => {
                $solver_ty::<egraph::core::LayoutNd<f64>>::$method($($arg),*).map($enum_name::NdF64)
            }
            $crate::solvers::layout_dispatch::SolverTargetKind::Manifold2d => {
                $solver_ty::<egraph::core::ManifoldLayout2d<f64>>::$method($($arg),*).map($enum_name::M2F64)
            }
        }
    }};
    (@f32 $enum_name:ident, $target:expr, $solver_ty:ident :: $method:ident ( $($arg:expr),* $(,)? )) => {{
        match $target {
            $crate::solvers::layout_dispatch::SolverTargetKind::Layout2d => {
                $solver_ty::<egraph::core::Layout2d<f32>>::$method($($arg),*).map($enum_name::L2F32)
            }
            $crate::solvers::layout_dispatch::SolverTargetKind::Layout3d => {
                $solver_ty::<egraph::core::Layout3d<f32>>::$method($($arg),*).map($enum_name::L3F32)
            }
            $crate::solvers::layout_dispatch::SolverTargetKind::LayoutNd => {
                $solver_ty::<egraph::core::LayoutNd<f32>>::$method($($arg),*).map($enum_name::NdF32)
            }
            $crate::solvers::layout_dispatch::SolverTargetKind::Manifold2d => {
                $solver_ty::<egraph::core::ManifoldLayout2d<f32>>::$method($($arg),*).map($enum_name::M2F32)
            }
        }
    }};
    ($enum_name:ident, $target:expr, $dtype:expr, $solver_ty:ident :: $method:ident ( $($arg:expr),* $(,)? )) => {{
        match $dtype {
            $crate::ffi_convert::ScalarDtype::F64 => {
                dispatch_riemannian_target!(@f64 $enum_name, $target, $solver_ty::$method($($arg),*))
            }
            $crate::ffi_convert::ScalarDtype::F32 => {
                dispatch_riemannian_target!(@f32 $enum_name, $target, $solver_ty::$method($($arg),*))
            }
        }
    }};
}

pub(crate) use dispatch_riemannian_target;

macro_rules! match_riemannian_target {
    ($enum_name:ident, $val:expr, $s:ident => $body:expr) => {
        match $val {
            $enum_name::L2F64($s) => $body,
            $enum_name::L2F32($s) => $body,
            $enum_name::L3F64($s) => $body,
            $enum_name::L3F32($s) => $body,
            $enum_name::NdF64($s) => $body,
            $enum_name::NdF32($s) => $body,
            $enum_name::M2F64($s) => $body,
            $enum_name::M2F32($s) => $body,
        }
    };
}

pub(crate) use match_riemannian_target;
