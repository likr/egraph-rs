//! Extracted target layout containers (`TargetLayout`, `SolverLayoutNd`) and FFI marshaling helpers.

use egraph::core::{
    CoreError, CoreResult, Layout2d, Layout3d, LayoutNd, ManifoldKind, ManifoldLayout2d,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use super::layout_dispatch::SolverTargetKind;
use crate::errors::{core_error_to_py, dimension_mismatch_err};
use crate::ffi_convert::ScalarDtype;
use crate::layout_buffers::PyLayout2d;
use crate::layout_nd::{LayoutNdInner, PyLayoutNd};
use crate::layout3d::PyLayout3d;
use crate::manifold::{
    ExtractedManifold, PyManifoldLayout2d, extract_manifold_for_solver, write_back_manifold,
};
use crate::solver_common::{
    SolverLayout, SolverLayout3d, extract_solver_layout, extract_solver_layout3d,
    write_back_solver_layout, write_back_solver_layout3d,
};

pub(crate) enum SolverLayoutNd {
    F64(LayoutNd<f64>),
    F32(LayoutNd<f32>),
}

fn layout_nd_f32_to_f64_checked(nd32: &LayoutNd<f32>) -> CoreResult<LayoutNd<f64>> {
    let coords: Vec<f64> = nd32.coords().iter().map(|&v| f64::from(v)).collect();
    LayoutNd::from_flat_vec(nd32.num_nodes(), nd32.dims(), coords)
}

fn layout_nd_f64_to_f32_checked(nd64: &LayoutNd<f64>) -> CoreResult<LayoutNd<f32>> {
    let mut coords = Vec::with_capacity(nd64.coords().len());
    for &v in nd64.coords() {
        if !v.is_finite() {
            return Err(CoreError::non_finite("LayoutNd::coords"));
        }
        let vf = v as f32;
        if !vf.is_finite() {
            return Err(CoreError::non_finite("LayoutNd::coords"));
        }
        coords.push(vf);
    }
    LayoutNd::from_flat_vec(nd64.num_nodes(), nd64.dims(), coords)
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
                return Err(dimension_mismatch_err(format!(
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
                return Err(dimension_mismatch_err(format!(
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
                return Err(dimension_mismatch_err("LayoutNd shape mismatch"));
            }
            dst.coords_mut().copy_from_slice(converted.coords());
            Ok(())
        }
        (LayoutNdInner::F64(dst), SolverLayoutNd::F32(src)) => {
            let converted = layout_nd_f32_to_f64_checked(src).map_err(core_error_to_py)?;
            if dst.num_nodes() != converted.num_nodes() || dst.dims() != converted.dims() {
                return Err(dimension_mismatch_err("LayoutNd shape mismatch"));
            }
            dst.coords_mut().copy_from_slice(converted.coords());
            Ok(())
        }
    }
}

pub(crate) enum TargetLayout {
    L2F64(Layout2d<f64>),
    L2F32(Layout2d<f32>),
    L3F64(Layout3d<f64>),
    L3F32(Layout3d<f32>),
    NdF64(LayoutNd<f64>),
    NdF32(LayoutNd<f32>),
    M2F64(ManifoldLayout2d<f64>),
    M2F32(ManifoldLayout2d<f32>),
}

impl TargetLayout {
    pub(crate) fn manifold_kind(&self) -> Option<ManifoldKind> {
        match self {
            Self::M2F64(m) => Some(m.kind()),
            Self::M2F32(m) => Some(m.kind()),
            _ => None,
        }
    }
}

fn layout_actual_target(layout: &Bound<'_, PyAny>) -> Option<SolverTargetKind> {
    if layout.is_instance_of::<PyLayout2d>() {
        Some(SolverTargetKind::Layout2d)
    } else if layout.is_instance_of::<PyLayout3d>() {
        Some(SolverTargetKind::Layout3d)
    } else if layout.is_instance_of::<PyLayoutNd>() {
        Some(SolverTargetKind::LayoutNd)
    } else if layout.is_instance_of::<PyManifoldLayout2d>() {
        Some(SolverTargetKind::Manifold2d)
    } else {
        None
    }
}

pub(crate) fn extract_target_layout(
    layout: &Bound<'_, PyAny>,
    expected_target: SolverTargetKind,
    target_dtype: ScalarDtype,
) -> PyResult<TargetLayout> {
    let Some(actual) = layout_actual_target(layout) else {
        return Err(PyTypeError::new_err(
            "layout must be a Layout2d, Layout3d, LayoutNd, or ManifoldLayout2d instance",
        ));
    };
    if actual != expected_target {
        return Err(dimension_mismatch_err(format!(
            "solver was constructed for target='{}', got '{}' layout",
            expected_target.as_str(),
            actual.as_str()
        )));
    }
    match expected_target {
        SolverTargetKind::Layout2d => {
            let l2 = layout.cast::<PyLayout2d>()?;
            match extract_solver_layout(l2, target_dtype)? {
                SolverLayout::F64(l) => Ok(TargetLayout::L2F64(l)),
                SolverLayout::F32(l) => Ok(TargetLayout::L2F32(l)),
            }
        }
        SolverTargetKind::Layout3d => {
            let l3 = layout.cast::<PyLayout3d>()?;
            match extract_solver_layout3d(l3, target_dtype)? {
                SolverLayout3d::F64(l) => Ok(TargetLayout::L3F64(l)),
                SolverLayout3d::F32(l) => Ok(TargetLayout::L3F32(l)),
            }
        }
        SolverTargetKind::LayoutNd => {
            let lnd = layout.cast::<PyLayoutNd>()?;
            match extract_solver_layout_nd(lnd, target_dtype)? {
                SolverLayoutNd::F64(l) => Ok(TargetLayout::NdF64(l)),
                SolverLayoutNd::F32(l) => Ok(TargetLayout::NdF32(l)),
            }
        }
        SolverTargetKind::Manifold2d => {
            let lm = layout.cast::<PyManifoldLayout2d>()?;
            match extract_manifold_for_solver(lm, target_dtype)? {
                ExtractedManifold::F64(m) => Ok(TargetLayout::M2F64(m)),
                ExtractedManifold::F32(m) => Ok(TargetLayout::M2F32(m)),
            }
        }
    }
}

pub(crate) fn write_back_target_layout(
    layout: &Bound<'_, PyAny>,
    extracted: TargetLayout,
) -> PyResult<()> {
    match extracted {
        TargetLayout::L2F64(l) => {
            write_back_solver_layout(layout.cast::<PyLayout2d>()?, &SolverLayout::F64(l))
        }
        TargetLayout::L2F32(l) => {
            write_back_solver_layout(layout.cast::<PyLayout2d>()?, &SolverLayout::F32(l))
        }
        TargetLayout::L3F64(l) => {
            write_back_solver_layout3d(layout.cast::<PyLayout3d>()?, &SolverLayout3d::F64(l))
        }
        TargetLayout::L3F32(l) => {
            write_back_solver_layout3d(layout.cast::<PyLayout3d>()?, &SolverLayout3d::F32(l))
        }
        TargetLayout::NdF64(l) => {
            write_back_solver_layout_nd(layout.cast::<PyLayoutNd>()?, &SolverLayoutNd::F64(l))
        }
        TargetLayout::NdF32(l) => {
            write_back_solver_layout_nd(layout.cast::<PyLayoutNd>()?, &SolverLayoutNd::F32(l))
        }
        TargetLayout::M2F64(m) => write_back_manifold(
            layout.cast::<PyManifoldLayout2d>()?,
            ExtractedManifold::F64(m),
        ),
        TargetLayout::M2F32(m) => write_back_manifold(
            layout.cast::<PyManifoldLayout2d>()?,
            ExtractedManifold::F32(m),
        ),
    }
}
