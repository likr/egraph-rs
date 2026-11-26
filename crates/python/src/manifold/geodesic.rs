//! Non-Euclidean manifold enum parsers, geodesic math functions, and solver bridges.

use egraph::core as egraph_core;
use egraph_core::{
    ManifoldKind, ManifoldLayout2d, poincare_distance as core_poincare_distance,
    poincare_mobius_add as core_poincare_mobius_add, spherical_distance as core_spherical_distance,
    torus_distance as core_torus_distance, wrap_torus_point as core_wrap_torus_point,
};
use numpy::PyArray2;
use numpy::ndarray::Array2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::ffi_convert::{ScalarDtype, StrictF64, core_error_to_py, normalize_enum_key};
use crate::solver_common::{layout_f32_to_f64_checked, layout_f64_to_f32_checked};

use super::layout::{ManifoldLayout2dInner, PyManifoldLayout2d};

pub(super) fn parse_manifold_kind(raw: &str) -> Result<ManifoldKind, String> {
    match normalize_enum_key(raw).as_str() {
        "hyperbolic" | "hyperbolicpoincare" | "poincare" => Ok(ManifoldKind::HyperbolicPoincare),
        "spherical" | "sphere" => Ok(ManifoldKind::Spherical),
        "torus" | "torus2d" => Ok(ManifoldKind::Torus2d),
        _ => Err(format!(
            "unknown manifold kind {raw:?}; expected 'hyperbolic', 'spherical', or 'torus'"
        )),
    }
}

pub(super) fn manifold_kind_str(kind: ManifoldKind) -> &'static str {
    match kind {
        ManifoldKind::HyperbolicPoincare => "hyperbolic",
        ManifoldKind::Spherical => "spherical",
        ManifoldKind::Torus2d => "torus",
        _ => "unknown",
    }
}

pub(super) fn manifold_coords_array<'py>(
    py: Python<'py>,
    inner: &ManifoldLayout2dInner,
) -> PyResult<Bound<'py, PyAny>> {
    match inner {
        ManifoldLayout2dInner::F64(m) => {
            let interleaved = m.as_layout().to_interleaved();
            let arr = Array2::from_shape_vec((m.len(), 2), interleaved)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(PyArray2::from_owned_array(py, arr).into_any())
        }
        ManifoldLayout2dInner::F32(m) => {
            let interleaved = m.as_layout().to_interleaved();
            let arr = Array2::from_shape_vec((m.len(), 2), interleaved)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(PyArray2::from_owned_array(py, arr).into_any())
        }
    }
}

pub(super) fn checked_f64_to_f32(val: f64, ctx: &str) -> PyResult<f32> {
    let f = val as f32;
    if val.is_finite() && !f.is_finite() {
        return Err(PyValueError::new_err(format!(
            "{ctx}={val} overflows float32"
        )));
    }
    Ok(f)
}

#[derive(Clone)]
pub(crate) enum ExtractedManifold {
    F64(ManifoldLayout2d<f64>, ScalarDtype),
    F32(ManifoldLayout2d<f32>, ScalarDtype),
}

pub(crate) fn extract_manifold_for_solver(
    layout: &Bound<'_, PyManifoldLayout2d>,
    target_dtype: ScalarDtype,
) -> PyResult<ExtractedManifold> {
    let borrowed = layout.try_borrow()?;
    let orig_dtype = borrowed.dtype_enum();
    match (borrowed.inner(), target_dtype) {
        (ManifoldLayout2dInner::F64(m), ScalarDtype::F64) => {
            Ok(ExtractedManifold::F64(m.clone(), orig_dtype))
        }
        (ManifoldLayout2dInner::F32(m), ScalarDtype::F32) => {
            Ok(ExtractedManifold::F32(m.clone(), orig_dtype))
        }
        (ManifoldLayout2dInner::F32(m), ScalarDtype::F64) => {
            let soa64 = layout_f32_to_f64_checked(m.as_layout()).map_err(core_error_to_py)?;
            let m64 = ManifoldLayout2d::from_layout(m.kind(), soa64).map_err(core_error_to_py)?;
            Ok(ExtractedManifold::F64(m64, orig_dtype))
        }
        (ManifoldLayout2dInner::F64(m), ScalarDtype::F32) => {
            let soa32 = layout_f64_to_f32_checked(m.as_layout()).map_err(core_error_to_py)?;
            let m32 = ManifoldLayout2d::from_layout(m.kind(), soa32).map_err(core_error_to_py)?;
            Ok(ExtractedManifold::F32(m32, orig_dtype))
        }
    }
}

pub(crate) fn write_back_manifold(
    layout: &Bound<'_, PyManifoldLayout2d>,
    extracted: ExtractedManifold,
) -> PyResult<()> {
    let mut borrowed = layout.try_borrow_mut()?;
    let next = match extracted {
        ExtractedManifold::F64(m64, ScalarDtype::F64) => ManifoldLayout2dInner::F64(m64),
        ExtractedManifold::F32(m32, ScalarDtype::F32) => ManifoldLayout2dInner::F32(m32),
        ExtractedManifold::F64(m64, ScalarDtype::F32) => {
            let soa32 = layout_f64_to_f32_checked(m64.as_layout()).map_err(core_error_to_py)?;
            let m32 = ManifoldLayout2d::from_layout(m64.kind(), soa32).map_err(core_error_to_py)?;
            ManifoldLayout2dInner::F32(m32)
        }
        ExtractedManifold::F32(m32, ScalarDtype::F64) => {
            let soa64 = layout_f32_to_f64_checked(m32.as_layout()).map_err(core_error_to_py)?;
            let m64 = ManifoldLayout2d::from_layout(m32.kind(), soa64).map_err(core_error_to_py)?;
            ManifoldLayout2dInner::F64(m64)
        }
    };
    borrowed.set_inner(next);
    Ok(())
}

/// Compute exact Poincaré disk hyperbolic geodesic distance between `(x1, y1)` and `(x2, y2)`.
#[pyfunction]
#[pyo3(signature = (x1, y1, x2, y2), text_signature = "(x1, y1, x2, y2)")]
fn poincare_distance(x1: StrictF64, y1: StrictF64, x2: StrictF64, y2: StrictF64) -> f64 {
    core_poincare_distance(x1.0, y1.0, x2.0, y2.0)
}

/// Compute Poincaré disk Möbius addition `(x1, y1) (+) (x2, y2)` inside the open unit disk.
#[pyfunction]
#[pyo3(signature = (x1, y1, x2, y2), text_signature = "(x1, y1, x2, y2)")]
fn poincare_mobius_add(x1: StrictF64, y1: StrictF64, x2: StrictF64, y2: StrictF64) -> (f64, f64) {
    core_poincare_mobius_add(x1.0, y1.0, x2.0, y2.0)
}

/// Compute great-circle distance on the unit sphere given `(lon, colatitude)` in radians.
#[pyfunction]
#[pyo3(signature = (lon1, colat1, lon2, colat2), text_signature = "(lon1, colat1, lon2, colat2)")]
fn spherical_distance(
    lon1: StrictF64,
    colat1: StrictF64,
    lon2: StrictF64,
    colat2: StrictF64,
) -> f64 {
    core_spherical_distance(lon1.0, colat1.0, lon2.0, colat2.0)
}

/// Compute shortest periodic Euclidean distance on the flat unit torus `[0, 1)^2`.
#[pyfunction]
#[pyo3(signature = (x1, y1, x2, y2), text_signature = "(x1, y1, x2, y2)")]
fn torus_distance(x1: StrictF64, y1: StrictF64, x2: StrictF64, y2: StrictF64) -> f64 {
    core_torus_distance(x1.0, y1.0, x2.0, y2.0)
}

/// Wrap `(x, y)` periodically into the fundamental domain `[0, 1)^2` of the unit torus.
#[pyfunction]
#[pyo3(signature = (x, y), text_signature = "(x, y)")]
fn wrap_torus_point(x: StrictF64, y: StrictF64) -> (f64, f64) {
    core_wrap_torus_point(x.0, y.0)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(poincare_distance, m)?)?;
    m.add_function(wrap_pyfunction!(poincare_mobius_add, m)?)?;
    m.add_function(wrap_pyfunction!(spherical_distance, m)?)?;
    m.add_function(wrap_pyfunction!(torus_distance, m)?)?;
    m.add_function(wrap_pyfunction!(wrap_torus_point, m)?)?;
    Ok(())
}
