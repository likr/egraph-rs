//! Non-Euclidean manifold enum parsers, geodesic math functions, and solver bridges.

use egraph::core as egraph_core;
use egraph_core::{
    CoreResult, ManifoldKind, ManifoldLayout2d, ManifoldMetric,
    lorentz_distance as core_lorentz_distance,
    lorentz_einstein_midpoint as core_lorentz_einstein_midpoint,
    lorentz_exp_map as core_lorentz_exp_map, lorentz_log_map as core_lorentz_log_map,
    lorentz_to_poincare_point as core_lorentz_to_poincare_point,
    poincare_distance as core_poincare_distance,
    poincare_einstein_midpoint as core_poincare_einstein_midpoint,
    poincare_mobius_add as core_poincare_mobius_add, spherical_distance as core_spherical_distance,
    torus_distance as core_torus_distance, wrap_torus_point as core_wrap_torus_point,
};
use egraph_ffi::InterleavedLayout2d;
use numpy::PyArray2;
use numpy::ndarray::Array2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::array_extract::extract_f64_1d;
use crate::ffi_convert::{ScalarDtype, StrictF64, core_error_to_py, normalize_enum_key};
use crate::solver_common::{layout_f32_to_f64_checked, layout_f64_to_f32_checked};

use super::layout::{ManifoldLayout2dInner, PyManifoldLayout2d};

pub(super) fn parse_manifold_kind(raw: &str) -> Result<ManifoldKind, String> {
    match normalize_enum_key(raw).as_str() {
        "hyperbolic" | "hyperbolicpoincare" | "poincare" => Ok(ManifoldKind::HyperbolicPoincare),
        "hyperboliclorentz" | "lorentz" | "hyperboloid" => Ok(ManifoldKind::HyperbolicLorentz),
        "spherical" | "sphere" => Ok(ManifoldKind::Spherical),
        "torus" | "torus2d" => Ok(ManifoldKind::Torus2d),
        _ => Err(format!(
            "unknown manifold kind {raw:?}; expected 'hyperbolic', 'hyperbolic_lorentz', 'spherical', or 'torus'"
        )),
    }
}

pub(super) fn manifold_kind_str(kind: ManifoldKind) -> &'static str {
    match kind {
        ManifoldKind::HyperbolicPoincare => "hyperbolic",
        ManifoldKind::HyperbolicLorentz => "hyperbolic_lorentz",
        ManifoldKind::Spherical => "spherical",
        ManifoldKind::Torus2d => "torus",
        _ => "unknown",
    }
}

pub(super) fn inner_from_core_f64(m: ManifoldLayout2d<f64>) -> ManifoldLayout2dInner {
    ManifoldLayout2dInner::F64 {
        kind: m.kind(),
        metric: m.metric(),
        buf: InterleavedLayout2d::from_layout2d(m.as_layout()),
    }
}

pub(super) fn inner_from_core_f32(m: ManifoldLayout2d<f32>) -> ManifoldLayout2dInner {
    ManifoldLayout2dInner::F32 {
        kind: m.kind(),
        metric: m.metric(),
        buf: InterleavedLayout2d::from_layout2d(m.as_layout()),
    }
}

pub(crate) fn build_core_f64(
    kind: ManifoldKind,
    metric: ManifoldMetric,
    buf: &InterleavedLayout2d<f64>,
) -> CoreResult<ManifoldLayout2d<f64>> {
    let mut m = ManifoldLayout2d::from_layout(kind, buf.to_layout2d())?;
    m.set_scale(metric.scale)?;
    m.set_aspect(metric.aspect)?;
    Ok(m)
}

pub(crate) fn build_core_f32(
    kind: ManifoldKind,
    metric: ManifoldMetric,
    buf: &InterleavedLayout2d<f32>,
) -> CoreResult<ManifoldLayout2d<f32>> {
    let mut m = ManifoldLayout2d::from_layout(kind, buf.to_layout2d())?;
    m.set_scale(metric.scale)?;
    m.set_aspect(metric.aspect)?;
    Ok(m)
}

pub(super) fn manifold_coords_array<'py>(
    slf: Bound<'py, PyManifoldLayout2d>,
    copy: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let py = slf.py();
    if copy {
        let borrowed = slf.borrow();
        return match borrowed.inner() {
            ManifoldLayout2dInner::F64 { buf, .. } => {
                let arr = Array2::from_shape_vec((buf.len(), 2), buf.as_slice().to_vec())
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyArray2::from_owned_array(py, arr).into_any())
            }
            ManifoldLayout2dInner::F32 { buf, .. } => {
                let arr = Array2::from_shape_vec((buf.len(), 2), buf.as_slice().to_vec())
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyArray2::from_owned_array(py, arr).into_any())
            }
        };
    }

    let container = slf.clone().into_any();
    let mut borrowed = slf.borrow_mut();
    match borrowed.inner_mut() {
        ManifoldLayout2dInner::F64 { buf, .. } => {
            let n = buf.len();
            let view = numpy::ndarray::ArrayViewMut2::from_shape((n, 2), buf.as_mut_slice())
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            // SAFETY: `container` is `slf` (`PyManifoldLayout2d`), registered as the NumPy
            // array's `base` object so `buf` outlives any view and never reallocates.
            Ok(unsafe { PyArray2::borrow_from_array(&view, container) }.into_any())
        }
        ManifoldLayout2dInner::F32 { buf, .. } => {
            let n = buf.len();
            let view = numpy::ndarray::ArrayViewMut2::from_shape((n, 2), buf.as_mut_slice())
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            // SAFETY: `container` is `slf` (`PyManifoldLayout2d`), registered as the NumPy
            // array's `base` object so `buf` outlives any view and never reallocates.
            Ok(unsafe { PyArray2::borrow_from_array(&view, container) }.into_any())
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
    F64(ManifoldLayout2d<f64>),
    F32(ManifoldLayout2d<f32>),
}

pub(crate) fn extract_manifold_for_solver(
    layout: &Bound<'_, PyManifoldLayout2d>,
    target_dtype: ScalarDtype,
) -> PyResult<ExtractedManifold> {
    let borrowed = layout.try_borrow()?;
    match (borrowed.inner(), target_dtype) {
        (ManifoldLayout2dInner::F64 { kind, metric, buf }, ScalarDtype::F64) => {
            let m = build_core_f64(*kind, *metric, buf).map_err(core_error_to_py)?;
            Ok(ExtractedManifold::F64(m))
        }
        (ManifoldLayout2dInner::F32 { kind, metric, buf }, ScalarDtype::F32) => {
            let m = build_core_f32(*kind, *metric, buf).map_err(core_error_to_py)?;
            Ok(ExtractedManifold::F32(m))
        }
        (ManifoldLayout2dInner::F32 { kind, metric, buf }, ScalarDtype::F64) => {
            let soa64 = layout_f32_to_f64_checked(&buf.to_layout2d()).map_err(core_error_to_py)?;
            let mut m64 = ManifoldLayout2d::from_layout(*kind, soa64).map_err(core_error_to_py)?;
            m64.set_scale(metric.scale).map_err(core_error_to_py)?;
            m64.set_aspect(metric.aspect).map_err(core_error_to_py)?;
            Ok(ExtractedManifold::F64(m64))
        }
        (ManifoldLayout2dInner::F64 { kind, metric, buf }, ScalarDtype::F32) => {
            let soa32 = layout_f64_to_f32_checked(&buf.to_layout2d()).map_err(core_error_to_py)?;
            let mut m32 = ManifoldLayout2d::from_layout(*kind, soa32).map_err(core_error_to_py)?;
            m32.set_scale(metric.scale).map_err(core_error_to_py)?;
            m32.set_aspect(metric.aspect).map_err(core_error_to_py)?;
            Ok(ExtractedManifold::F32(m32))
        }
    }
}

pub(crate) fn write_back_manifold(
    layout: &Bound<'_, PyManifoldLayout2d>,
    extracted: ExtractedManifold,
) -> PyResult<()> {
    let mut borrowed = layout.try_borrow_mut()?;
    match (borrowed.inner_mut(), extracted) {
        (ManifoldLayout2dInner::F64 { kind, metric, buf }, ExtractedManifold::F64(m64)) => {
            *kind = m64.kind();
            *metric = m64.metric();
            buf.copy_from_layout2d(m64.as_layout())
                .map_err(crate::ffi_convert::ffi_error_to_py)
        }
        (ManifoldLayout2dInner::F32 { kind, metric, buf }, ExtractedManifold::F32(m32)) => {
            *kind = m32.kind();
            *metric = m32.metric();
            buf.copy_from_layout2d(m32.as_layout())
                .map_err(crate::ffi_convert::ffi_error_to_py)
        }
        (ManifoldLayout2dInner::F32 { kind, metric, buf }, ExtractedManifold::F64(m64)) => {
            let soa32 = layout_f64_to_f32_checked(m64.as_layout()).map_err(core_error_to_py)?;
            *kind = m64.kind();
            *metric = m64.metric();
            buf.copy_from_layout2d(&soa32)
                .map_err(crate::ffi_convert::ffi_error_to_py)
        }
        (ManifoldLayout2dInner::F64 { kind, metric, buf }, ExtractedManifold::F32(m32)) => {
            let soa64 = layout_f32_to_f64_checked(m32.as_layout()).map_err(core_error_to_py)?;
            *kind = m32.kind();
            *metric = m32.metric();
            buf.copy_from_layout2d(&soa64)
                .map_err(crate::ffi_convert::ffi_error_to_py)
        }
    }
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

/// Compute the Poincaré disk Einstein midpoint of points `(xs, ys)`.
#[pyfunction]
#[pyo3(signature = (xs, ys), text_signature = "(xs, ys)")]
fn poincare_einstein_midpoint(
    xs: &Bound<'_, PyAny>,
    ys: &Bound<'_, PyAny>,
) -> PyResult<(f64, f64)> {
    let x_vec = extract_f64_1d(xs, "xs")?;
    let y_vec = extract_f64_1d(ys, "ys")?;
    core_poincare_einstein_midpoint(&x_vec, &y_vec).map_err(core_error_to_py)
}

/// Compute hyperbolic geodesic distance in spatial Lorentz hyperboloid coordinates `(x, y)`.
#[pyfunction]
#[pyo3(signature = (x1, y1, x2, y2), text_signature = "(x1, y1, x2, y2)")]
fn lorentz_distance(x1: StrictF64, y1: StrictF64, x2: StrictF64, y2: StrictF64) -> f64 {
    core_lorentz_distance(x1.0, y1.0, x2.0, y2.0)
}

/// Project spatial Lorentz hyperboloid point `(x, y)` onto the Poincaré unit disk.
#[pyfunction]
#[pyo3(signature = (x, y), text_signature = "(x, y)")]
fn lorentz_to_poincare_point(x: StrictF64, y: StrictF64) -> (f64, f64) {
    core_lorentz_to_poincare_point(x.0, y.0)
}

/// Advance Lorentz point `(x, y)` along geodesic tangent vector `(vx, vy)`.
#[pyfunction]
#[pyo3(signature = (x, y, vx, vy), text_signature = "(x, y, vx, vy)")]
fn lorentz_exp_map(x: StrictF64, y: StrictF64, vx: StrictF64, vy: StrictF64) -> (f64, f64) {
    core_lorentz_exp_map(x.0, y.0, vx.0, vy.0)
}

/// Compute the Riemannian logarithmic map at Lorentz point `(x1, y1)` toward `(x2, y2)`.
#[pyfunction]
#[pyo3(signature = (x1, y1, x2, y2), text_signature = "(x1, y1, x2, y2)")]
fn lorentz_log_map(x1: StrictF64, y1: StrictF64, x2: StrictF64, y2: StrictF64) -> (f64, f64) {
    core_lorentz_log_map(x1.0, y1.0, x2.0, y2.0)
}

/// Compute the hyperbolic Einstein midpoint of Lorentz hyperboloid points `(xs, ys)`.
#[pyfunction]
#[pyo3(signature = (xs, ys), text_signature = "(xs, ys)")]
fn lorentz_einstein_midpoint(xs: &Bound<'_, PyAny>, ys: &Bound<'_, PyAny>) -> PyResult<(f64, f64)> {
    let x_vec = extract_f64_1d(xs, "xs")?;
    let y_vec = extract_f64_1d(ys, "ys")?;
    core_lorentz_einstein_midpoint(&x_vec, &y_vec).map_err(core_error_to_py)
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
    m.add_function(wrap_pyfunction!(poincare_einstein_midpoint, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_distance, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_to_poincare_point, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_exp_map, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_log_map, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_einstein_midpoint, m)?)?;
    m.add_function(wrap_pyfunction!(spherical_distance, m)?)?;
    m.add_function(wrap_pyfunction!(torus_distance, m)?)?;
    m.add_function(wrap_pyfunction!(wrap_torus_point, m)?)?;
    Ok(())
}
