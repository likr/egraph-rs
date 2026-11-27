//! Helper operations backing `ManifoldLayout2d` Python bindings.

use egraph::core::ManifoldLayout2d;
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, ffi_error_to_py,
};
use crate::layout_buffers::{Layout2dInner, PyLayout2d, extract_layout_inner};

use super::geodesic::{
    build_core_f32, build_core_f64, checked_f64_to_f32, inner_from_core_f32, inner_from_core_f64,
    parse_manifold_kind,
};
use super::layout::ManifoldLayout2dInner;

pub(super) fn from_layout_impl(
    py: Python<'_>,
    kind: &str,
    layout: &PyLayout2d,
) -> PyResult<ManifoldLayout2dInner> {
    let mk = parse_manifold_kind(kind).map_err(pyo3::exceptions::PyValueError::new_err)?;
    py.detach(|| match layout.inner() {
        Layout2dInner::F64(buf) => {
            ManifoldLayout2d::from_layout(mk, buf.to_layout2d()).map(inner_from_core_f64)
        }
        Layout2dInner::F32(buf) => {
            ManifoldLayout2d::from_layout(mk, buf.to_layout2d()).map(inner_from_core_f32)
        }
    })
    .map_err(core_error_to_py)
}

pub(super) fn from_coords_impl(
    py: Python<'_>,
    kind: &str,
    coords: &Bound<'_, PyAny>,
) -> PyResult<ManifoldLayout2dInner> {
    let mk = parse_manifold_kind(kind).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let buf_inner = extract_layout_inner(coords)?;
    py.detach(|| match buf_inner {
        Layout2dInner::F64(buf) => {
            ManifoldLayout2d::from_layout(mk, buf.to_layout2d()).map(inner_from_core_f64)
        }
        Layout2dInner::F32(buf) => {
            ManifoldLayout2d::from_layout(mk, buf.to_layout2d()).map(inner_from_core_f32)
        }
    })
    .map_err(core_error_to_py)
}

pub(super) fn random_init_impl(
    py: Python<'_>,
    kind: &str,
    n: StrictUsize,
    scale: StrictF64,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<ManifoldLayout2dInner> {
    let mk = parse_manifold_kind(kind).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let (n_val, scale_val, seed_val) = (n.0, scale.0, seed.0);
    py.detach(|| match dtype {
        ScalarDtype::F64 => ManifoldLayout2d::<f64>::random_init(mk, n_val, scale_val, seed_val)
            .map(inner_from_core_f64),
        ScalarDtype::F32 => ManifoldLayout2d::<f32>::random_init(mk, n_val, scale_val, seed_val)
            .map(inner_from_core_f32),
    })
    .map_err(core_error_to_py)
}

pub(super) fn from_layout_scaled_impl(
    py: Python<'_>,
    kind: &str,
    layout: &PyLayout2d,
    max_radius: StrictF64,
) -> PyResult<ManifoldLayout2dInner> {
    let mk = parse_manifold_kind(kind).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let mr = max_radius.0;
    py.detach(|| match layout.inner() {
        Layout2dInner::F64(buf) => ManifoldLayout2d::from_layout_scaled(mk, &buf.to_layout2d(), mr)
            .map(inner_from_core_f64),
        Layout2dInner::F32(buf) => ManifoldLayout2d::from_layout_scaled(mk, &buf.to_layout2d(), mr)
            .map(inner_from_core_f32),
    })
    .map_err(core_error_to_py)
}

pub(super) fn torus_spectral_phase_init_impl(
    py: Python<'_>,
    graph: &PyCsrGraph,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<ManifoldLayout2dInner> {
    let g = graph.inner();
    let seed_val = seed.0;
    py.detach(|| match dtype {
        ScalarDtype::F64 => ManifoldLayout2d::<f64>::torus_spectral_phase_init(&*g, seed_val)
            .map(inner_from_core_f64),
        ScalarDtype::F32 => ManifoldLayout2d::<f32>::torus_spectral_phase_init(&*g, seed_val)
            .map(inner_from_core_f32),
    })
    .map_err(core_error_to_py)
}

pub(super) fn set_torus_extents_impl(
    inner: &mut ManifoldLayout2dInner,
    lx: f64,
    ly: f64,
) -> PyResult<()> {
    match inner {
        ManifoldLayout2dInner::F64 { kind, metric, buf } => {
            let mut m = build_core_f64(*kind, *metric, buf).map_err(core_error_to_py)?;
            m.set_torus_extents(lx, ly).map_err(core_error_to_py)?;
            *metric = m.metric();
            Ok(())
        }
        ManifoldLayout2dInner::F32 { kind, metric, buf } => {
            let mut m = build_core_f32(*kind, *metric, buf).map_err(core_error_to_py)?;
            m.set_torus_extents(lx, ly).map_err(core_error_to_py)?;
            *metric = m.metric();
            Ok(())
        }
    }
}

pub(super) fn rescale_geodesic_impl(
    inner: &mut ManifoldLayout2dInner,
    new_scale: f64,
) -> PyResult<()> {
    match inner {
        ManifoldLayout2dInner::F64 { kind, metric, buf } => {
            let mut m = build_core_f64(*kind, *metric, buf).map_err(core_error_to_py)?;
            m.rescale_geodesic(new_scale).map_err(core_error_to_py)?;
            *metric = m.metric();
            buf.copy_from_layout2d(m.as_layout())
                .map_err(ffi_error_to_py)
        }
        ManifoldLayout2dInner::F32 { kind, metric, buf } => {
            let mut m = build_core_f32(*kind, *metric, buf).map_err(core_error_to_py)?;
            m.rescale_geodesic(new_scale).map_err(core_error_to_py)?;
            *metric = m.metric();
            buf.copy_from_layout2d(m.as_layout())
                .map_err(ffi_error_to_py)
        }
    }
}

pub(super) fn set_point_impl(
    inner: &mut ManifoldLayout2dInner,
    index: usize,
    x: f64,
    y: f64,
) -> PyResult<()> {
    match inner {
        ManifoldLayout2dInner::F64 { kind, metric, buf } => {
            let mut m = build_core_f64(*kind, *metric, buf).map_err(core_error_to_py)?;
            m.set_point(index, x, y).map_err(core_error_to_py)?;
            buf.copy_from_layout2d(m.as_layout())
                .map_err(ffi_error_to_py)
        }
        ManifoldLayout2dInner::F32 { kind, metric, buf } => {
            let x32 = checked_f64_to_f32(x, "ManifoldLayout2d.set_point x")?;
            let y32 = checked_f64_to_f32(y, "ManifoldLayout2d.set_point y")?;
            let mut m = build_core_f32(*kind, *metric, buf).map_err(core_error_to_py)?;
            m.set_point(index, x32, y32).map_err(core_error_to_py)?;
            buf.copy_from_layout2d(m.as_layout())
                .map_err(ffi_error_to_py)
        }
    }
}

pub(super) fn apply_tangent_step_impl(
    inner: &mut ManifoldLayout2dInner,
    i: usize,
    vx: f64,
    vy: f64,
) -> PyResult<()> {
    match inner {
        ManifoldLayout2dInner::F64 { kind, metric, buf } => {
            let mut m = build_core_f64(*kind, *metric, buf).map_err(core_error_to_py)?;
            m.apply_tangent_step(i, vx, vy).map_err(core_error_to_py)?;
            buf.copy_from_layout2d(m.as_layout())
                .map_err(ffi_error_to_py)
        }
        ManifoldLayout2dInner::F32 { kind, metric, buf } => {
            let vx32 = checked_f64_to_f32(vx, "ManifoldLayout2d.apply_tangent_step vx")?;
            let vy32 = checked_f64_to_f32(vy, "ManifoldLayout2d.apply_tangent_step vy")?;
            let mut m = build_core_f32(*kind, *metric, buf).map_err(core_error_to_py)?;
            m.apply_tangent_step(i, vx32, vy32)
                .map_err(core_error_to_py)?;
            buf.copy_from_layout2d(m.as_layout())
                .map_err(ffi_error_to_py)
        }
    }
}

pub(super) fn project_all_impl(inner: &mut ManifoldLayout2dInner) -> PyResult<()> {
    match inner {
        ManifoldLayout2dInner::F64 { kind, metric, buf } => {
            let m = build_core_f64(*kind, *metric, buf).map_err(core_error_to_py)?;
            buf.copy_from_layout2d(m.as_layout())
                .map_err(ffi_error_to_py)
        }
        ManifoldLayout2dInner::F32 { kind, metric, buf } => {
            let m = build_core_f32(*kind, *metric, buf).map_err(core_error_to_py)?;
            buf.copy_from_layout2d(m.as_layout())
                .map_err(ffi_error_to_py)
        }
    }
}

pub(super) fn recenter_poincare_impl(
    inner: &mut ManifoldLayout2dInner,
    cx: f64,
    cy: f64,
) -> PyResult<()> {
    match inner {
        ManifoldLayout2dInner::F64 { kind, metric, buf } => {
            let mut m = build_core_f64(*kind, *metric, buf).map_err(core_error_to_py)?;
            m.recenter_poincare(cx, cy).map_err(core_error_to_py)?;
            buf.copy_from_layout2d(m.as_layout())
                .map_err(ffi_error_to_py)
        }
        ManifoldLayout2dInner::F32 { kind, metric, buf } => {
            let cx32 = checked_f64_to_f32(cx, "ManifoldLayout2d.recenter_poincare cx")?;
            let cy32 = checked_f64_to_f32(cy, "ManifoldLayout2d.recenter_poincare cy")?;
            let mut m = build_core_f32(*kind, *metric, buf).map_err(core_error_to_py)?;
            m.recenter_poincare(cx32, cy32).map_err(core_error_to_py)?;
            buf.copy_from_layout2d(m.as_layout())
                .map_err(ffi_error_to_py)
        }
    }
}

pub(super) fn recenter_poincare_at_impl(
    inner: &mut ManifoldLayout2dInner,
    index: usize,
) -> PyResult<()> {
    match inner {
        ManifoldLayout2dInner::F64 { kind, metric, buf } => {
            let mut m = build_core_f64(*kind, *metric, buf).map_err(core_error_to_py)?;
            m.recenter_poincare_at(index).map_err(core_error_to_py)?;
            buf.copy_from_layout2d(m.as_layout())
                .map_err(ffi_error_to_py)
        }
        ManifoldLayout2dInner::F32 { kind, metric, buf } => {
            let mut m = build_core_f32(*kind, *metric, buf).map_err(core_error_to_py)?;
            m.recenter_poincare_at(index).map_err(core_error_to_py)?;
            buf.copy_from_layout2d(m.as_layout())
                .map_err(ffi_error_to_py)
        }
    }
}

pub(super) fn center_poincare_centroid_impl(inner: &mut ManifoldLayout2dInner) -> PyResult<()> {
    match inner {
        ManifoldLayout2dInner::F64 { kind, metric, buf } => {
            let mut m = build_core_f64(*kind, *metric, buf).map_err(core_error_to_py)?;
            m.center_poincare_centroid().map_err(core_error_to_py)?;
            buf.copy_from_layout2d(m.as_layout())
                .map_err(ffi_error_to_py)
        }
        ManifoldLayout2dInner::F32 { kind, metric, buf } => {
            let mut m = build_core_f32(*kind, *metric, buf).map_err(core_error_to_py)?;
            m.center_poincare_centroid().map_err(core_error_to_py)?;
            buf.copy_from_layout2d(m.as_layout())
                .map_err(ffi_error_to_py)
        }
    }
}

pub(super) fn to_poincare_layout_impl(
    py: Python<'_>,
    inner: &ManifoldLayout2dInner,
) -> PyResult<ManifoldLayout2dInner> {
    py.detach(|| match inner {
        ManifoldLayout2dInner::F64 { kind, metric, buf } => {
            let m = build_core_f64(*kind, *metric, buf)?;
            m.to_poincare_layout().map(inner_from_core_f64)
        }
        ManifoldLayout2dInner::F32 { kind, metric, buf } => {
            let m = build_core_f32(*kind, *metric, buf)?;
            m.to_poincare_layout().map(inner_from_core_f32)
        }
    })
    .map_err(core_error_to_py)
}

pub(super) fn torus_auto_pan_impl(
    py: Python<'_>,
    inner: &ManifoldLayout2dInner,
    graph: &PyCsrGraph,
) -> PyResult<ManifoldLayout2dInner> {
    let g = graph.inner();
    py.detach(|| match inner {
        ManifoldLayout2dInner::F64 { kind, metric, buf } => {
            let m = build_core_f64(*kind, *metric, buf)?;
            m.torus_auto_pan(&*g).map(inner_from_core_f64)
        }
        ManifoldLayout2dInner::F32 { kind, metric, buf } => {
            let m = build_core_f32(*kind, *metric, buf)?;
            m.torus_auto_pan(&*g).map(inner_from_core_f32)
        }
    })
    .map_err(core_error_to_py)
}
