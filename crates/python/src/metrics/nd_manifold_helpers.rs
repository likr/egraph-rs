//! Shared 2D, 3D, N-D, and manifold layout dispatch and result conversion helpers for metrics.

use egraph::core::{Layout2d, Layout3d, LayoutNd, ManifoldLayout2d};
use egraph::metrics::{CrossingPair, TorusCrossing};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::errors::core_error_to_py;
use crate::layout_buffers::{Layout2dInner, PyLayout2d};
use crate::layout_nd::{LayoutNdInner, PyLayoutNd};
use crate::layout3d::{Layout3dInner, PyLayout3d};
use crate::manifold::{ManifoldLayout2dInner, PyManifoldLayout2d, build_core_f32, build_core_f64};

#[derive(FromPyObject)]
pub(crate) enum MetricLayoutInput<'py> {
    Layout2d(Bound<'py, PyLayout2d>),
    Layout3d(Bound<'py, PyLayout3d>),
    LayoutNd(Bound<'py, PyLayoutNd>),
    Manifold(Bound<'py, PyManifoldLayout2d>),
}

pub(crate) enum OwnedMetricLayout {
    F64_2d(Layout2d<f64>),
    F32_2d(Layout2d<f32>),
    F64_3d(Layout3d<f64>),
    F32_3d(Layout3d<f32>),
    F64Nd(LayoutNd<f64>),
    F32Nd(LayoutNd<f32>),
    F64Manifold(ManifoldLayout2d<f64>),
    F32Manifold(ManifoldLayout2d<f32>),
}

impl MetricLayoutInput<'_> {
    pub(crate) fn extract_owned(&self) -> PyResult<OwnedMetricLayout> {
        match self {
            Self::Layout2d(obj) => {
                let b = obj.try_borrow()?;
                Ok(match b.inner() {
                    Layout2dInner::F64(buf) => OwnedMetricLayout::F64_2d(buf.to_layout2d()),
                    Layout2dInner::F32(buf) => OwnedMetricLayout::F32_2d(buf.to_layout2d()),
                })
            }
            Self::Layout3d(obj) => {
                let b = obj.try_borrow()?;
                Ok(match b.inner() {
                    Layout3dInner::F64(buf) => OwnedMetricLayout::F64_3d(buf.to_layout3d()),
                    Layout3dInner::F32(buf) => OwnedMetricLayout::F32_3d(buf.to_layout3d()),
                })
            }
            Self::LayoutNd(obj) => {
                let b = obj.try_borrow()?;
                Ok(match b.inner() {
                    LayoutNdInner::F64(nd) => OwnedMetricLayout::F64Nd(nd.clone()),
                    LayoutNdInner::F32(nd) => OwnedMetricLayout::F32Nd(nd.clone()),
                })
            }
            Self::Manifold(obj) => {
                let b = obj.try_borrow()?;
                match b.inner() {
                    ManifoldLayout2dInner::F64 { kind, metric, buf } => {
                        let m = build_core_f64(*kind, metric, buf).map_err(core_error_to_py)?;
                        Ok(OwnedMetricLayout::F64Manifold(m))
                    }
                    ManifoldLayout2dInner::F32 { kind, metric, buf } => {
                        let m = build_core_f32(*kind, metric, buf).map_err(core_error_to_py)?;
                        Ok(OwnedMetricLayout::F32Manifold(m))
                    }
                }
            }
        }
    }
}

macro_rules! dispatch_owned_metric {
    ($py:expr, $owned:expr, |$l:ident| $body:expr) => {{
        use $crate::metrics::nd_manifold_helpers::OwnedMetricLayout;
        $py.detach(|| match &$owned {
            OwnedMetricLayout::F64_2d($l) => $body,
            OwnedMetricLayout::F32_2d($l) => $body,
            OwnedMetricLayout::F64_3d($l) => $body,
            OwnedMetricLayout::F32_3d($l) => $body,
            OwnedMetricLayout::F64Nd($l) => $body,
            OwnedMetricLayout::F32Nd($l) => $body,
            OwnedMetricLayout::F64Manifold($l) => $body,
            OwnedMetricLayout::F32Manifold($l) => $body,
        })
        .map_err($crate::errors::core_error_to_py)
    }};
}

pub(crate) use dispatch_owned_metric;

pub(crate) fn crossing_pairs_to_py_dicts(
    py: Python<'_>,
    pairs: Vec<CrossingPair>,
) -> PyResult<Vec<Py<PyDict>>> {
    let mut out = Vec::with_capacity(pairs.len());
    for pair in pairs {
        let dict = PyDict::new(py);
        dict.set_item("edge_a", pair.edge_a)?;
        dict.set_item("edge_b", pair.edge_b)?;
        dict.set_item("u1", pair.u1)?;
        dict.set_item("v1", pair.v1)?;
        dict.set_item("u2", pair.u2)?;
        dict.set_item("v2", pair.v2)?;
        dict.set_item("angle_rad", pair.angle_rad)?;
        dict.set_item("cos_sq", pair.cos_sq)?;
        out.push(dict.unbind());
    }
    Ok(out)
}

pub(crate) fn torus_crossings_to_py_dicts(
    py: Python<'_>,
    pairs: Vec<TorusCrossing>,
) -> PyResult<Vec<Py<PyDict>>> {
    let mut out = Vec::with_capacity(pairs.len());
    for pair in pairs {
        let dict = PyDict::new(py);
        dict.set_item("edge_a", pair.edge_a)?;
        dict.set_item("edge_b", pair.edge_b)?;
        dict.set_item("u1", pair.u1)?;
        dict.set_item("v1", pair.v1)?;
        dict.set_item("u2", pair.u2)?;
        dict.set_item("v2", pair.v2)?;
        dict.set_item("seg_a", pair.seg_a.to_vec())?;
        dict.set_item("seg_b", pair.seg_b.to_vec())?;
        dict.set_item("angle_rad", pair.angle_rad)?;
        dict.set_item("cos_sq", pair.cos_sq)?;
        out.push(dict.unbind());
    }
    Ok(out)
}
