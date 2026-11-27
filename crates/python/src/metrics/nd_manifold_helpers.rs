//! Shared 2D, N-D, and manifold layout dispatch and dict conversion helpers for metrics.

use egraph::core as egraph_core;
use egraph::metrics as egraph_metrics;
use egraph_core::{CoreResult, Layout2d, LayoutNd, ManifoldLayout2d};
use egraph_metrics::{
    CrossingPair, LayoutDegeneracy, NorMetrics, PartitionAgreement, StressEvaluation, TorusCrossing,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::layout_buffers::{Layout2dInner, PyLayout2d};
use crate::layout_nd::{LayoutNdInner, PyLayoutNd};
use crate::manifold::{ManifoldLayout2dInner, PyManifoldLayout2d, build_core_f32, build_core_f64};

#[derive(FromPyObject)]
pub(crate) enum MetricLayoutInput<'py> {
    Layout2d(Bound<'py, PyLayout2d>),
    LayoutNd(Bound<'py, PyLayoutNd>),
    Manifold(Bound<'py, PyManifoldLayout2d>),
}

pub(crate) enum OwnedMetricLayout {
    F64_2d(Layout2d<f64>),
    F32_2d(Layout2d<f32>),
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
                        let m = build_core_f64(*kind, *metric, buf)
                            .map_err(crate::ffi_convert::core_error_to_py)?;
                        Ok(OwnedMetricLayout::F64Manifold(m))
                    }
                    ManifoldLayout2dInner::F32 { kind, metric, buf } => {
                        let m = build_core_f32(*kind, *metric, buf)
                            .map_err(crate::ffi_convert::core_error_to_py)?;
                        Ok(OwnedMetricLayout::F32Manifold(m))
                    }
                }
            }
        }
    }
}

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

pub(crate) fn layout_degeneracy_to_py_dict(
    py: Python<'_>,
    deg: LayoutDegeneracy,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("num_nodes", deg.num_nodes)?;
    dict.set_item("num_edges", deg.num_edges)?;
    dict.set_item("duplicate_nodes_f32", deg.duplicate_nodes_f32)?;
    dict.set_item("duplicate_ratio_f32", deg.duplicate_ratio_f32)?;
    dict.set_item("duplicate_nodes_f64", deg.duplicate_nodes_f64)?;
    dict.set_item("duplicate_ratio_f64", deg.duplicate_ratio_f64)?;
    dict.set_item("excess_duplicate_ratio", deg.excess_duplicate_ratio)?;
    dict.set_item("near_collision_ratio", deg.near_collision_ratio)?;
    dict.set_item("zero_length_edges", deg.zero_length_edges)?;
    dict.set_item("zero_length_edge_ratio", deg.zero_length_edge_ratio)?;
    dict.set_item("bbox_width", deg.bbox_width)?;
    dict.set_item("bbox_height", deg.bbox_height)?;
    dict.set_item("is_bbox_degenerate", deg.is_bbox_degenerate)?;
    Ok(dict.unbind())
}

pub(crate) fn stress_evaluation_to_py_dict(
    py: Python<'_>,
    eval: StressEvaluation,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("normalized_stress", eval.normalized_stress)?;
    dict.set_item("raw_stress", eval.raw_stress)?;
    dict.set_item("alpha_star", eval.alpha_star)?;
    dict.set_item("pair_count", eval.pair_count)?;
    dict.set_item("weight_distance_sq_sum", eval.weight_distance_sq_sum)?;
    Ok(dict.unbind())
}

pub(crate) fn partition_agreement_to_py_dict(
    py: Python<'_>,
    pa: PartitionAgreement,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("ari", pa.ari)?;
    dict.set_item("ami", pa.ami)?;
    dict.set_item("homogeneity", pa.homogeneity)?;
    dict.set_item("completeness", pa.completeness)?;
    dict.set_item("v_measure", pa.v_measure)?;
    dict.set_item("fmi", pa.fmi)?;
    Ok(dict.unbind())
}

pub(crate) fn nor_metrics_to_py_dict(py: Python<'_>, nor: NorMetrics) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("oo_nni", nor.oo_nni)?;
    dict.set_item("sp_ch_a", nor.sp_ch_a)?;
    dict.set_item("gs_bb_iar", nor.gs_bb_iar)?;
    dict.set_item("nm_dm_imse", nor.nm_dm_imse)?;
    dict.set_item("el_rsd", nor.el_rsd)?;
    Ok(dict.unbind())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_owned_metric_layout<R: Send>(
    py: Python<'_>,
    owned: OwnedMetricLayout,
    f_2d_64: impl FnOnce(&Layout2d<f64>) -> CoreResult<R> + Send,
    f_2d_32: impl FnOnce(&Layout2d<f32>) -> CoreResult<R> + Send,
    f_nd_64: impl FnOnce(&LayoutNd<f64>) -> CoreResult<R> + Send,
    f_nd_32: impl FnOnce(&LayoutNd<f32>) -> CoreResult<R> + Send,
    f_man_64: impl FnOnce(&ManifoldLayout2d<f64>) -> CoreResult<R> + Send,
    f_man_32: impl FnOnce(&ManifoldLayout2d<f32>) -> CoreResult<R> + Send,
) -> PyResult<R> {
    py.detach(|| match &owned {
        OwnedMetricLayout::F64_2d(l) => f_2d_64(l),
        OwnedMetricLayout::F32_2d(l) => f_2d_32(l),
        OwnedMetricLayout::F64Nd(l) => f_nd_64(l),
        OwnedMetricLayout::F32Nd(l) => f_nd_32(l),
        OwnedMetricLayout::F64Manifold(l) => f_man_64(l),
        OwnedMetricLayout::F32Manifold(l) => f_man_32(l),
    })
    .map_err(crate::ffi_convert::core_error_to_py)
}
