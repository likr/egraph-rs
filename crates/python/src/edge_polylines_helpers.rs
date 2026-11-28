//! Internal Rust helpers and FFI marshaling for `PyEdgePolylines`.

use egraph::core::{EdgePolylines, Layout2d, Layout3d, LayoutNd};
use egraph_ffi::validate_edge_offsets;
use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::edge_polylines::{PolylinePointsArray, PolylineTarget, PyEdgePolylines};
use crate::errors::{
    core_error_to_py, dimension_mismatch_err, ffi_error_to_py, invalid_input_err,
    non_finite_value_err,
};
use crate::ffi_convert::ScalarDtype;
use crate::layout_buffers::{Layout2dInner, PyLayout2d};
use crate::layout_nd::{LayoutNdInner, PyLayoutNd};
use crate::layout3d::{Layout3dInner, PyLayout3d};
use pyo3::exceptions::PyTypeError;

impl PyEdgePolylines {
    pub(crate) fn extract_offsets_vec(&self, py: Python<'_>) -> PyResult<Vec<u32>> {
        let ro = self.offsets_ref().bind(py).readonly();
        Ok(match ro.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => ro.as_array().iter().copied().collect(),
        })
    }

    pub(crate) fn extract_points_f64(&self, py: Python<'_>) -> PyResult<Vec<f64>> {
        match self.points_ref() {
            PolylinePointsArray::F64(arr) => {
                let ro = arr.bind(py).readonly();
                Ok(match ro.as_slice() {
                    Ok(s) => s.to_vec(),
                    Err(_) => ro.as_array().iter().copied().collect(),
                })
            }
            PolylinePointsArray::F32(_) => Err(invalid_input_err(
                "EdgePolylines dtype mismatch: expected float64",
            )),
        }
    }

    pub(crate) fn extract_points_f32(&self, py: Python<'_>) -> PyResult<Vec<f32>> {
        match self.points_ref() {
            PolylinePointsArray::F32(arr) => {
                let ro = arr.bind(py).readonly();
                Ok(match ro.as_slice() {
                    Ok(s) => s.to_vec(),
                    Err(_) => ro.as_array().iter().copied().collect(),
                })
            }
            PolylinePointsArray::F64(_) => Err(invalid_input_err(
                "EdgePolylines dtype mismatch: expected float32",
            )),
        }
    }

    pub(crate) fn to_core_2d_f64(&self, py: Python<'_>) -> PyResult<EdgePolylines<Layout2d<f64>>> {
        let offsets = self.extract_offsets_vec(py)?;
        let flat = self.extract_points_f64(py)?;
        let pts = Layout2d::from_interleaved(&flat).map_err(core_error_to_py)?;
        EdgePolylines::from_parts_vec(offsets, pts).map_err(core_error_to_py)
    }

    pub(crate) fn to_core_2d_f32(&self, py: Python<'_>) -> PyResult<EdgePolylines<Layout2d<f32>>> {
        let offsets = self.extract_offsets_vec(py)?;
        let flat = self.extract_points_f32(py)?;
        let pts = Layout2d::from_interleaved(&flat).map_err(core_error_to_py)?;
        EdgePolylines::from_parts_vec(offsets, pts).map_err(core_error_to_py)
    }

    pub(crate) fn to_core_3d_f64(&self, py: Python<'_>) -> PyResult<EdgePolylines<Layout3d<f64>>> {
        let offsets = self.extract_offsets_vec(py)?;
        let flat = self.extract_points_f64(py)?;
        let pts = Layout3d::from_interleaved(&flat).map_err(core_error_to_py)?;
        EdgePolylines::from_parts_vec(offsets, pts).map_err(core_error_to_py)
    }

    pub(crate) fn to_core_3d_f32(&self, py: Python<'_>) -> PyResult<EdgePolylines<Layout3d<f32>>> {
        let offsets = self.extract_offsets_vec(py)?;
        let flat = self.extract_points_f32(py)?;
        let pts = Layout3d::from_interleaved(&flat).map_err(core_error_to_py)?;
        EdgePolylines::from_parts_vec(offsets, pts).map_err(core_error_to_py)
    }

    pub(crate) fn to_core_nd_f64(&self, py: Python<'_>) -> PyResult<EdgePolylines<LayoutNd<f64>>> {
        let offsets = self.extract_offsets_vec(py)?;
        let flat = self.extract_points_f64(py)?;
        let dims = self.dims_count();
        let k = flat.len().checked_div(dims).unwrap_or(0);
        let pts = LayoutNd::from_flat_vec(k, dims, flat).map_err(core_error_to_py)?;
        EdgePolylines::from_parts_vec(offsets, pts).map_err(core_error_to_py)
    }

    pub(crate) fn to_core_nd_f32(&self, py: Python<'_>) -> PyResult<EdgePolylines<LayoutNd<f32>>> {
        let offsets = self.extract_offsets_vec(py)?;
        let flat = self.extract_points_f32(py)?;
        let dims = self.dims_count();
        let k = flat.len().checked_div(dims).unwrap_or(0);
        let pts = LayoutNd::from_flat_vec(k, dims, flat).map_err(core_error_to_py)?;
        EdgePolylines::from_parts_vec(offsets, pts).map_err(core_error_to_py)
    }

    fn offsets_differ(&self, py: Python<'_>, offsets: &[u32]) -> bool {
        let ro = self.offsets_ref().bind(py).readonly();
        match ro.as_slice() {
            Ok(cur) => cur != offsets,
            Err(_) => ro.as_array().iter().copied().ne(offsets.iter().copied()),
        }
    }

    pub(crate) fn write_back_f64(
        &mut self,
        py: Python<'_>,
        offsets: Vec<u32>,
        flat: Vec<f64>,
        dims: usize,
    ) -> PyResult<()> {
        let k = flat.len().checked_div(dims).unwrap_or(0);
        let offsets_differ = self.offsets_differ(py, &offsets);
        let points_reusable = !offsets_differ
            && match self.points_ref() {
                PolylinePointsArray::F64(arr) => {
                    let bound = arr.bind(py);
                    bound.shape() == [k, dims] && bound.is_c_contiguous()
                }
                PolylinePointsArray::F32(_) => false,
            };
        if !points_reusable || offsets_differ {
            *self.offsets_mut() = PyArray1::from_slice(py, &offsets).unbind();
        }
        if points_reusable && let PolylinePointsArray::F64(arr) = self.points_ref() {
            let mut rw = arr.bind(py).readwrite();
            if let Some(dst) = rw.as_array_mut().as_slice_mut() {
                dst.copy_from_slice(&flat);
                return Ok(());
            }
        }
        let arr = Array2::from_shape_vec((k, dims), flat)
            .map_err(|e| dimension_mismatch_err(e.to_string()))?;
        *self.points_mut() = PolylinePointsArray::F64(PyArray2::from_owned_array(py, arr).unbind());
        self.set_dims(dims);
        Ok(())
    }

    pub(crate) fn write_back_f32(
        &mut self,
        py: Python<'_>,
        offsets: Vec<u32>,
        flat: Vec<f32>,
        dims: usize,
    ) -> PyResult<()> {
        let k = flat.len().checked_div(dims).unwrap_or(0);
        let offsets_differ = self.offsets_differ(py, &offsets);
        let points_reusable = !offsets_differ
            && match self.points_ref() {
                PolylinePointsArray::F32(arr) => {
                    let bound = arr.bind(py);
                    bound.shape() == [k, dims] && bound.is_c_contiguous()
                }
                PolylinePointsArray::F64(_) => false,
            };
        if !points_reusable || offsets_differ {
            *self.offsets_mut() = PyArray1::from_slice(py, &offsets).unbind();
        }
        if points_reusable && let PolylinePointsArray::F32(arr) = self.points_ref() {
            let mut rw = arr.bind(py).readwrite();
            if let Some(dst) = rw.as_array_mut().as_slice_mut() {
                dst.copy_from_slice(&flat);
                return Ok(());
            }
        }
        let arr = Array2::from_shape_vec((k, dims), flat)
            .map_err(|e| dimension_mismatch_err(e.to_string()))?;
        *self.points_mut() = PolylinePointsArray::F32(PyArray2::from_owned_array(py, arr).unbind());
        self.set_dims(dims);
        Ok(())
    }

    pub(crate) fn build_owned_f64(
        py: Python<'_>,
        target: PolylineTarget,
        offsets: Vec<u32>,
        flat: Vec<f64>,
        dims: usize,
    ) -> PyResult<Self> {
        let k = flat.len().checked_div(dims).unwrap_or(0);
        validate_edge_offsets(&offsets, k).map_err(ffi_error_to_py)?;
        if !flat.iter().all(|v| v.is_finite()) {
            return Err(non_finite_value_err(
                "EdgePolylines points contain non-finite value",
            ));
        }
        let arr = Array2::from_shape_vec((k, dims), flat)
            .map_err(|e| dimension_mismatch_err(e.to_string()))?;
        Ok(Self::from_raw_parts(
            target,
            ScalarDtype::F64,
            dims,
            PyArray1::from_vec(py, offsets).unbind(),
            PolylinePointsArray::F64(PyArray2::from_owned_array(py, arr).unbind()),
        ))
    }

    pub(crate) fn build_owned_f32(
        py: Python<'_>,
        target: PolylineTarget,
        offsets: Vec<u32>,
        flat: Vec<f32>,
        dims: usize,
    ) -> PyResult<Self> {
        let k = flat.len().checked_div(dims).unwrap_or(0);
        validate_edge_offsets(&offsets, k).map_err(ffi_error_to_py)?;
        if !flat.iter().all(|v| v.is_finite()) {
            return Err(non_finite_value_err(
                "EdgePolylines points contain non-finite value",
            ));
        }
        let arr = Array2::from_shape_vec((k, dims), flat)
            .map_err(|e| dimension_mismatch_err(e.to_string()))?;
        Ok(Self::from_raw_parts(
            target,
            ScalarDtype::F32,
            dims,
            PyArray1::from_vec(py, offsets).unbind(),
            PolylinePointsArray::F32(PyArray2::from_owned_array(py, arr).unbind()),
        ))
    }

    pub(crate) fn straight_line_impl(
        py: Python<'_>,
        graph: &PyCsrGraph,
        layout: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let g = graph.inner();
        if let Ok(l2) = layout.cast::<PyLayout2d>() {
            let borrowed = l2.borrow();
            return match borrowed.inner() {
                Layout2dInner::F64(buf) => {
                    let soa = buf.to_layout2d();
                    let poly = py
                        .detach(|| EdgePolylines::straight_line_from_graph(&*g, &soa))
                        .map_err(core_error_to_py)?;
                    let (offs, pts) = poly.into_parts();
                    Self::build_owned_f64(
                        py,
                        PolylineTarget::Layout2d,
                        offs,
                        pts.to_interleaved(),
                        2,
                    )
                }
                Layout2dInner::F32(buf) => {
                    let soa = buf.to_layout2d();
                    let poly = py
                        .detach(|| EdgePolylines::straight_line_from_graph(&*g, &soa))
                        .map_err(core_error_to_py)?;
                    let (offs, pts) = poly.into_parts();
                    Self::build_owned_f32(
                        py,
                        PolylineTarget::Layout2d,
                        offs,
                        pts.to_interleaved(),
                        2,
                    )
                }
            };
        }
        if let Ok(l3) = layout.cast::<PyLayout3d>() {
            let borrowed = l3.borrow();
            return match borrowed.inner() {
                Layout3dInner::F64(buf) => {
                    let soa = buf.to_layout3d();
                    let poly = py
                        .detach(|| EdgePolylines::straight_line_from_graph(&*g, &soa))
                        .map_err(core_error_to_py)?;
                    let (offs, pts) = poly.into_parts();
                    Self::build_owned_f64(
                        py,
                        PolylineTarget::Layout3d,
                        offs,
                        pts.to_interleaved(),
                        3,
                    )
                }
                Layout3dInner::F32(buf) => {
                    let soa = buf.to_layout3d();
                    let poly = py
                        .detach(|| EdgePolylines::straight_line_from_graph(&*g, &soa))
                        .map_err(core_error_to_py)?;
                    let (offs, pts) = poly.into_parts();
                    Self::build_owned_f32(
                        py,
                        PolylineTarget::Layout3d,
                        offs,
                        pts.to_interleaved(),
                        3,
                    )
                }
            };
        }
        if let Ok(lnd) = layout.cast::<PyLayoutNd>() {
            let borrowed = lnd.borrow();
            return match borrowed.inner() {
                LayoutNdInner::F64(nd) => {
                    let dims = nd.dims();
                    let poly = py
                        .detach(|| EdgePolylines::straight_line_from_graph(&*g, nd))
                        .map_err(core_error_to_py)?;
                    let (offs, pts) = poly.into_parts();
                    Self::build_owned_f64(
                        py,
                        PolylineTarget::LayoutNd,
                        offs,
                        pts.coords().to_vec(),
                        dims,
                    )
                }
                LayoutNdInner::F32(nd) => {
                    let dims = nd.dims();
                    let poly = py
                        .detach(|| EdgePolylines::straight_line_from_graph(&*g, nd))
                        .map_err(core_error_to_py)?;
                    let (offs, pts) = poly.into_parts();
                    Self::build_owned_f32(
                        py,
                        PolylineTarget::LayoutNd,
                        offs,
                        pts.coords().to_vec(),
                        dims,
                    )
                }
            };
        }
        Err(PyTypeError::new_err(
            "layout must be a Layout2d, Layout3d, or LayoutNd instance",
        ))
    }

    pub(crate) fn with_subdivisions_impl(&self, py: Python<'_>, k_sub: usize) -> PyResult<Self> {
        match (self.target_kind(), self.dtype_kind()) {
            (PolylineTarget::Layout2d, ScalarDtype::F64) => {
                let core = self.to_core_2d_f64(py)?;
                let sub = py
                    .detach(|| core.with_subdivisions(k_sub))
                    .map_err(core_error_to_py)?;
                let (offs, pts) = sub.into_parts();
                Self::build_owned_f64(py, PolylineTarget::Layout2d, offs, pts.to_interleaved(), 2)
            }
            (PolylineTarget::Layout2d, ScalarDtype::F32) => {
                let core = self.to_core_2d_f32(py)?;
                let sub = py
                    .detach(|| core.with_subdivisions(k_sub))
                    .map_err(core_error_to_py)?;
                let (offs, pts) = sub.into_parts();
                Self::build_owned_f32(py, PolylineTarget::Layout2d, offs, pts.to_interleaved(), 2)
            }
            (PolylineTarget::Layout3d, ScalarDtype::F64) => {
                let core = self.to_core_3d_f64(py)?;
                let sub = py
                    .detach(|| core.with_subdivisions(k_sub))
                    .map_err(core_error_to_py)?;
                let (offs, pts) = sub.into_parts();
                Self::build_owned_f64(py, PolylineTarget::Layout3d, offs, pts.to_interleaved(), 3)
            }
            (PolylineTarget::Layout3d, ScalarDtype::F32) => {
                let core = self.to_core_3d_f32(py)?;
                let sub = py
                    .detach(|| core.with_subdivisions(k_sub))
                    .map_err(core_error_to_py)?;
                let (offs, pts) = sub.into_parts();
                Self::build_owned_f32(py, PolylineTarget::Layout3d, offs, pts.to_interleaved(), 3)
            }
            (PolylineTarget::LayoutNd, ScalarDtype::F64) => {
                let core = self.to_core_nd_f64(py)?;
                let sub = py
                    .detach(|| core.with_subdivisions(k_sub))
                    .map_err(core_error_to_py)?;
                let (offs, pts) = sub.into_parts();
                let dims = pts.dims();
                Self::build_owned_f64(
                    py,
                    PolylineTarget::LayoutNd,
                    offs,
                    pts.coords().to_vec(),
                    dims,
                )
            }
            (PolylineTarget::LayoutNd, ScalarDtype::F32) => {
                let core = self.to_core_nd_f32(py)?;
                let sub = py
                    .detach(|| core.with_subdivisions(k_sub))
                    .map_err(core_error_to_py)?;
                let (offs, pts) = sub.into_parts();
                let dims = pts.dims();
                Self::build_owned_f32(
                    py,
                    PolylineTarget::LayoutNd,
                    offs,
                    pts.coords().to_vec(),
                    dims,
                )
            }
        }
    }
}
