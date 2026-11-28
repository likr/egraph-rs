//! UAF-safe owned NumPy polyline buffer (`EdgePolylines`) across `Layout2d`, `Layout3d`, and `LayoutNd`.

use numpy::ndarray::Array2;
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::array_extract::extract_f64_2d;
use crate::csr_graph::PyCsrGraph;
use crate::errors::{dimension_mismatch_err, index_out_of_bounds_err, unsupported_operation_err};
use crate::ffi_convert::{ScalarDtype, StrictUsize, extract_u32_1d, normalize_enum_key};
use crate::layout_buffers::{Layout2dInner, PyLayout2d};
use crate::layout_nd::{LayoutNdInner, PyLayoutNd};
use crate::layout3d::{Layout3dInner, PyLayout3d};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PolylineTarget {
    Layout2d,
    Layout3d,
    LayoutNd,
}

impl PolylineTarget {
    pub(crate) fn from_str(raw: &str) -> PyResult<Self> {
        match normalize_enum_key(raw).as_str() {
            "layout2d" => Ok(Self::Layout2d),
            "layout3d" => Ok(Self::Layout3d),
            "layoutnd" => Ok(Self::LayoutNd),
            _ => Err(unsupported_operation_err(format!(
                "unsupported EdgePolylines target {raw:?}; expected 'layout2d', 'layout3d', or 'layout_nd'"
            ))),
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Layout2d => "layout2d",
            Self::Layout3d => "layout3d",
            Self::LayoutNd => "layout_nd",
        }
    }
}

pub(crate) enum PolylinePointsArray {
    F64(Py<PyArray2<f64>>),
    F32(Py<PyArray2<f32>>),
}

/// Owned, UAF-safe edge polyline buffer (`edge_offsets` + 2D control-point array).
#[pyclass(name = "EdgePolylines", module = "egraph._core")]
pub struct PyEdgePolylines {
    target: PolylineTarget,
    dtype: ScalarDtype,
    dims: usize,
    edge_offsets: Py<PyArray1<u32>>,
    points: PolylinePointsArray,
}

impl PyEdgePolylines {
    pub(crate) fn from_raw_parts(
        target: PolylineTarget,
        dtype: ScalarDtype,
        dims: usize,
        edge_offsets: Py<PyArray1<u32>>,
        points: PolylinePointsArray,
    ) -> Self {
        Self {
            target,
            dtype,
            dims,
            edge_offsets,
            points,
        }
    }

    pub(crate) fn target_kind(&self) -> PolylineTarget {
        self.target
    }

    pub(crate) fn dtype_kind(&self) -> ScalarDtype {
        self.dtype
    }

    pub(crate) fn dims_count(&self) -> usize {
        self.dims
    }

    pub(crate) fn set_dims(&mut self, dims: usize) {
        self.dims = dims;
    }

    pub(crate) fn offsets_ref(&self) -> &Py<PyArray1<u32>> {
        &self.edge_offsets
    }

    pub(crate) fn offsets_mut(&mut self) -> &mut Py<PyArray1<u32>> {
        &mut self.edge_offsets
    }

    pub(crate) fn points_ref(&self) -> &PolylinePointsArray {
        &self.points
    }

    pub(crate) fn points_mut(&mut self) -> &mut PolylinePointsArray {
        &mut self.points
    }
}

#[pymethods]
impl PyEdgePolylines {
    /// Initialize 2-point straight-line edge polylines from `graph` and node positions `layout`.
    #[staticmethod]
    #[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
    fn straight_line(
        py: Python<'_>,
        graph: &PyCsrGraph,
        layout: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Self::straight_line_impl(py, graph, layout)
    }

    /// Construct `EdgePolylines` from `edge_offsets` and control-point coordinates `points`.
    #[staticmethod]
    #[pyo3(
        signature = (edge_offsets, points, *, target = None),
        text_signature = "(edge_offsets, points, *, target=None)"
    )]
    fn from_parts(
        py: Python<'_>,
        edge_offsets: &Bound<'_, PyAny>,
        points: &Bound<'_, PyAny>,
        target: Option<&str>,
    ) -> PyResult<Self> {
        let offsets = extract_u32_1d(edge_offsets, "edge_offsets")?;
        let requested_target = target.map(PolylineTarget::from_str).transpose()?;
        if let Ok(l2) = points.cast::<PyLayout2d>() {
            if let Some(t) = requested_target
                && t != PolylineTarget::Layout2d
            {
                return Err(dimension_mismatch_err(format!(
                    "Layout2d points require target='layout2d', got '{}'",
                    t.as_str()
                )));
            }
            return match l2.borrow().inner() {
                Layout2dInner::F64(buf) => Self::build_owned_f64(
                    py,
                    PolylineTarget::Layout2d,
                    offsets,
                    buf.as_slice().to_vec(),
                    2,
                ),
                Layout2dInner::F32(buf) => Self::build_owned_f32(
                    py,
                    PolylineTarget::Layout2d,
                    offsets,
                    buf.as_slice().to_vec(),
                    2,
                ),
            };
        }
        if let Ok(l3) = points.cast::<PyLayout3d>() {
            if let Some(t) = requested_target
                && t != PolylineTarget::Layout3d
            {
                return Err(dimension_mismatch_err(format!(
                    "Layout3d points require target='layout3d', got '{}'",
                    t.as_str()
                )));
            }
            return match l3.borrow().inner() {
                Layout3dInner::F64(buf) => Self::build_owned_f64(
                    py,
                    PolylineTarget::Layout3d,
                    offsets,
                    buf.as_slice().to_vec(),
                    3,
                ),
                Layout3dInner::F32(buf) => Self::build_owned_f32(
                    py,
                    PolylineTarget::Layout3d,
                    offsets,
                    buf.as_slice().to_vec(),
                    3,
                ),
            };
        }
        if let Ok(lnd) = points.cast::<PyLayoutNd>() {
            if let Some(t) = requested_target
                && t != PolylineTarget::LayoutNd
            {
                return Err(dimension_mismatch_err(format!(
                    "LayoutNd points require target='layout_nd', got '{}'",
                    t.as_str()
                )));
            }
            return match lnd.borrow().inner() {
                LayoutNdInner::F64(nd) => Self::build_owned_f64(
                    py,
                    PolylineTarget::LayoutNd,
                    offsets,
                    nd.coords().to_vec(),
                    nd.dims(),
                ),
                LayoutNdInner::F32(nd) => Self::build_owned_f32(
                    py,
                    PolylineTarget::LayoutNd,
                    offsets,
                    nd.coords().to_vec(),
                    nd.dims(),
                ),
            };
        }

        if let Ok(untyped) = points.cast::<PyUntypedArray>() {
            if untyped.ndim() != 2 {
                return Err(dimension_mismatch_err(format!(
                    "points must be a 2D array, got {}D",
                    untyped.ndim()
                )));
            }
            let dims = untyped.shape()[1];
            let resolved_target = match requested_target {
                Some(PolylineTarget::Layout2d) if dims != 2 => {
                    return Err(dimension_mismatch_err(format!(
                        "target='layout2d' requires points with 2 columns, got {dims}"
                    )));
                }
                Some(PolylineTarget::Layout3d) if dims != 3 => {
                    return Err(dimension_mismatch_err(format!(
                        "target='layout3d' requires points with 3 columns, got {dims}"
                    )));
                }
                Some(t) => t,
                None => match dims {
                    2 => PolylineTarget::Layout2d,
                    3 => PolylineTarget::Layout3d,
                    _ => PolylineTarget::LayoutNd,
                },
            };
            if let Ok(arr) = points.extract::<PyReadonlyArray2<'_, f32>>() {
                let flat: Vec<f32> = arr.as_array().iter().copied().collect();
                return Self::build_owned_f32(py, resolved_target, offsets, flat, dims);
            }
            if let Ok(arr) = points.extract::<PyReadonlyArray2<'_, f64>>() {
                let flat: Vec<f64> = arr.as_array().iter().copied().collect();
                return Self::build_owned_f64(py, resolved_target, offsets, flat, dims);
            }
            return Err(PyTypeError::new_err(
                "points NumPy array must have dtype float64 or float32",
            ));
        }

        let (flat, _, dims) = extract_f64_2d(points, "points")?;
        let resolved_target = match requested_target {
            Some(PolylineTarget::Layout2d) if dims != 2 => {
                return Err(dimension_mismatch_err(format!(
                    "target='layout2d' requires points with 2 columns, got {dims}"
                )));
            }
            Some(PolylineTarget::Layout3d) if dims != 3 => {
                return Err(dimension_mismatch_err(format!(
                    "target='layout3d' requires points with 3 columns, got {dims}"
                )));
            }
            Some(t) => t,
            None => match dims {
                2 => PolylineTarget::Layout2d,
                3 => PolylineTarget::Layout3d,
                _ => PolylineTarget::LayoutNd,
            },
        };
        Self::build_owned_f64(py, resolved_target, offsets, flat, dims)
    }

    /// Return a new `EdgePolylines` with `interior_points_per_segment` added per segment.
    #[pyo3(signature = (interior_points_per_segment), text_signature = "($self, interior_points_per_segment)")]
    fn with_subdivisions(
        &self,
        py: Python<'_>,
        interior_points_per_segment: StrictUsize,
    ) -> PyResult<Self> {
        self.with_subdivisions_impl(py, interior_points_per_segment.0)
    }

    #[getter]
    fn target(&self) -> &'static str {
        self.target.as_str()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.dtype.as_str()
    }

    #[getter]
    fn dims(&self) -> usize {
        self.dims
    }

    #[getter]
    fn num_edges(&self, py: Python<'_>) -> usize {
        self.edge_offsets.bind(py).len().saturating_sub(1)
    }

    #[getter]
    fn total_points(&self, py: Python<'_>) -> usize {
        match &self.points {
            PolylinePointsArray::F64(arr) => arr.bind(py).shape()[0],
            PolylinePointsArray::F32(arr) => arr.bind(py).shape()[0],
        }
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        self.num_edges(py)
    }

    fn is_empty(&self, py: Python<'_>) -> bool {
        self.num_edges(py) == 0
    }

    #[pyo3(signature = (edge), text_signature = "($self, edge)")]
    fn edge_point_range(&self, py: Python<'_>, edge: StrictUsize) -> PyResult<(usize, usize)> {
        let offsets = self.extract_offsets_vec(py)?;
        let num_edges = offsets.len().saturating_sub(1);
        if edge.0 >= num_edges {
            return Err(index_out_of_bounds_err(format!(
                "edge index {} out of bounds for {num_edges} edges",
                edge.0
            )));
        }
        Ok((offsets[edge.0] as usize, offsets[edge.0 + 1] as usize))
    }

    /// Return the 1D `u32` edge offsets array (`length = num_edges + 1`).
    #[pyo3(signature = (copy = false), text_signature = "($self, copy=False)")]
    fn edge_offsets<'py>(
        &self,
        py: Python<'py>,
        copy: bool,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let bound = self.edge_offsets.bind(py);
        if copy {
            let offs = self.extract_offsets_vec(py)?;
            return Ok(PyArray1::from_vec(py, offs));
        }
        Ok(bound.clone())
    }

    /// Return the 2D `(total_points, dims)` control-point NumPy array.
    #[pyo3(signature = (copy = false), text_signature = "($self, copy=False)")]
    fn points<'py>(&self, py: Python<'py>, copy: bool) -> PyResult<Bound<'py, PyAny>> {
        match &self.points {
            PolylinePointsArray::F64(arr) => {
                let bound = arr.bind(py);
                if copy {
                    let flat = self.extract_points_f64(py)?;
                    let k = bound.shape()[0];
                    let owned = Array2::from_shape_vec((k, self.dims), flat)
                        .map_err(|e| dimension_mismatch_err(e.to_string()))?;
                    Ok(PyArray2::from_owned_array(py, owned).into_any())
                } else {
                    Ok(bound.clone().into_any())
                }
            }
            PolylinePointsArray::F32(arr) => {
                let bound = arr.bind(py);
                if copy {
                    let flat = self.extract_points_f32(py)?;
                    let k = bound.shape()[0];
                    let owned = Array2::from_shape_vec((k, self.dims), flat)
                        .map_err(|e| dimension_mismatch_err(e.to_string()))?;
                    Ok(PyArray2::from_owned_array(py, owned).into_any())
                } else {
                    Ok(bound.clone().into_any())
                }
            }
        }
    }

    /// Extract coordinate column `axis` across all control points.
    #[pyo3(signature = (axis), text_signature = "($self, axis)")]
    fn points_axis<'py>(&self, py: Python<'py>, axis: StrictUsize) -> PyResult<Bound<'py, PyAny>> {
        if axis.0 >= self.dims {
            return Err(index_out_of_bounds_err(format!(
                "axis {} out of bounds for {}D polylines",
                axis.0, self.dims
            )));
        }
        match &self.points {
            PolylinePointsArray::F64(arr) => {
                let ro = arr.bind(py).readonly();
                let col: Vec<f64> = ro.as_array().column(axis.0).iter().copied().collect();
                Ok(PyArray1::from_vec(py, col).into_any())
            }
            PolylinePointsArray::F32(arr) => {
                let ro = arr.bind(py).readonly();
                let col: Vec<f32> = ro.as_array().column(axis.0).iter().copied().collect();
                Ok(PyArray1::from_vec(py, col).into_any())
            }
        }
    }

    /// Extract the `z` coordinate (`axis=2`) across all control points.
    #[pyo3(signature = (), text_signature = "($self)")]
    fn points_z<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.dims < 3 {
            return Err(dimension_mismatch_err(format!(
                "points_z requires at least 3 dimensions, got {}",
                self.dims
            )));
        }
        self.points_axis(py, StrictUsize(2))
    }

    fn is_all_finite(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(match &self.points {
            PolylinePointsArray::F64(arr) => arr
                .bind(py)
                .readonly()
                .as_array()
                .iter()
                .all(|v| v.is_finite()),
            PolylinePointsArray::F32(arr) => arr
                .bind(py)
                .readonly()
                .as_array()
                .iter()
                .all(|v| v.is_finite()),
        })
    }

    fn copy(&self, py: Python<'_>) -> PyResult<Self> {
        let offsets = self.extract_offsets_vec(py)?;
        match self.dtype {
            ScalarDtype::F64 => {
                let flat = self.extract_points_f64(py)?;
                Self::build_owned_f64(py, self.target, offsets, flat, self.dims)
            }
            ScalarDtype::F32 => {
                let flat = self.extract_points_f32(py)?;
                Self::build_owned_f32(py, self.target, offsets, flat, self.dims)
            }
        }
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "EdgePolylines(num_edges={}, total_points={}, dims={}, target='{}', dtype='{}')",
            self.num_edges(py),
            self.total_points(py),
            self.dims,
            self.target.as_str(),
            self.dtype.as_str()
        )
    }
}

/// Register `EdgePolylines` on `egraph._core`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEdgePolylines>()?;
    Ok(())
}
