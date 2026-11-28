//! Edge-list extraction and dual-precision (`f64`/`f32`) `CsrGraph` helpers.

use std::borrow::Cow;

use egraph::core::CsrGraph;
use egraph_ffi::{build_csr_f32, build_csr_f64, build_edges_f32, build_edges_f64};
use numpy::{PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::errors::{dimension_mismatch_err, invalid_input_err};

use crate::ffi_convert::{
    ScalarDtype, extract_u32_item, ffi_error_to_py, is_non_sequence_container,
};
use crate::solver_common::vec_f64_to_f32_checked;

pub(crate) enum CsrGraphInner {
    F64(CsrGraph<f64>),
    F32(CsrGraph<f32>),
}

impl Clone for CsrGraphInner {
    fn clone(&self) -> Self {
        match self {
            Self::F64(g) => Self::F64(g.clone()),
            Self::F32(g) => Self::F32(g.clone()),
        }
    }
}

impl CsrGraphInner {
    pub(crate) fn as_f64_cow(&self) -> Cow<'_, CsrGraph<f64>> {
        match self {
            Self::F64(g) => Cow::Borrowed(g),
            Self::F32(g) => {
                let weights_f64: Option<Vec<f64>> = g
                    .weights()
                    .map(|ws| ws.iter().map(|&v| f64::from(v)).collect());
                let converted = CsrGraph::from_csr(
                    g.indptr(),
                    g.indices(),
                    weights_f64.as_deref(),
                    g.is_directed(),
                    false,
                )
                .expect("valid CsrGraph<f32> converts losslessly to CsrGraph<f64>");
                Cow::Owned(converted)
            }
        }
    }

    pub(crate) fn weight_dtype(&self) -> ScalarDtype {
        match self {
            Self::F64(_) => ScalarDtype::F64,
            Self::F32(_) => ScalarDtype::F32,
        }
    }
}

pub(crate) fn build_csr_inner(
    indptr: &[u32],
    indices: &[u32],
    weights: Option<&[f64]>,
    directed: bool,
    sanitize: bool,
    weight_dtype: ScalarDtype,
) -> PyResult<CsrGraphInner> {
    match weight_dtype {
        ScalarDtype::F64 => build_csr_f64(indptr, indices, weights, directed, sanitize)
            .map(CsrGraphInner::F64)
            .map_err(ffi_error_to_py),
        ScalarDtype::F32 => {
            let w32 = weights
                .map(|ws| vec_f64_to_f32_checked(ws, "weights"))
                .transpose()?;
            build_csr_f32(indptr, indices, w32.as_deref(), directed, sanitize)
                .map(CsrGraphInner::F32)
                .map_err(ffi_error_to_py)
        }
    }
}

pub(crate) fn build_edges_inner(
    num_nodes: usize,
    flat_edges: &[u32],
    weights: Option<&[f64]>,
    directed: bool,
    sanitize: bool,
    weight_dtype: ScalarDtype,
) -> PyResult<CsrGraphInner> {
    match weight_dtype {
        ScalarDtype::F64 => build_edges_f64(num_nodes, flat_edges, weights, directed, sanitize)
            .map(CsrGraphInner::F64)
            .map_err(ffi_error_to_py),
        ScalarDtype::F32 => {
            let w32 = weights
                .map(|ws| vec_f64_to_f32_checked(ws, "weights"))
                .transpose()?;
            build_edges_f32(num_nodes, flat_edges, w32.as_deref(), directed, sanitize)
                .map(CsrGraphInner::F32)
                .map_err(ffi_error_to_py)
        }
    }
}

fn convert_edges_2d_to_flat<T, F>(
    view: numpy::ndarray::ArrayView2<'_, T>,
    mut convert: F,
) -> PyResult<Vec<u32>>
where
    T: Copy,
    F: FnMut(T) -> PyResult<u32>,
{
    let [m, cols] = [view.nrows(), view.ncols()];
    if cols != 2 {
        return Err(dimension_mismatch_err(format!(
            "edges must have shape (M, 2), got ({m}, {cols})"
        )));
    }
    let mut out = Vec::with_capacity(m * 2);
    for row in view.rows() {
        out.push(convert(row[0])?);
        out.push(convert(row[1])?);
    }
    Ok(out)
}

pub(crate) fn extract_flat_edges(edges: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    if let Ok(untyped) = edges.cast::<PyUntypedArray>() {
        if untyped.ndim() != 2 {
            return Err(dimension_mismatch_err(format!(
                "edges must be a 2D array of shape (M, 2), got {}D",
                untyped.ndim()
            )));
        }
        let shape = untyped.shape();
        if shape[1] != 2 {
            return Err(dimension_mismatch_err(format!(
                "edges must have shape (M, 2), got ({}, {})",
                shape[0], shape[1]
            )));
        }
        if let Ok(arr) = edges.extract::<PyReadonlyArray2<'_, u32>>() {
            return convert_edges_2d_to_flat(arr.as_array(), Ok);
        }
        if let Ok(arr) = edges.extract::<PyReadonlyArray2<'_, i64>>() {
            return convert_edges_2d_to_flat(arr.as_array(), |v| {
                u32::try_from(v).map_err(|_| {
                    invalid_input_err(format!("edge endpoint {v} out of range for u32"))
                })
            });
        }
        if let Ok(arr) = edges.extract::<PyReadonlyArray2<'_, u64>>() {
            return convert_edges_2d_to_flat(arr.as_array(), |v| {
                u32::try_from(v).map_err(|_| {
                    invalid_input_err(format!("edge endpoint {v} out of range for u32"))
                })
            });
        }
        if let Ok(arr) = edges.extract::<PyReadonlyArray2<'_, i32>>() {
            return convert_edges_2d_to_flat(arr.as_array(), |v| {
                u32::try_from(v).map_err(|_| {
                    invalid_input_err(format!("edge endpoint {v} out of range for u32"))
                })
            });
        }
        if let Ok(arr) = edges.extract::<PyReadonlyArray2<'_, usize>>() {
            return convert_edges_2d_to_flat(arr.as_array(), |v| {
                u32::try_from(v).map_err(|_| {
                    invalid_input_err(format!("edge endpoint {v} out of range for u32"))
                })
            });
        }
        if let Ok(arr) = edges.extract::<PyReadonlyArray2<'_, isize>>() {
            return convert_edges_2d_to_flat(arr.as_array(), |v| {
                u32::try_from(v).map_err(|_| {
                    invalid_input_err(format!("edge endpoint {v} out of range for u32"))
                })
            });
        }
        if let Ok(arr) = edges.extract::<PyReadonlyArray2<'_, u16>>() {
            return convert_edges_2d_to_flat(arr.as_array(), |v| Ok(u32::from(v)));
        }
        if let Ok(arr) = edges.extract::<PyReadonlyArray2<'_, u8>>() {
            return convert_edges_2d_to_flat(arr.as_array(), |v| Ok(u32::from(v)));
        }
        if let Ok(arr) = edges.extract::<PyReadonlyArray2<'_, i16>>() {
            return convert_edges_2d_to_flat(arr.as_array(), |v| {
                u32::try_from(v).map_err(|_| {
                    invalid_input_err(format!("edge endpoint {v} out of range for u32"))
                })
            });
        }
        if let Ok(arr) = edges.extract::<PyReadonlyArray2<'_, i8>>() {
            return convert_edges_2d_to_flat(arr.as_array(), |v| {
                u32::try_from(v).map_err(|_| {
                    invalid_input_err(format!("edge endpoint {v} out of range for u32"))
                })
            });
        }
        return Err(PyTypeError::new_err(
            "edges must be a 2D integer NumPy array of shape (M, 2)",
        ));
    }

    if is_non_sequence_container(edges) {
        return Err(PyTypeError::new_err(
            "edges must be a 2D integer array or sequence of (u, v) pairs",
        ));
    }
    let outer = edges.try_iter().map_err(|_| {
        PyTypeError::new_err("edges must be a 2D integer array or sequence of (u, v) pairs")
    })?;
    let mut flat = Vec::new();
    for pair_res in outer {
        let pair = pair_res?;
        if is_non_sequence_container(&pair) {
            return Err(PyTypeError::new_err("each edge must be a 2-tuple (u, v)"));
        }
        let inner = pair
            .try_iter()
            .map_err(|_| PyTypeError::new_err("each edge must be a 2-element sequence (u, v)"))?;
        let mut endpoints = [0u32; 2];
        let mut count = 0usize;
        for elem_res in inner {
            let elem = elem_res?;
            if count < 2 {
                endpoints[count] = extract_u32_item(&elem, "edges")?;
            }
            count += 1;
        }
        if count != 2 {
            return Err(dimension_mismatch_err(format!(
                "each edge must contain exactly 2 endpoints, got {count}"
            )));
        }
        flat.push(endpoints[0]);
        flat.push(endpoints[1]);
    }
    Ok(flat)
}
