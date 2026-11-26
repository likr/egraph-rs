//! Shared JavaScript/Rust primitive conversion and CSR edge-weight helpers.

use egraph::core::{CoreError, CsrGraph};
use egraph_ffi::FfiError;
use wasm_bindgen::{JsCast, prelude::*};

const DEFAULT_SEED: u64 = 42;
// IEEE-754 double precision represents integers exactly up to 2^53 - 1 (`Number.MAX_SAFE_INTEGER`),
// allowing JS callers to pass numeric seeds like `42` without requiring `BigInt`.
const MAX_SAFE_INTEGER_F64: f64 = 9_007_199_254_740_991.0;

pub(crate) fn ffi_err_to_js(err: FfiError) -> JsError {
    JsError::new(&err.to_string())
}

pub(crate) fn core_err_to_js(err: CoreError) -> JsError {
    JsError::new(&err.to_string())
}

pub(crate) fn parse_non_negative_usize(value: f64, name: &str) -> Result<usize, JsError> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > (u32::MAX as f64) {
        return Err(JsError::new(&format!(
            "{name} must be a non-negative integer <= u32::MAX, got {value}"
        )));
    }
    Ok(value as usize)
}

pub(crate) fn parse_non_negative_u32(value: f64, name: &str) -> Result<u32, JsError> {
    let v = parse_non_negative_usize(value, name)?;
    u32::try_from(v).map_err(|_| JsError::new(&format!("{name} must be <= u32::MAX, got {value}")))
}

pub(crate) fn parse_optional_usize(
    value: Option<f64>,
    default: usize,
    name: &str,
) -> Result<usize, JsError> {
    match value {
        None => Ok(default),
        Some(v) => parse_non_negative_usize(v, name),
    }
}

pub(crate) fn parse_opt_usize_param(
    value: Option<f64>,
    name: &str,
) -> Result<Option<usize>, JsError> {
    value.map(|v| parse_non_negative_usize(v, name)).transpose()
}

pub(crate) fn parse_u32_sequence(val: &JsValue, name: &str) -> Result<Vec<u32>, JsError> {
    if let Some(arr) = val.dyn_ref::<js_sys::Uint32Array>() {
        return Ok(arr.to_vec());
    }
    if val.is_null() || val.is_undefined() || val.is_string() || !val.is_object() {
        return Err(JsError::new(&format!(
            "{name} must be a Uint32Array or integer array"
        )));
    }
    let iter = js_sys::try_iter(val)
        .map_err(|err| JsError::new(&format!("failed to iterate {name}: {err:?}")))?
        .ok_or_else(|| JsError::new(&format!("{name} must be a Uint32Array or integer array")))?;
    let mut out = Vec::new();
    for item_res in iter {
        let item =
            item_res.map_err(|err| JsError::new(&format!("failed to read {name}: {err:?}")))?;
        let num = item
            .as_f64()
            .ok_or_else(|| JsError::new(&format!("{name} entries must be integers")))?;
        out.push(parse_non_negative_u32(num, name)?);
    }
    Ok(out)
}

pub(crate) fn parse_optional_f64_sequence(
    val: &JsValue,
    name: &str,
) -> Result<Option<Vec<f64>>, JsError> {
    if val.is_null() || val.is_undefined() {
        return Ok(None);
    }
    if let Some(arr) = val.dyn_ref::<js_sys::Float64Array>() {
        return Ok(Some(arr.to_vec()));
    }
    if let Some(arr) = val.dyn_ref::<js_sys::Float32Array>() {
        return Ok(Some(arr.to_vec().into_iter().map(f64::from).collect()));
    }
    if val.is_string() || !val.is_object() {
        return Err(JsError::new(&format!(
            "{name} must be a Float64Array, Float32Array, or numeric array"
        )));
    }
    let iter = js_sys::try_iter(val)
        .map_err(|err| JsError::new(&format!("failed to iterate {name}: {err:?}")))?
        .ok_or_else(|| {
            JsError::new(&format!(
                "{name} must be a Float64Array, Float32Array, or numeric array"
            ))
        })?;
    let mut out = Vec::new();
    for item_res in iter {
        let item =
            item_res.map_err(|err| JsError::new(&format!("failed to read {name}: {err:?}")))?;
        let num = item
            .as_f64()
            .ok_or_else(|| JsError::new(&format!("{name} entries must be numbers")))?;
        out.push(num);
    }
    Ok(Some(out))
}

pub(crate) fn parse_f64_sequence(val: &JsValue, name: &str) -> Result<Vec<f64>, JsError> {
    if let Some(arr) = val.dyn_ref::<js_sys::Float64Array>() {
        return Ok(arr.to_vec());
    }
    if let Some(arr) = val.dyn_ref::<js_sys::Float32Array>() {
        return Ok(arr.to_vec().into_iter().map(f64::from).collect());
    }
    if val.is_null() || val.is_undefined() || val.is_string() || !val.is_object() {
        return Err(JsError::new(&format!(
            "{name} must be a Float64Array, Float32Array, or numeric iterable"
        )));
    }
    let iter = js_sys::try_iter(val)
        .map_err(|err| JsError::new(&format!("failed to iterate {name}: {err:?}")))?
        .ok_or_else(|| {
            JsError::new(&format!(
                "{name} must be a Float64Array, Float32Array, or numeric iterable"
            ))
        })?;
    let mut out = Vec::new();
    for item_res in iter {
        let item =
            item_res.map_err(|err| JsError::new(&format!("failed to read {name}: {err:?}")))?;
        let num = item
            .as_f64()
            .ok_or_else(|| JsError::new(&format!("{name} entries must be numbers")))?;
        out.push(num);
    }
    Ok(out)
}

pub(crate) fn parse_seed_u64(seed: Option<f64>) -> Result<u64, JsError> {
    match seed {
        None => Ok(DEFAULT_SEED),
        Some(value) => {
            if !value.is_finite()
                || value < 0.0
                || value.fract() != 0.0
                || value > MAX_SAFE_INTEGER_F64
            {
                return Err(JsError::new(&format!(
                    "seed must be a non-negative safe integer <= 9007199254740991, got {value}"
                )));
            }
            Ok(value as u64)
        }
    }
}

pub(crate) fn csr_f64_to_f32(graph: &CsrGraph<f64>) -> Result<CsrGraph<f32>, JsError> {
    let weights_f32: Option<Vec<f32>> = graph
        .weights()
        .map(|ws| {
            ws.iter()
                .map(|&w| {
                    let wf = w as f32;
                    if !wf.is_finite() {
                        return Err(JsError::new("coarsened edge weight overflowed f32"));
                    }
                    Ok(wf)
                })
                .collect::<Result<Vec<f32>, JsError>>()
        })
        .transpose()?;
    CsrGraph::<f32>::from_csr(
        graph.indptr(),
        graph.indices(),
        weights_f32.as_deref(),
        graph.is_directed(),
        false,
    )
    .map_err(core_err_to_js)
}
