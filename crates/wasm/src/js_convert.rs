//! Shared JavaScript/Rust primitive conversion, structured error helpers, and CSR edge-weight helpers.

use egraph::core::{CoreError, CsrGraph};
use egraph_ffi::{ErrorKind, FfiError};
use wasm_bindgen::{JsCast, prelude::*};

pub(crate) const DEFAULT_SEED: u64 = 42;
// IEEE-754 double precision represents integers exactly up to 2^53 - 1 (`Number.MAX_SAFE_INTEGER`),
// allowing JS callers to pass numeric seeds like `42` without requiring `BigInt`.
const MAX_SAFE_INTEGER_F64: f64 = 9_007_199_254_740_991.0;

pub(crate) fn error_with_code(code: &str, msg: &str) -> JsValue {
    let err = js_sys::Error::new(msg);
    let _ = js_sys::Reflect::set(&err, &JsValue::from_str("code"), &JsValue::from_str(code));
    err.into()
}

pub(crate) fn ffi_err_to_js(err: FfiError) -> JsValue {
    error_with_code(err.kind.code_str(), &err.to_string())
}

pub(crate) fn core_err_to_js(err: CoreError) -> JsValue {
    let kind = ErrorKind::from_core_error(&err);
    error_with_code(kind.code_str(), &err.to_string())
}

pub(crate) fn invalid_input_err(msg: impl AsRef<str>) -> JsValue {
    error_with_code(ErrorKind::InvalidInput.code_str(), msg.as_ref())
}

pub(crate) fn unsupported_err(msg: impl AsRef<str>) -> JsValue {
    error_with_code(ErrorKind::UnsupportedOperation.code_str(), msg.as_ref())
}

pub(crate) fn non_finite_err(msg: impl AsRef<str>) -> JsValue {
    error_with_code(ErrorKind::NonFiniteValue.code_str(), msg.as_ref())
}

pub(crate) fn parse_non_negative_usize(value: f64, name: &str) -> Result<usize, JsValue> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > (u32::MAX as f64) {
        return Err(invalid_input_err(format!(
            "{name} must be a non-negative integer <= u32::MAX, got {value}"
        )));
    }
    Ok(value as usize)
}

pub(crate) fn parse_non_negative_u32(value: f64, name: &str) -> Result<u32, JsValue> {
    let v = parse_non_negative_usize(value, name)?;
    u32::try_from(v)
        .map_err(|_| invalid_input_err(format!("{name} must be <= u32::MAX, got {value}")))
}

pub(crate) fn parse_optional_usize(
    value: Option<f64>,
    default: usize,
    name: &str,
) -> Result<usize, JsValue> {
    match value {
        None => Ok(default),
        Some(v) => parse_non_negative_usize(v, name),
    }
}

pub(crate) fn parse_u32_sequence(val: &JsValue, name: &str) -> Result<Vec<u32>, JsValue> {
    if let Some(arr) = val.dyn_ref::<js_sys::Uint32Array>() {
        return Ok(arr.to_vec());
    }
    if val.is_null() || val.is_undefined() || val.is_string() || !val.is_object() {
        return Err(invalid_input_err(format!(
            "{name} must be a Uint32Array or integer array"
        )));
    }
    let iter = js_sys::try_iter(val)
        .map_err(|err| invalid_input_err(format!("failed to iterate {name}: {err:?}")))?
        .ok_or_else(|| {
            invalid_input_err(format!("{name} must be a Uint32Array or integer array"))
        })?;
    let mut out = Vec::new();
    for item_res in iter {
        let item =
            item_res.map_err(|err| invalid_input_err(format!("failed to read {name}: {err:?}")))?;
        let num = item
            .as_f64()
            .ok_or_else(|| invalid_input_err(format!("{name} entries must be integers")))?;
        out.push(parse_non_negative_u32(num, name)?);
    }
    Ok(out)
}

pub(crate) fn parse_optional_f64_sequence(
    val: &JsValue,
    name: &str,
) -> Result<Option<Vec<f64>>, JsValue> {
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
        return Err(invalid_input_err(format!(
            "{name} must be a Float64Array, Float32Array, or numeric array"
        )));
    }
    let iter = js_sys::try_iter(val)
        .map_err(|err| invalid_input_err(format!("failed to iterate {name}: {err:?}")))?
        .ok_or_else(|| {
            invalid_input_err(format!(
                "{name} must be a Float64Array, Float32Array, or numeric array"
            ))
        })?;
    let mut out = Vec::new();
    for item_res in iter {
        let item =
            item_res.map_err(|err| invalid_input_err(format!("failed to read {name}: {err:?}")))?;
        let num = item
            .as_f64()
            .ok_or_else(|| invalid_input_err(format!("{name} entries must be numbers")))?;
        out.push(num);
    }
    Ok(Some(out))
}

pub(crate) fn parse_f64_sequence(val: &JsValue, name: &str) -> Result<Vec<f64>, JsValue> {
    if let Some(arr) = val.dyn_ref::<js_sys::Float64Array>() {
        return Ok(arr.to_vec());
    }
    if let Some(arr) = val.dyn_ref::<js_sys::Float32Array>() {
        return Ok(arr.to_vec().into_iter().map(f64::from).collect());
    }
    if val.is_null() || val.is_undefined() || val.is_string() || !val.is_object() {
        return Err(invalid_input_err(format!(
            "{name} must be a Float64Array, Float32Array, or numeric iterable"
        )));
    }
    let iter = js_sys::try_iter(val)
        .map_err(|err| invalid_input_err(format!("failed to iterate {name}: {err:?}")))?
        .ok_or_else(|| {
            invalid_input_err(format!(
                "{name} must be a Float64Array, Float32Array, or numeric iterable"
            ))
        })?;
    let mut out = Vec::new();
    for item_res in iter {
        let item =
            item_res.map_err(|err| invalid_input_err(format!("failed to read {name}: {err:?}")))?;
        let num = item
            .as_f64()
            .ok_or_else(|| invalid_input_err(format!("{name} entries must be numbers")))?;
        out.push(num);
    }
    Ok(out)
}

pub(crate) fn parse_seed_u64(val: &JsValue) -> Result<u64, JsValue> {
    if val.is_undefined() || val.is_null() {
        return Ok(DEFAULT_SEED);
    }
    if let Some(value) = val.as_f64() {
        if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > MAX_SAFE_INTEGER_F64
        {
            return Err(invalid_input_err(format!(
                "seed must be a non-negative safe integer <= 9007199254740991, got {value}"
            )));
        }
        return Ok(value as u64);
    }
    if val.is_bigint() {
        let bi: &js_sys::BigInt = val.unchecked_ref();
        let s = bi
            .to_string(10)
            .map_err(|err| invalid_input_err(format!("failed to convert BigInt seed: {err:?}")))?
            .as_string()
            .ok_or_else(|| invalid_input_err("BigInt seed did not produce a string"))?;
        return s.parse::<u64>().map_err(|_| {
            invalid_input_err(format!(
                "BigInt seed must be an unsigned 64-bit integer in [0, 2^64 - 1], got {s}"
            ))
        });
    }
    Err(invalid_input_err(
        "seed must be a non-negative safe integer or BigInt in [0, 2^64 - 1]",
    ))
}

pub(crate) fn csr_f64_to_f32(graph: &CsrGraph<f64>) -> Result<CsrGraph<f32>, JsValue> {
    let weights_f32: Option<Vec<f32>> = graph
        .weights()
        .map(|ws| {
            ws.iter()
                .map(|&w| {
                    let wf = w as f32;
                    if !wf.is_finite() {
                        return Err(non_finite_err("coarsened edge weight overflowed f32"));
                    }
                    Ok(wf)
                })
                .collect::<Result<Vec<f32>, JsValue>>()
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

pub(crate) fn set_js_prop(obj: &js_sys::Object, key: &str, val: &JsValue) -> Result<(), JsValue> {
    js_sys::Reflect::set(obj, &JsValue::from_str(key), val)
        .map(|_| ())
        .map_err(|err| invalid_input_err(format!("failed to set {key}: {err:?}")))
}

pub(crate) fn set_f64_prop(obj: &js_sys::Object, key: &str, val: f64) -> Result<(), JsValue> {
    set_js_prop(obj, key, &JsValue::from_f64(val))
}

pub(crate) fn parse_f32_sequence(val: &JsValue, name: &str) -> Result<Vec<f32>, JsValue> {
    if let Some(arr) = val.dyn_ref::<js_sys::Float32Array>() {
        return Ok(arr.to_vec());
    }
    if let Some(arr) = val.dyn_ref::<js_sys::Float64Array>() {
        return arr
            .to_vec()
            .into_iter()
            .map(|v| {
                let f = v as f32;
                if v.is_finite() && !f.is_finite() {
                    return Err(non_finite_err(format!("{name} entry {v} overflows f32")));
                }
                Ok(f)
            })
            .collect();
    }
    if val.is_null() || val.is_undefined() || val.is_string() || !val.is_object() {
        return Err(invalid_input_err(format!(
            "{name} must be a Float32Array, Float64Array, or numeric iterable"
        )));
    }
    let iter = js_sys::try_iter(val)
        .map_err(|err| invalid_input_err(format!("failed to iterate {name}: {err:?}")))?
        .ok_or_else(|| {
            invalid_input_err(format!(
                "{name} must be a Float32Array, Float64Array, or numeric iterable"
            ))
        })?;
    let mut out = Vec::new();
    for item_res in iter {
        let item =
            item_res.map_err(|err| invalid_input_err(format!("failed to read {name}: {err:?}")))?;
        let v = item
            .as_f64()
            .ok_or_else(|| invalid_input_err(format!("{name} entries must be numbers")))?;
        let f = v as f32;
        if v.is_finite() && !f.is_finite() {
            return Err(non_finite_err(format!("{name} entry {v} overflows f32")));
        }
        out.push(f);
    }
    Ok(out)
}

pub(crate) fn parse_optional_f32_sequence(
    val: &JsValue,
    name: &str,
) -> Result<Option<Vec<f32>>, JsValue> {
    if val.is_undefined() || val.is_null() {
        return Ok(None);
    }
    parse_f32_sequence(val, name).map(Some)
}

pub(crate) fn parse_rta_landmark_f64_sequence(
    val: &JsValue,
    name: &str,
) -> Result<Vec<f64>, JsValue> {
    if let Some(arr) = val.dyn_ref::<js_sys::Float64Array>() {
        return Ok(arr.to_vec());
    }
    if let Some(arr) = val.dyn_ref::<js_sys::Float32Array>() {
        return Ok(arr.to_vec().into_iter().map(f64::from).collect());
    }
    if let Some(arr) = val.dyn_ref::<js_sys::Uint32Array>() {
        return Ok(arr.to_vec().into_iter().map(f64::from).collect());
    }
    if let Some(arr) = val.dyn_ref::<js_sys::Int32Array>() {
        return Ok(arr.to_vec().into_iter().map(f64::from).collect());
    }
    parse_f64_sequence(val, name)
}
