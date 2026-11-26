//! Shared JavaScript plain-object option parsers, enum token normalizers, and solver config builders.

use crate::js_convert::{
    core_err_to_js, parse_non_negative_usize, parse_opt_usize_param, parse_optional_usize,
    parse_seed_u64,
};
use egraph::core::SolverStatus;
use egraph::layout::PacmapGraphConfig;
use egraph::neighborhood::DogVaradhanConfig;
use egraph::spatial::InterpolationOrder;
use wasm_bindgen::prelude::*;

pub(crate) fn normalize_enum_token(raw: &str) -> String {
    raw.chars()
        .filter(|ch| !matches!(ch, '_' | '-' | ' '))
        .map(|ch| ch.to_ascii_lowercase())
        .collect()
}

pub(crate) fn solver_status_str(status: SolverStatus) -> &'static str {
    match status {
        SolverStatus::Initialized => "initialized",
        SolverStatus::Running => "running",
        SolverStatus::Converged => "converged",
        SolverStatus::Completed => "completed",
    }
}

pub(crate) fn parse_ibfft_order(val: &JsValue) -> Result<InterpolationOrder, JsError> {
    if let Some(num) = val.as_f64() {
        let ord = parse_non_negative_usize(num, "ibfftOrder")?;
        return InterpolationOrder::from_order(ord).map_err(core_err_to_js);
    }
    if let Some(raw) = val.as_string() {
        return match normalize_enum_token(&raw).as_str() {
            "bilinear" | "1" => Ok(InterpolationOrder::Bilinear),
            "cubic" | "3" => Ok(InterpolationOrder::Cubic),
            _ => Err(JsError::new(&format!(
                "unknown ibfftOrder {raw:?}; expected 'bilinear' (1) or 'cubic' (3)"
            ))),
        };
    }
    Err(JsError::new(
        "ibfftOrder must be 'bilinear', 'cubic', 1, or 3",
    ))
}

pub(crate) fn is_plain_js_object(val: &JsValue) -> bool {
    val.is_object()
        && !js_sys::Array::is_array(val)
        && !js_sys::ArrayBuffer::is_view(val)
        && !val.is_function()
}

pub(crate) fn check_options_arg(options: &JsValue, ctx: &str) -> Result<bool, JsError> {
    if options.is_undefined() || options.is_null() {
        return Ok(false);
    }
    if is_plain_js_object(options) {
        return Ok(true);
    }
    Err(JsError::new(&format!(
        "{ctx}: options must be a plain object, null, or undefined"
    )))
}

pub(crate) fn get_prop(obj: &JsValue, key: &str) -> Result<JsValue, JsError> {
    js_sys::Reflect::get(obj, &JsValue::from_str(key))
        .map_err(|err| JsError::new(&format!("failed to read options.{key}: {err:?}")))
}

pub(crate) fn opt_usize_prop(obj: &JsValue, key: &str, default: usize) -> Result<usize, JsError> {
    let val = get_prop(obj, key)?;
    if val.is_undefined() || val.is_null() {
        return Ok(default);
    }
    let num = val
        .as_f64()
        .ok_or_else(|| JsError::new(&format!("options.{key} must be an integer number")))?;
    parse_non_negative_usize(num, key)
}

pub(crate) fn opt_maybe_usize_prop(obj: &JsValue, key: &str) -> Result<Option<usize>, JsError> {
    let val = get_prop(obj, key)?;
    if val.is_undefined() || val.is_null() {
        return Ok(None);
    }
    let num = val
        .as_f64()
        .ok_or_else(|| JsError::new(&format!("options.{key} must be an integer number")))?;
    parse_non_negative_usize(num, key).map(Some)
}

pub(crate) fn opt_f64_prop(obj: &JsValue, key: &str, default: f64) -> Result<f64, JsError> {
    let val = get_prop(obj, key)?;
    if val.is_undefined() || val.is_null() {
        return Ok(default);
    }
    val.as_f64()
        .ok_or_else(|| JsError::new(&format!("options.{key} must be a number")))
}

pub(crate) fn opt_maybe_f64_prop(obj: &JsValue, key: &str) -> Result<Option<f64>, JsError> {
    let val = get_prop(obj, key)?;
    if val.is_undefined() || val.is_null() {
        return Ok(None);
    }
    val.as_f64()
        .map(Some)
        .ok_or_else(|| JsError::new(&format!("options.{key} must be a number")))
}

pub(crate) fn opt_bool_prop(obj: &JsValue, key: &str, default: bool) -> Result<bool, JsError> {
    let val = get_prop(obj, key)?;
    if val.is_undefined() || val.is_null() {
        return Ok(default);
    }
    val.as_bool()
        .ok_or_else(|| JsError::new(&format!("options.{key} must be a boolean")))
}

pub(crate) fn opt_string_prop(obj: &JsValue, key: &str) -> Result<Option<String>, JsError> {
    let val = get_prop(obj, key)?;
    if val.is_undefined() || val.is_null() {
        return Ok(None);
    }
    val.as_string()
        .map(Some)
        .ok_or_else(|| JsError::new(&format!("options.{key} must be a string")))
}

pub(crate) fn opt_seed_prop(obj: &JsValue) -> Result<u64, JsError> {
    let val = get_prop(obj, "seed")?;
    if val.is_undefined() || val.is_null() {
        return parse_seed_u64(None);
    }
    let num = val
        .as_f64()
        .ok_or_else(|| JsError::new("options.seed must be a non-negative safe integer"))?;
    parse_seed_u64(Some(num))
}

pub(crate) fn reject_extra_args_with_options(has_extra: bool) -> Result<(), JsError> {
    if has_extra {
        return Err(JsError::new(
            "unexpected trailing positional argument when passing an options object",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn parse_dog_config_from_options_or_args(
    maybe_options: &JsValue,
    seed: u64,
    degree: Option<f64>,
    num_landmarks: Option<f64>,
    degree_normalize: Option<bool>,
    p_exp: Option<f64>,
    eps_psd: Option<f64>,
    t_coarse: Option<f64>,
    t1: Option<f64>,
    t2: Option<f64>,
    alpha: Option<f64>,
    beta_floor: Option<f64>,
    min_degree: Option<f64>,
    jl_dim: Option<f64>,
) -> Result<DogVaradhanConfig, JsError> {
    let mut cfg = DogVaradhanConfig {
        seed,
        ..DogVaradhanConfig::default()
    };
    if is_plain_js_object(maybe_options) {
        let deg_primary = opt_maybe_usize_prop(maybe_options, "degree")?;
        let deg_alt = opt_maybe_usize_prop(maybe_options, "dogDegree")?;
        cfg.degree = deg_primary.or(deg_alt);

        let lm_primary = opt_maybe_usize_prop(maybe_options, "numLandmarks")?;
        let lm_alt = opt_maybe_usize_prop(maybe_options, "dogNumLandmarks")?;
        cfg.n_landmarks = lm_primary.or(lm_alt).unwrap_or(cfg.n_landmarks);

        cfg.min_degree = opt_usize_prop(maybe_options, "minDegree", cfg.min_degree)?;
        cfg.jl_dim = opt_usize_prop(maybe_options, "jlDim", cfg.jl_dim)?;

        let dn_val = get_prop(maybe_options, "degreeNormalize")?;
        let dn_alt = get_prop(maybe_options, "dogDegreeNormalize")?;
        if !dn_val.is_undefined() && !dn_val.is_null() {
            cfg.degree_normalize = dn_val
                .as_bool()
                .ok_or_else(|| JsError::new("options.degreeNormalize must be a boolean"))?;
        } else if !dn_alt.is_undefined() && !dn_alt.is_null() {
            cfg.degree_normalize = dn_alt
                .as_bool()
                .ok_or_else(|| JsError::new("options.dogDegreeNormalize must be a boolean"))?;
        }

        let p_primary = opt_maybe_f64_prop(maybe_options, "pExp")?;
        let p_alt = opt_maybe_f64_prop(maybe_options, "dogPExp")?;
        cfg.p_exp = p_primary.or(p_alt).unwrap_or(cfg.p_exp);

        let psd_primary = opt_maybe_f64_prop(maybe_options, "epsPsd")?;
        let psd_alt = opt_maybe_f64_prop(maybe_options, "dogEpsPsd")?;
        cfg.eps_psd = psd_primary.or(psd_alt).unwrap_or(cfg.eps_psd);

        let tc_primary = opt_maybe_f64_prop(maybe_options, "tCoarse")?;
        let tc_alt = opt_maybe_f64_prop(maybe_options, "dogTCoarse")?;
        cfg.t_coarse = tc_primary.or(tc_alt).unwrap_or(cfg.t_coarse);

        cfg.t1 = opt_maybe_f64_prop(maybe_options, "t1")?
            .or(opt_maybe_f64_prop(maybe_options, "dogT1")?);
        cfg.t2 = opt_maybe_f64_prop(maybe_options, "t2")?
            .or(opt_maybe_f64_prop(maybe_options, "dogT2")?);
        cfg.alpha = opt_maybe_f64_prop(maybe_options, "alpha")?
            .or(opt_maybe_f64_prop(maybe_options, "dogAlpha")?);
        cfg.beta_floor = opt_maybe_f64_prop(maybe_options, "betaFloor")?
            .or(opt_maybe_f64_prop(maybe_options, "dogBetaFloor")?);
        return Ok(cfg);
    }

    cfg.degree = parse_opt_usize_param(degree, "degree")?;
    cfg.n_landmarks = parse_optional_usize(num_landmarks, cfg.n_landmarks, "numLandmarks")?;
    cfg.min_degree = parse_optional_usize(min_degree, cfg.min_degree, "minDegree")?;
    cfg.jl_dim = parse_optional_usize(jl_dim, cfg.jl_dim, "jlDim")?;
    cfg.degree_normalize = degree_normalize.unwrap_or(cfg.degree_normalize);
    cfg.p_exp = p_exp.unwrap_or(cfg.p_exp);
    cfg.eps_psd = eps_psd.unwrap_or(cfg.eps_psd);
    cfg.t_coarse = t_coarse.unwrap_or(cfg.t_coarse);
    cfg.t1 = t1;
    cfg.t2 = t2;
    cfg.alpha = alpha;
    cfg.beta_floor = beta_floor;
    Ok(cfg)
}

pub(crate) fn parse_pacmap_graph_config(
    options: &JsValue,
    ctx: &str,
) -> Result<PacmapGraphConfig, JsError> {
    let mut cfg = PacmapGraphConfig::default();
    if check_options_arg(options, ctx)? {
        cfg.n_neighbors = opt_usize_prop(options, "nNeighbors", cfg.n_neighbors)?;
        cfg.mn_ratio = opt_f64_prop(options, "mnRatio", cfg.mn_ratio)?;
        cfg.fp_ratio = opt_f64_prop(options, "fpRatio", cfg.fp_ratio)?;
        cfg.phase1_iters = opt_usize_prop(options, "phase1Iters", cfg.phase1_iters)?;
        cfg.phase2_iters = opt_usize_prop(options, "phase2Iters", cfg.phase2_iters)?;
        cfg.phase3_iters = opt_usize_prop(options, "phase3Iters", cfg.phase3_iters)?;
        cfg.learning_rate = opt_f64_prop(options, "learningRate", cfg.learning_rate)?;
        cfg.keep_all_1hop = opt_bool_prop(options, "keepAll1hop", cfg.keep_all_1hop)?;
        cfg.max_1hop_per_node = opt_maybe_usize_prop(options, "max1hopPerNode")?;
        cfg.degree_weight_beta = opt_f64_prop(options, "degreeWeightBeta", cfg.degree_weight_beta)?;
        cfg.num_landmarks = opt_usize_prop(options, "numLandmarks", cfg.num_landmarks)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    Ok(cfg)
}
