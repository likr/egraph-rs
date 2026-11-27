//! Option and positional-argument parsers for `TsNetSolver`.

use egraph::layout::{TsNetBackend, TsNetConfig};
use egraph::neighborhood::{SymmetrizationMode, TwoHopSamplingMode};
use wasm_bindgen::prelude::*;

use crate::js_convert::{
    parse_non_negative_usize, parse_opt_usize_param, parse_optional_usize, parse_seed_u64,
};
use crate::js_options::{
    get_prop, is_plain_js_object, normalize_enum_token, opt_bool_prop, opt_f64_prop,
    opt_maybe_f64_prop, opt_maybe_usize_prop, opt_seed_prop, opt_string_prop, opt_usize_prop,
    parse_ibfft_order, reject_extra_args_with_options,
};
use wasm_bindgen::JsCast;

pub(crate) fn parse_tsnet_backend(raw: &str) -> Result<TsNetBackend, JsError> {
    match normalize_enum_token(raw).as_str() {
        "exact" => Ok(TsNetBackend::Exact),
        "barneshut" | "bh" => Ok(TsNetBackend::BarnesHut),
        "fft" | "fftinterpolation" | "fit" => Ok(TsNetBackend::FftInterpolation),
        "linearfft" | "l" => Ok(TsNetBackend::LinearFft),
        "auto" => Ok(TsNetBackend::Auto),
        _ => Err(JsError::new(&format!(
            "unknown TsNetBackend {raw:?}; expected 'exact', 'barnes_hut', 'fft', 'linear_fft', or 'auto'"
        ))),
    }
}

pub(crate) fn tsnet_backend_str(backend: TsNetBackend) -> &'static str {
    match backend {
        TsNetBackend::Exact => "exact",
        TsNetBackend::BarnesHut => "barnes_hut",
        TsNetBackend::FftInterpolation => "fft",
        TsNetBackend::LinearFft => "linear_fft",
        TsNetBackend::Auto => "auto",
    }
}

pub(crate) fn parse_tsnet_sampling_mode(raw: &str) -> Result<TwoHopSamplingMode, JsError> {
    match normalize_enum_token(raw).as_str() {
        "degreeorderedhubsuppressed" | "degreeordered" | "hubsuppressed" => {
            Ok(TwoHopSamplingMode::DegreeOrderedHubSuppressed)
        }
        "twosteprandomwalk" | "randomwalk" => Ok(TwoHopSamplingMode::TwoStepRandomWalk),
        _ => Err(JsError::new(&format!(
            "unknown TwoHopSamplingMode {raw:?}; expected 'degree_ordered_hub_suppressed' or 'two_step_random_walk'"
        ))),
    }
}

pub(crate) fn tsnet_sampling_mode_str(mode: TwoHopSamplingMode) -> &'static str {
    match mode {
        TwoHopSamplingMode::DegreeOrderedHubSuppressed => "degree_ordered_hub_suppressed",
        TwoHopSamplingMode::TwoStepRandomWalk => "two_step_random_walk",
    }
}

pub(crate) fn parse_stage_iterations_val(
    si_val: &JsValue,
    default: [usize; 3],
) -> Result<[usize; 3], JsError> {
    if si_val.is_undefined() || si_val.is_null() {
        return Ok(default);
    }
    let mut vals = Vec::with_capacity(3);
    if let Some(arr) = si_val.dyn_ref::<js_sys::Uint32Array>() {
        for v in arr.to_vec() {
            vals.push(v as usize);
        }
    } else if !si_val.is_string() && si_val.is_object() {
        let iter = js_sys::try_iter(si_val)
            .map_err(|err| JsError::new(&format!("failed to iterate stageIterations: {err:?}")))?
            .ok_or_else(|| JsError::new("stageIterations must be a 3-element integer array"))?;
        for item_res in iter {
            let item = item_res
                .map_err(|err| JsError::new(&format!("failed to read stageIterations: {err:?}")))?;
            let num = item
                .as_f64()
                .ok_or_else(|| JsError::new("stageIterations entries must be integers"))?;
            vals.push(parse_non_negative_usize(num, "stageIterations")?);
        }
    } else {
        return Err(JsError::new(
            "stageIterations must be a 3-element integer array",
        ));
    }
    if vals.len() != 3 {
        return Err(JsError::new(&format!(
            "stageIterations must have length 3, got {}",
            vals.len()
        )));
    }
    Ok([vals[0], vals[1], vals[2]])
}

pub(crate) fn parse_sampling_mode_from_options(
    options: &JsValue,
    default: TwoHopSamplingMode,
) -> Result<TwoHopSamplingMode, JsError> {
    if let Some(raw_mode) = opt_string_prop(options, "mode")? {
        return parse_tsnet_sampling_mode(&raw_mode);
    }
    if let Some(raw_mode) = opt_string_prop(options, "samplingMode")? {
        return parse_tsnet_sampling_mode(&raw_mode);
    }
    let rw_val = get_prop(options, "twoStepRandomWalk")?;
    if !rw_val.is_undefined() && !rw_val.is_null() {
        let rw = rw_val
            .as_bool()
            .ok_or_else(|| JsError::new("options.twoStepRandomWalk must be a boolean"))?;
        return Ok(if rw {
            TwoHopSamplingMode::TwoStepRandomWalk
        } else {
            TwoHopSamplingMode::DegreeOrderedHubSuppressed
        });
    }
    Ok(default)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn parse_tsnet_config(
    backend_or_options: &JsValue,
    stage_iterations: &JsValue,
    learning_rate: Option<f64>,
    perplexity_c: Option<f64>,
    min_perplexity: Option<f64>,
    max_perplexity: Option<f64>,
    exaggeration: Option<f64>,
    lambda_c_stage1: Option<f64>,
    lambda_c_stage2: Option<f64>,
    lambda_c_stage3: Option<f64>,
    lambda_r_stage3: Option<f64>,
    momentum: Option<f64>,
    epsilon_r: Option<f64>,
    theta: Option<f64>,
    power: Option<f64>,
    sigma_iters: Option<f64>,
    sigma_tolerance: Option<f64>,
    row_normalize: Option<bool>,
    seed: Option<f64>,
    intervals: Option<f64>,
    interpolation_order: &JsValue,
    min_grid_dim: Option<f64>,
    max_grid_dim: Option<f64>,
) -> Result<TsNetConfig, JsError> {
    let mut cfg = TsNetConfig::pure_2hop();
    if is_plain_js_object(backend_or_options) {
        let has_extra = (!stage_iterations.is_undefined() && !stage_iterations.is_null())
            || learning_rate.is_some()
            || perplexity_c.is_some()
            || min_perplexity.is_some()
            || max_perplexity.is_some()
            || exaggeration.is_some()
            || lambda_c_stage1.is_some()
            || lambda_c_stage2.is_some()
            || lambda_c_stage3.is_some()
            || lambda_r_stage3.is_some()
            || momentum.is_some()
            || epsilon_r.is_some()
            || theta.is_some()
            || power.is_some()
            || sigma_iters.is_some()
            || sigma_tolerance.is_some()
            || row_normalize.is_some()
            || seed.is_some()
            || intervals.is_some()
            || (!interpolation_order.is_undefined() && !interpolation_order.is_null())
            || min_grid_dim.is_some()
            || max_grid_dim.is_some();
        reject_extra_args_with_options(has_extra)?;

        if let Some(b_str) = opt_string_prop(backend_or_options, "backend")? {
            cfg.backend = parse_tsnet_backend(&b_str)?;
        }
        cfg.perplexity_c = opt_f64_prop(backend_or_options, "perplexityC", cfg.perplexity_c)?;
        cfg.min_perplexity = opt_f64_prop(backend_or_options, "minPerplexity", cfg.min_perplexity)?;
        cfg.max_perplexity = opt_f64_prop(backend_or_options, "maxPerplexity", cfg.max_perplexity)?;
        cfg.power = opt_f64_prop(backend_or_options, "power", cfg.power)?;
        cfg.sigma_iters = opt_usize_prop(backend_or_options, "sigmaIters", cfg.sigma_iters)?;
        cfg.sigma_tolerance =
            opt_f64_prop(backend_or_options, "sigmaTolerance", cfg.sigma_tolerance)?;
        if opt_bool_prop(backend_or_options, "rowNormalize", false)? {
            cfg.symmetrization = SymmetrizationMode::RowNormalized;
        }
        cfg.jacobi_precondition = opt_bool_prop(
            backend_or_options,
            "jacobiPrecondition",
            cfg.jacobi_precondition,
        )?;
        let si_val = get_prop(backend_or_options, "stageIterations")?;
        cfg.stage_iterations = parse_stage_iterations_val(&si_val, cfg.stage_iterations)?;
        cfg.exaggeration = opt_f64_prop(backend_or_options, "exaggeration", cfg.exaggeration)?;
        cfg.lambda_c_stage1 =
            opt_f64_prop(backend_or_options, "lambdaCStage1", cfg.lambda_c_stage1)?;
        cfg.lambda_c_stage2 =
            opt_f64_prop(backend_or_options, "lambdaCStage2", cfg.lambda_c_stage2)?;
        cfg.lambda_c_stage3 =
            opt_f64_prop(backend_or_options, "lambdaCStage3", cfg.lambda_c_stage3)?;
        cfg.lambda_r_stage3 =
            opt_f64_prop(backend_or_options, "lambdaRStage3", cfg.lambda_r_stage3)?;
        cfg.learning_rate =
            opt_maybe_f64_prop(backend_or_options, "learningRate")?.or(cfg.learning_rate);
        cfg.learning_rate_divisor = opt_f64_prop(
            backend_or_options,
            "learningRateDivisor",
            cfg.learning_rate_divisor,
        )?;
        cfg.momentum = opt_f64_prop(backend_or_options, "momentum", cfg.momentum)?;
        cfg.epsilon_r = opt_f64_prop(backend_or_options, "epsilonR", cfg.epsilon_r)?;
        cfg.theta = opt_f64_prop(backend_or_options, "theta", cfg.theta)?;

        let ord_primary = get_prop(backend_or_options, "interpolationOrder")?;
        if !ord_primary.is_undefined() && !ord_primary.is_null() {
            cfg.ibfft.order = parse_ibfft_order(&ord_primary)?;
        }
        cfg.ibfft.intervals = opt_usize_prop(backend_or_options, "intervals", cfg.ibfft.intervals)?;
        cfg.ibfft.min_grid_dim =
            opt_usize_prop(backend_or_options, "minGridDim", cfg.ibfft.min_grid_dim)?;
        cfg.ibfft.max_grid_dim =
            opt_usize_prop(backend_or_options, "maxGridDim", cfg.ibfft.max_grid_dim)?;

        cfg.seed = opt_seed_prop(backend_or_options)?;
        return Ok(cfg);
    }

    if !backend_or_options.is_undefined() && !backend_or_options.is_null() {
        let b_str = backend_or_options.as_string().ok_or_else(|| {
            JsError::new(
                "backendOrOptions must be a backend string, options object, null, or undefined",
            )
        })?;
        cfg.backend = parse_tsnet_backend(&b_str)?;
    }

    cfg.stage_iterations = parse_stage_iterations_val(stage_iterations, cfg.stage_iterations)?;
    cfg.learning_rate = learning_rate.or(cfg.learning_rate);
    cfg.perplexity_c = perplexity_c.unwrap_or(cfg.perplexity_c);
    cfg.min_perplexity = min_perplexity.unwrap_or(cfg.min_perplexity);
    cfg.max_perplexity = max_perplexity.unwrap_or(cfg.max_perplexity);
    cfg.exaggeration = exaggeration.unwrap_or(cfg.exaggeration);
    cfg.lambda_c_stage1 = lambda_c_stage1.unwrap_or(cfg.lambda_c_stage1);
    cfg.lambda_c_stage2 = lambda_c_stage2.unwrap_or(cfg.lambda_c_stage2);
    cfg.lambda_c_stage3 = lambda_c_stage3.unwrap_or(cfg.lambda_c_stage3);
    cfg.lambda_r_stage3 = lambda_r_stage3.unwrap_or(cfg.lambda_r_stage3);
    cfg.momentum = momentum.unwrap_or(cfg.momentum);
    cfg.epsilon_r = epsilon_r.unwrap_or(cfg.epsilon_r);
    cfg.theta = theta.unwrap_or(cfg.theta);
    cfg.power = power.unwrap_or(cfg.power);
    cfg.sigma_iters = parse_optional_usize(sigma_iters, cfg.sigma_iters, "sigmaIters")?;
    cfg.sigma_tolerance = sigma_tolerance.unwrap_or(cfg.sigma_tolerance);
    if row_normalize.unwrap_or(false) {
        cfg.symmetrization = SymmetrizationMode::RowNormalized;
    }
    cfg.seed = parse_seed_u64(seed)?;
    cfg.ibfft.intervals = parse_optional_usize(intervals, cfg.ibfft.intervals, "intervals")?;
    if !interpolation_order.is_undefined() && !interpolation_order.is_null() {
        cfg.ibfft.order = parse_ibfft_order(interpolation_order)?;
    }
    cfg.ibfft.min_grid_dim =
        parse_optional_usize(min_grid_dim, cfg.ibfft.min_grid_dim, "minGridDim")?;
    cfg.ibfft.max_grid_dim =
        parse_optional_usize(max_grid_dim, cfg.ibfft.max_grid_dim, "maxGridDim")?;
    Ok(cfg)
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn parse_row_normalized_tsnet_args(
    w2_or_options: &JsValue,
    max_2hop_per_node: Option<f64>,
    backend_or_options: &JsValue,
    stage_iterations: &JsValue,
    learning_rate: Option<f64>,
    exaggeration: Option<f64>,
    lambda_c_stage1: Option<f64>,
    lambda_c_stage2: Option<f64>,
    lambda_c_stage3: Option<f64>,
    lambda_r_stage3: Option<f64>,
    momentum: Option<f64>,
    epsilon_r: Option<f64>,
    theta: Option<f64>,
    seed: Option<f64>,
    intervals: Option<f64>,
    interpolation_order: &JsValue,
    min_grid_dim: Option<f64>,
    max_grid_dim: Option<f64>,
    degree_beta: Option<f64>,
) -> Result<
    (
        f64,
        Option<usize>,
        f64,
        TwoHopSamplingMode,
        Option<usize>,
        TsNetConfig,
    ),
    JsError,
> {
    if is_plain_js_object(w2_or_options) {
        let has_extra = max_2hop_per_node.is_some()
            || (!backend_or_options.is_undefined() && !backend_or_options.is_null())
            || (!stage_iterations.is_undefined() && !stage_iterations.is_null())
            || learning_rate.is_some()
            || exaggeration.is_some()
            || lambda_c_stage1.is_some()
            || lambda_c_stage2.is_some()
            || lambda_c_stage3.is_some()
            || lambda_r_stage3.is_some()
            || momentum.is_some()
            || epsilon_r.is_some()
            || theta.is_some()
            || seed.is_some()
            || intervals.is_some()
            || (!interpolation_order.is_undefined() && !interpolation_order.is_null())
            || min_grid_dim.is_some()
            || max_grid_dim.is_some()
            || degree_beta.is_some();
        reject_extra_args_with_options(has_extra)?;
        let w2_val = opt_f64_prop(w2_or_options, "w2", 0.5)?;
        let max_2hop = opt_maybe_usize_prop(w2_or_options, "max2hopPerNode")?;
        let beta = opt_f64_prop(w2_or_options, "degreeBeta", 0.0)?;
        let sampling_mode =
            parse_sampling_mode_from_options(w2_or_options, TwoHopSamplingMode::TwoStepRandomWalk)?;
        let max_bridge_degree = opt_maybe_usize_prop(w2_or_options, "maxBridgeDegree")?;
        let cfg = parse_tsnet_config(
            w2_or_options,
            &JsValue::UNDEFINED,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &JsValue::UNDEFINED,
            None,
            None,
        )?;
        Ok((
            w2_val,
            max_2hop,
            beta,
            sampling_mode,
            max_bridge_degree,
            cfg,
        ))
    } else {
        let w2_val = if w2_or_options.is_undefined() || w2_or_options.is_null() {
            0.5
        } else {
            w2_or_options.as_f64().ok_or_else(|| {
                JsError::new(
                    "w2OrOptions must be a finite non-negative number, options object, null, or undefined",
                )
            })?
        };
        let max_2hop = parse_opt_usize_param(max_2hop_per_node, "max2hopPerNode")?;
        let beta = degree_beta.unwrap_or(0.0);
        let cfg = parse_tsnet_config(
            backend_or_options,
            stage_iterations,
            learning_rate,
            None,
            None,
            None,
            exaggeration,
            lambda_c_stage1,
            lambda_c_stage2,
            lambda_c_stage3,
            lambda_r_stage3,
            momentum,
            epsilon_r,
            theta,
            None,
            None,
            None,
            None,
            seed,
            intervals,
            interpolation_order,
            min_grid_dim,
            max_grid_dim,
        )?;
        Ok((
            w2_val,
            max_2hop,
            beta,
            TwoHopSamplingMode::TwoStepRandomWalk,
            None,
            cfg,
        ))
    }
}
