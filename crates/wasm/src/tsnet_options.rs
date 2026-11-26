//! Option and positional-argument parsers for `TsNetSolver`.

use egraph::layout::{TsNetBackend, TsNetConfig};
use egraph::neighborhood::SymmetrizationMode;
use wasm_bindgen::prelude::*;

use crate::js_convert::{parse_opt_usize_param, parse_optional_usize, parse_seed_u64};
use crate::js_options::{
    get_prop, is_plain_js_object, normalize_enum_token, opt_bool_prop, opt_f64_prop,
    opt_maybe_f64_prop, opt_maybe_usize_prop, opt_seed_prop, opt_string_prop, opt_usize_prop,
    parse_ibfft_order, reject_extra_args_with_options,
};

fn parse_tsnet_backend(raw: &str) -> Result<TsNetBackend, JsError> {
    match normalize_enum_token(raw).as_str() {
        "exact" => Ok(TsNetBackend::Exact),
        "barneshut" | "bh" => Ok(TsNetBackend::BarnesHut),
        "fft" | "fftinterpolation" | "fit" => Ok(TsNetBackend::FftInterpolation),
        "linearfft" | "l" => Ok(TsNetBackend::LinearFft),
        _ => Err(JsError::new(&format!(
            "unknown TsNetBackend {raw:?}; expected 'exact', 'barnes_hut', 'fft', or 'linear_fft'"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn parse_tsnet_config(
    backend_or_options: &JsValue,
    iterations_stage1: Option<f64>,
    iterations_stage2: Option<f64>,
    iterations_stage3: Option<f64>,
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
        let has_extra = iterations_stage1.is_some()
            || iterations_stage2.is_some()
            || iterations_stage3.is_some()
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
        cfg.iterations_stage1 = opt_usize_prop(
            backend_or_options,
            "iterationsStage1",
            cfg.iterations_stage1,
        )?;
        cfg.iterations_stage2 = opt_usize_prop(
            backend_or_options,
            "iterationsStage2",
            cfg.iterations_stage2,
        )?;
        cfg.iterations_stage3 = opt_usize_prop(
            backend_or_options,
            "iterationsStage3",
            cfg.iterations_stage3,
        )?;
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
        cfg.momentum = opt_f64_prop(backend_or_options, "momentum", cfg.momentum)?;
        cfg.epsilon_r = opt_f64_prop(backend_or_options, "epsilonR", cfg.epsilon_r)?;
        cfg.theta = opt_f64_prop(backend_or_options, "theta", cfg.theta)?;

        let ord_primary = get_prop(backend_or_options, "interpolationOrder")?;
        let ord_alt = get_prop(backend_or_options, "ibfftOrder")?;
        if !ord_primary.is_undefined() && !ord_primary.is_null() {
            cfg.ibfft.order = parse_ibfft_order(&ord_primary)?;
        } else if !ord_alt.is_undefined() && !ord_alt.is_null() {
            cfg.ibfft.order = parse_ibfft_order(&ord_alt)?;
        }
        let int_primary = opt_maybe_usize_prop(backend_or_options, "intervals")?;
        let int_alt = opt_maybe_usize_prop(backend_or_options, "ibfftIntervals")?;
        cfg.ibfft.intervals = int_primary.or(int_alt).unwrap_or(cfg.ibfft.intervals);

        let min_g_primary = opt_maybe_usize_prop(backend_or_options, "minGridDim")?;
        let min_g_alt = opt_maybe_usize_prop(backend_or_options, "ibfftMinGridDim")?;
        cfg.ibfft.min_grid_dim = min_g_primary
            .or(min_g_alt)
            .unwrap_or(cfg.ibfft.min_grid_dim);

        let max_g_primary = opt_maybe_usize_prop(backend_or_options, "maxGridDim")?;
        let max_g_alt = opt_maybe_usize_prop(backend_or_options, "ibfftMaxGridDim")?;
        cfg.ibfft.max_grid_dim = max_g_primary
            .or(max_g_alt)
            .unwrap_or(cfg.ibfft.max_grid_dim);

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

    cfg.iterations_stage1 =
        parse_optional_usize(iterations_stage1, cfg.iterations_stage1, "iterationsStage1")?;
    cfg.iterations_stage2 =
        parse_optional_usize(iterations_stage2, cfg.iterations_stage2, "iterationsStage2")?;
    cfg.iterations_stage3 =
        parse_optional_usize(iterations_stage3, cfg.iterations_stage3, "iterationsStage3")?;
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn parse_row_normalized_tsnet_args(
    w2_or_options: &JsValue,
    max_2hop_per_node: Option<f64>,
    backend_or_options: &JsValue,
    iterations_stage1: Option<f64>,
    iterations_stage2: Option<f64>,
    iterations_stage3: Option<f64>,
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
) -> Result<(f64, Option<usize>, f64, TsNetConfig), JsError> {
    if is_plain_js_object(w2_or_options) {
        let has_extra = max_2hop_per_node.is_some()
            || (!backend_or_options.is_undefined() && !backend_or_options.is_null())
            || iterations_stage1.is_some()
            || iterations_stage2.is_some()
            || iterations_stage3.is_some()
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
        let cfg = parse_tsnet_config(
            w2_or_options,
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
            None,
            None,
            None,
            &JsValue::UNDEFINED,
            None,
            None,
        )?;
        Ok((w2_val, max_2hop, beta, cfg))
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
            iterations_stage1,
            iterations_stage2,
            iterations_stage3,
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
        Ok((w2_val, max_2hop, beta, cfg))
    }
}
