//! Option, stage, and `(target, backend)` parsers for `TsNetSolver`.

use egraph::layout::{
    TsNetConfig, TsNetStage,
    neighbor::{
        Auto2d, Auto3d, BarnesHut as NeighborBarnesHut, DEFAULT_AUTO_EXACT_THRESHOLD,
        Exact as NeighborExact, FitFft, LinearFft, TsNetBackend2d, TsNetBackend3d,
    },
};
use egraph::neighborhood::{SymmetrizationMode, TwoHopSamplingMode};
use wasm_bindgen::{JsCast, prelude::*};

use crate::js_convert::{invalid_input_err, parse_non_negative_usize, unsupported_err};
use crate::js_options::{
    JsTargetKind, check_options_arg, get_prop, normalize_enum_token, opt_bool_prop, opt_f64_prop,
    opt_maybe_f64_prop, opt_seed_prop, opt_string_prop, opt_target_prop, opt_usize_prop,
};
use crate::t_fdp::parse_ibfft_from_options;

pub(crate) enum TsNetBackendChoice {
    Dim2(TsNetBackend2d),
    Dim3(TsNetBackend3d),
    DimN(NeighborExact),
    Manifold(NeighborExact),
}

pub(crate) fn tsnet_resolved_backend_2d(
    backend: &TsNetBackend2d,
    num_nodes: usize,
) -> &'static str {
    match backend {
        TsNetBackend2d::Exact(_) => "exact",
        TsNetBackend2d::BarnesHut(_) => "barnesHut",
        TsNetBackend2d::FitFft(_) => "fitFft",
        TsNetBackend2d::LinearFft(_) => "linearFft",
        TsNetBackend2d::Auto(a) => {
            if num_nodes <= a.exact_threshold() {
                "exact"
            } else {
                "linearFft"
            }
        }
        _ => "auto2d",
    }
}

pub(crate) fn tsnet_resolved_backend_3d(
    backend: &TsNetBackend3d,
    num_nodes: usize,
) -> &'static str {
    match backend {
        TsNetBackend3d::Exact(_) => "exact",
        TsNetBackend3d::BarnesHut(_) => "barnesHut",
        TsNetBackend3d::Auto(a) => {
            if num_nodes <= a.exact_threshold() {
                "exact"
            } else {
                "barnesHut"
            }
        }
        _ => "auto3d",
    }
}

pub(crate) fn parse_tsnet_sampling_mode(raw: &str) -> Result<TwoHopSamplingMode, JsValue> {
    match normalize_enum_token(raw)?.as_str() {
        "degreeorderedhubsuppressed" | "degreeordered" | "hubsuppressed" => {
            Ok(TwoHopSamplingMode::DegreeOrderedHubSuppressed)
        }
        "twosteprandomwalk" | "randomwalk" => Ok(TwoHopSamplingMode::TwoStepRandomWalk),
        _ => Err(invalid_input_err(format!(
            "unknown TwoHopSamplingMode {raw:?}; expected 'degreeOrderedHubSuppressed' or 'twoStepRandomWalk'"
        ))),
    }
}

pub(crate) fn tsnet_sampling_mode_str(mode: TwoHopSamplingMode) -> &'static str {
    match mode {
        TwoHopSamplingMode::DegreeOrderedHubSuppressed => "degreeOrderedHubSuppressed",
        TwoHopSamplingMode::TwoStepRandomWalk => "twoStepRandomWalk",
        _ => "degreeOrderedHubSuppressed",
    }
}

pub(crate) fn parse_stage_iterations_val(
    si_val: &JsValue,
    default: [usize; 3],
) -> Result<[usize; 3], JsValue> {
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
            .map_err(|err| {
                invalid_input_err(format!("failed to iterate stageIterations: {err:?}"))
            })?
            .ok_or_else(|| {
                invalid_input_err("stageIterations must be a 3-element integer array")
            })?;
        for item_res in iter {
            let item = item_res.map_err(|err| {
                invalid_input_err(format!("failed to read stageIterations: {err:?}"))
            })?;
            let num = item
                .as_f64()
                .ok_or_else(|| invalid_input_err("stageIterations entries must be integers"))?;
            vals.push(parse_non_negative_usize(num, "stageIterations")?);
        }
    } else {
        return Err(invalid_input_err(
            "stageIterations must be a 3-element integer array",
        ));
    }
    if vals.len() != 3 {
        return Err(invalid_input_err(format!(
            "stageIterations must have length 3, got {}",
            vals.len()
        )));
    }
    Ok([vals[0], vals[1], vals[2]])
}

pub(crate) fn parse_stages_array(stages: &JsValue) -> Result<Vec<TsNetStage>, JsValue> {
    if stages.is_null() || stages.is_undefined() || stages.is_string() || !stages.is_object() {
        return Err(invalid_input_err(
            "stages must be an array or iterable of TsNetStage objects",
        ));
    }
    let iter = js_sys::try_iter(stages)
        .map_err(|err| invalid_input_err(format!("failed to iterate stages: {err:?}")))?
        .ok_or_else(|| {
            invalid_input_err("stages must be an array or iterable of TsNetStage objects")
        })?;
    let mut out = Vec::new();
    for item_res in iter {
        let item = item_res
            .map_err(|err| invalid_input_err(format!("failed to read stage item: {err:?}")))?;
        if item.is_null() || !item.is_object() {
            return Err(invalid_input_err(
                "each stage entry must be a TsNetStage or plain stage object",
            ));
        }
        let iters = opt_usize_prop(&item, "iterations", 0)?;
        let exagg = opt_f64_prop(&item, "exaggeration", 1.0)?;
        let lambda_kl = opt_f64_prop(&item, "lambdaKl", 1.0)?;
        let lambda_c = opt_f64_prop(&item, "lambdaC", 0.0)?;
        let lambda_r = opt_f64_prop(&item, "lambdaR", 0.0)?;
        let stage = TsNetStage::new()
            .with_iterations(iters)
            .with_exaggeration(exagg)
            .with_lambda_kl(lambda_kl)
            .with_lambda_c(lambda_c)
            .with_lambda_r(lambda_r);
        stage
            .validate()
            .map_err(crate::js_convert::core_err_to_js)?;
        out.push(stage);
    }
    Ok(out)
}

pub(crate) fn parse_sampling_mode_from_options(
    options: &JsValue,
    default: TwoHopSamplingMode,
) -> Result<TwoHopSamplingMode, JsValue> {
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
            .ok_or_else(|| invalid_input_err("options.twoStepRandomWalk must be a boolean"))?;
        return Ok(if rw {
            TwoHopSamplingMode::TwoStepRandomWalk
        } else {
            TwoHopSamplingMode::DegreeOrderedHubSuppressed
        });
    }
    Ok(default)
}

pub(crate) fn parse_tsnet_options(
    options: &JsValue,
    ctx: &str,
) -> Result<(TsNetConfig, JsTargetKind, TsNetBackendChoice), JsValue> {
    let mut cfg = TsNetConfig::pure_2hop();
    let has_opts = check_options_arg(options, ctx)?;
    let target = if has_opts {
        opt_target_prop(options)?
    } else {
        JsTargetKind::Layout2d
    };
    let backend_opt = if has_opts {
        opt_string_prop(options, "backend")?
    } else {
        None
    };
    let exact_threshold = if has_opts {
        opt_usize_prop(options, "exactThreshold", DEFAULT_AUTO_EXACT_THRESHOLD)?
    } else {
        DEFAULT_AUTO_EXACT_THRESHOLD
    };
    let theta = if has_opts {
        opt_f64_prop(options, "theta", NeighborBarnesHut::default().theta)?
    } else {
        NeighborBarnesHut::default().theta
    };
    let ibfft_cfg = if has_opts {
        parse_ibfft_from_options(options)?
    } else {
        Default::default()
    };

    let backend_choice = match (target, backend_opt) {
        (JsTargetKind::Layout2d, None) => TsNetBackendChoice::Dim2(TsNetBackend2d::Auto(
            Auto2d::new()
                .with_exact_threshold(exact_threshold)
                .with_ibfft(ibfft_cfg),
        )),
        (JsTargetKind::Layout2d, Some(ref b)) => match normalize_enum_token(b)?.as_str() {
            "auto" | "auto2d" => TsNetBackendChoice::Dim2(TsNetBackend2d::Auto(
                Auto2d::new()
                    .with_exact_threshold(exact_threshold)
                    .with_ibfft(ibfft_cfg),
            )),
            "exact" => TsNetBackendChoice::Dim2(TsNetBackend2d::Exact(NeighborExact::new())),
            "barneshut" | "bh" => {
                TsNetBackendChoice::Dim2(TsNetBackend2d::BarnesHut(NeighborBarnesHut::new(theta)))
            }
            "fitfft" | "fft" => TsNetBackendChoice::Dim2(TsNetBackend2d::FitFft(
                FitFft::new().with_ibfft(ibfft_cfg),
            )),
            "linearfft" => TsNetBackendChoice::Dim2(TsNetBackend2d::LinearFft(
                LinearFft::new().with_ibfft(ibfft_cfg),
            )),
            "auto3d" => {
                return Err(unsupported_err(
                    "TsNetSolver target 'layout2d' does not support 'auto3d'; use 'auto2d'",
                ));
            }
            _ => {
                return Err(invalid_input_err(format!(
                    "unknown TsNetSolver 2D backend {b:?}; expected 'auto2d', 'exact', 'barnesHut', 'fitFft', or 'linearFft'"
                )));
            }
        },
        (JsTargetKind::Layout3d, None) => TsNetBackendChoice::Dim3(TsNetBackend3d::Auto(
            Auto3d::new()
                .with_exact_threshold(exact_threshold)
                .with_theta(theta),
        )),
        (JsTargetKind::Layout3d, Some(ref b)) => match normalize_enum_token(b)?.as_str() {
            "auto" | "auto3d" => TsNetBackendChoice::Dim3(TsNetBackend3d::Auto(
                Auto3d::new()
                    .with_exact_threshold(exact_threshold)
                    .with_theta(theta),
            )),
            "exact" => TsNetBackendChoice::Dim3(TsNetBackend3d::Exact(NeighborExact::new())),
            "barneshut" | "bh" => {
                TsNetBackendChoice::Dim3(TsNetBackend3d::BarnesHut(NeighborBarnesHut::new(theta)))
            }
            "fitfft" | "fft" | "linearfft" | "auto2d" => {
                return Err(unsupported_err(format!(
                    "TsNetSolver target 'layout3d' does not support 2D FFT backend {b:?}"
                )));
            }
            _ => {
                return Err(invalid_input_err(format!(
                    "unknown TsNetSolver 3D backend {b:?}; expected 'auto3d', 'exact', or 'barnesHut'"
                )));
            }
        },
        (JsTargetKind::LayoutNd, None) => TsNetBackendChoice::DimN(NeighborExact::new()),
        (JsTargetKind::LayoutNd, Some(ref b)) => match normalize_enum_token(b)?.as_str() {
            "exact" => TsNetBackendChoice::DimN(NeighborExact::new()),
            "barneshut" | "bh" | "fitfft" | "fft" | "linearfft" | "auto" | "auto2d" | "auto3d" => {
                return Err(unsupported_err(format!(
                    "TsNetSolver target 'layoutNd' only supports backend 'exact', got {b:?}"
                )));
            }
            _ => {
                return Err(invalid_input_err(format!(
                    "unknown TsNetSolver N-D backend {b:?}; expected 'exact'"
                )));
            }
        },
        (JsTargetKind::Manifold2d, None) => TsNetBackendChoice::Manifold(NeighborExact::new()),
        (JsTargetKind::Manifold2d, Some(ref b)) => match normalize_enum_token(b)?.as_str() {
            "exact" => TsNetBackendChoice::Manifold(NeighborExact::new()),
            "barneshut" | "bh" | "fitfft" | "fft" | "linearfft" | "auto" | "auto2d" | "auto3d" => {
                return Err(unsupported_err(format!(
                    "TsNetSolver target 'manifold2d' only supports backend 'exact', got {b:?}"
                )));
            }
            _ => {
                return Err(invalid_input_err(format!(
                    "unknown TsNetSolver manifold backend {b:?}; expected 'exact'"
                )));
            }
        },
    };

    if has_opts {
        cfg.perplexity_c = opt_f64_prop(options, "perplexityC", cfg.perplexity_c)?;
        cfg.min_perplexity = opt_f64_prop(options, "minPerplexity", cfg.min_perplexity)?;
        cfg.max_perplexity = opt_f64_prop(options, "maxPerplexity", cfg.max_perplexity)?;
        cfg.power = opt_f64_prop(options, "power", cfg.power)?;
        cfg.sigma_iters = opt_usize_prop(options, "sigmaIters", cfg.sigma_iters)?;
        cfg.sigma_tolerance = opt_f64_prop(options, "sigmaTolerance", cfg.sigma_tolerance)?;
        if opt_bool_prop(options, "rowNormalize", false)? {
            cfg.symmetrization = SymmetrizationMode::RowNormalized;
        }
        cfg.jacobi_precondition =
            opt_bool_prop(options, "jacobiPrecondition", cfg.jacobi_precondition)?;

        let stages_val = get_prop(options, "stages")?;
        if !stages_val.is_undefined() && !stages_val.is_null() {
            cfg.stages = parse_stages_array(&stages_val)?;
        } else {
            let si_val = get_prop(options, "stageIterations")?;
            let iters = parse_stage_iterations_val(&si_val, [250, 250, 500])?;
            let exagg = opt_f64_prop(options, "exaggeration", 4.0)?;
            let lc1 = opt_f64_prop(options, "lambdaCStage1", 0.1)?;
            let lc2 = opt_f64_prop(options, "lambdaCStage2", 0.01)?;
            let lc3 = opt_f64_prop(options, "lambdaCStage3", 0.0)?;
            let lr3 = opt_f64_prop(options, "lambdaRStage3", 0.0)?;
            cfg.stages = vec![
                TsNetStage::new()
                    .with_iterations(iters[0])
                    .with_exaggeration(exagg)
                    .with_lambda_kl(1.0)
                    .with_lambda_c(lc1)
                    .with_lambda_r(0.0),
                TsNetStage::new()
                    .with_iterations(iters[1])
                    .with_exaggeration(1.0)
                    .with_lambda_kl(1.0)
                    .with_lambda_c(lc2)
                    .with_lambda_r(0.0),
                TsNetStage::new()
                    .with_iterations(iters[2])
                    .with_exaggeration(1.0)
                    .with_lambda_kl(1.0)
                    .with_lambda_c(lc3)
                    .with_lambda_r(lr3),
            ];
        }
        cfg.learning_rate = opt_maybe_f64_prop(options, "learningRate")?.or(cfg.learning_rate);
        cfg.learning_rate_divisor =
            opt_f64_prop(options, "learningRateDivisor", cfg.learning_rate_divisor)?;
        cfg.momentum = opt_f64_prop(options, "momentum", cfg.momentum)?;
        cfg.epsilon_r = opt_f64_prop(options, "epsilonR", cfg.epsilon_r)?;
        cfg.seed = opt_seed_prop(options)?;
    }

    Ok((cfg, target, backend_choice))
}
