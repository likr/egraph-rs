//! Shared JavaScript plain-object option parsers, enum token normalizers, and solver config builders.

use crate::js_convert::{
    core_err_to_js, parse_non_negative_usize, parse_opt_usize_param, parse_optional_usize,
    parse_seed_u64,
};
use egraph::core::SolverStatus;
use egraph::layout::{FieldRepulsionBackend, PacmapGraphConfig};
use egraph::linalg::{
    AmgPcgConfig, EffectiveResistanceConfig, LaplacianKind, ResistanceCurvatureConfig,
};
use egraph::neighborhood::DogVaradhanConfig;
use egraph::spatial::InterpolationOrder;
use wasm_bindgen::{JsCast, prelude::*};

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
        let pi_val = get_prop(options, "phaseIterations")?;
        if !pi_val.is_undefined() && !pi_val.is_null() {
            let mut vals = Vec::with_capacity(3);
            if let Some(arr) = pi_val.dyn_ref::<js_sys::Uint32Array>() {
                for v in arr.to_vec() {
                    vals.push(v as usize);
                }
            } else if !pi_val.is_string() && pi_val.is_object() {
                let iter = js_sys::try_iter(&pi_val)
                    .map_err(|err| {
                        JsError::new(&format!(
                            "failed to iterate options.phaseIterations: {err:?}"
                        ))
                    })?
                    .ok_or_else(|| {
                        JsError::new("options.phaseIterations must be a 3-element integer array")
                    })?;
                for item_res in iter {
                    let item = item_res.map_err(|err| {
                        JsError::new(&format!("failed to read options.phaseIterations: {err:?}"))
                    })?;
                    let num = item.as_f64().ok_or_else(|| {
                        JsError::new("options.phaseIterations entries must be integers")
                    })?;
                    vals.push(parse_non_negative_usize(num, "phaseIterations")?);
                }
            } else {
                return Err(JsError::new(
                    "options.phaseIterations must be a 3-element integer array",
                ));
            }
            if vals.len() != 3 {
                return Err(JsError::new(&format!(
                    "options.phaseIterations must have length 3, got {}",
                    vals.len()
                )));
            }
            cfg.phase_iterations = [vals[0], vals[1], vals[2]];
        }
        cfg.learning_rate = opt_f64_prop(options, "learningRate", cfg.learning_rate)?;
        cfg.keep_all_1hop = opt_bool_prop(options, "keepAll1hop", cfg.keep_all_1hop)?;
        cfg.max_1hop_per_node = opt_maybe_usize_prop(options, "max1hopPerNode")?;
        cfg.degree_weight_beta = opt_f64_prop(options, "degreeWeightBeta", cfg.degree_weight_beta)?;
        cfg.num_landmarks = opt_usize_prop(options, "numLandmarks", cfg.num_landmarks)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    Ok(cfg)
}

pub(crate) fn parse_amg_pcg_config(
    options: &JsValue,
) -> Result<(AmgPcgConfig, Option<Vec<f64>>), JsError> {
    let mut cfg = AmgPcgConfig::default();
    let mut near_nullspace: Option<Vec<f64>> = None;
    if !options.is_undefined() && !options.is_null() {
        if !is_plain_js_object(options) {
            return Err(JsError::new(
                "AmgPcgSolver: options must be a plain object, null, or undefined",
            ));
        }
        cfg.rel_tol = opt_f64_prop(options, "relTol", cfg.rel_tol)?;
        cfg.abs_tol = opt_f64_prop(options, "absTol", cfg.abs_tol)?;
        cfg.max_iterations = opt_usize_prop(options, "maxIterations", cfg.max_iterations)?;
        cfg.theta = opt_f64_prop(options, "theta", cfg.theta)?;
        cfg.max_levels = opt_usize_prop(options, "maxLevels", cfg.max_levels)?;
        cfg.max_coarse_size = opt_usize_prop(options, "maxCoarseSize", cfg.max_coarse_size)?;
        cfg.pre_sweeps = opt_usize_prop(options, "preSweeps", cfg.pre_sweeps)?;
        cfg.post_sweeps = opt_usize_prop(options, "postSweeps", cfg.post_sweeps)?;
        cfg.max_agg_size = opt_usize_prop(options, "maxAggSize", cfg.max_agg_size)?;
        cfg.diag_eps = opt_f64_prop(options, "diagEps", cfg.diag_eps)?;
        cfg.eig_eps = opt_f64_prop(options, "eigEps", cfg.eig_eps)?;
        cfg.sym_tol = opt_f64_prop(options, "symTol", cfg.sym_tol)?;
        cfg.seed = opt_seed_prop(options)?;
        cfg.randomize_ties = opt_bool_prop(options, "randomizeTies", cfg.randomize_ties)?;

        let ns_val = get_prop(options, "nearNullspace")?;
        if !ns_val.is_undefined() && !ns_val.is_null() {
            if let Some(arr) = ns_val.dyn_ref::<js_sys::Float64Array>() {
                near_nullspace = Some(arr.to_vec());
            } else if let Some(arr) = ns_val.dyn_ref::<js_sys::Float32Array>() {
                near_nullspace = Some(arr.to_vec().into_iter().map(f64::from).collect());
            } else if !ns_val.is_string() && ns_val.is_object() {
                let iter = js_sys::try_iter(&ns_val)
                    .map_err(|err| {
                        JsError::new(&format!("failed to iterate nearNullspace: {err:?}"))
                    })?
                    .ok_or_else(|| {
                        JsError::new("options.nearNullspace must be a Float64Array or number array")
                    })?;
                let mut vec = Vec::new();
                for item_res in iter {
                    let item = item_res.map_err(|err| {
                        JsError::new(&format!("failed to read nearNullspace: {err:?}"))
                    })?;
                    let num = item.as_f64().ok_or_else(|| {
                        JsError::new("options.nearNullspace entries must be numbers")
                    })?;
                    vec.push(num);
                }
                near_nullspace = Some(vec);
            } else {
                return Err(JsError::new(
                    "options.nearNullspace must be a Float64Array or number array",
                ));
            }
        }
    }
    cfg.validate().map_err(core_err_to_js)?;
    Ok((cfg, near_nullspace))
}

pub(crate) fn parse_amg_laplacian_kind(raw: &str) -> Result<LaplacianKind, JsError> {
    match normalize_enum_token(raw).as_str() {
        "combinatorial" | "unnormalized" | "standard" => Ok(LaplacianKind::Combinatorial),
        "symmetricnormalized" | "symnorm" | "normalized" | "sym" => {
            Ok(LaplacianKind::SymmetricNormalized)
        }
        _ => Err(JsError::new(&format!(
            "unknown Laplacian kind {raw:?}; expected 'combinatorial' or 'symmetric_normalized'"
        ))),
    }
}

fn opt_maybe_bool_prop(obj: &JsValue, key: &str) -> Result<Option<bool>, JsError> {
    let val = get_prop(obj, key)?;
    if val.is_undefined() || val.is_null() {
        return Ok(None);
    }
    val.as_bool()
        .map(Some)
        .ok_or_else(|| JsError::new(&format!("options.{key} must be a boolean")))
}

pub(crate) fn parse_effective_resistance_options(
    options: &JsValue,
) -> Result<EffectiveResistanceConfig, JsError> {
    let mut cfg = EffectiveResistanceConfig::default();
    if !options.is_undefined() && !options.is_null() {
        if !is_plain_js_object(options) {
            return Err(JsError::new(
                "computeEffectiveResistances: options must be a plain object, null, or undefined",
            ));
        }
        cfg.jl_dim = opt_usize_prop(options, "jlDim", cfg.jl_dim)?;
        cfg.seed = opt_seed_prop(options)?;
        cfg.weights_are_lengths =
            opt_bool_prop(options, "weightsAreLengths", cfg.weights_are_lengths)?;
        cfg.exact = opt_maybe_bool_prop(options, "exact")?;
        cfg.exact_max_nodes = opt_usize_prop(options, "exactMaxNodes", cfg.exact_max_nodes)?;
        cfg.eig_eps = opt_f64_prop(options, "eigEps", cfg.eig_eps)?;
        cfg.amg_config.rel_tol = opt_f64_prop(options, "relTol", cfg.amg_config.rel_tol)?;
        cfg.amg_config.max_iterations =
            opt_usize_prop(options, "maxIterations", cfg.amg_config.max_iterations)?;
        cfg.amg_config.eig_eps = cfg.eig_eps;
        cfg.amg_config.seed = cfg.seed;
    }
    cfg.validate().map_err(core_err_to_js)?;
    Ok(cfg)
}

pub(crate) fn parse_resistance_curvature_options(
    options: &JsValue,
) -> Result<ResistanceCurvatureConfig, JsError> {
    let mut cfg = ResistanceCurvatureConfig::default();
    if !options.is_undefined() && !options.is_null() {
        if !is_plain_js_object(options) {
            return Err(JsError::new(
                "resistanceCurvatureFlow: options must be a plain object, null, or undefined",
            ));
        }
        cfg.steps = opt_usize_prop(options, "steps", cfg.steps)?;
        cfg.eta = opt_f64_prop(options, "eta", cfg.eta)?;
        cfg.kappa_max = opt_f64_prop(options, "kappaMax", cfg.kappa_max)?;
        cfg.min_edge_length = opt_f64_prop(options, "minEdgeLength", cfg.min_edge_length)?;
        cfg.hyperbolic_threshold =
            opt_f64_prop(options, "hyperbolicThreshold", cfg.hyperbolic_threshold)?;
        cfg.spherical_threshold =
            opt_f64_prop(options, "sphericalThreshold", cfg.spherical_threshold)?;
        cfg.torus_std_threshold =
            opt_f64_prop(options, "torusStdThreshold", cfg.torus_std_threshold)?;
        cfg.resistance = parse_effective_resistance_options(options)?;
    }
    cfg.validate().map_err(core_err_to_js)?;
    Ok(cfg)
}

pub(crate) fn parse_field_repulsion_backend(raw: &str) -> Result<FieldRepulsionBackend, JsError> {
    match normalize_enum_token(raw).as_str() {
        "ibfft" | "fft" => Ok(FieldRepulsionBackend::IbFft),
        "barneshut" | "bh" => Ok(FieldRepulsionBackend::BarnesHut),
        "exact" => Ok(FieldRepulsionBackend::Exact),
        _ => Err(JsError::new(&format!(
            "unknown FieldRepulsionBackend {raw:?}; expected 'ibfft', 'barnes_hut', or 'exact'"
        ))),
    }
}
