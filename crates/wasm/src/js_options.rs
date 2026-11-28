//! Shared JavaScript plain-object option parsers, enum token normalizers, and solver config builders.

use crate::js_convert::{
    DEFAULT_SEED, core_err_to_js, invalid_input_err, parse_non_negative_usize, parse_seed_u64,
};
use egraph::core::SolverStatus;
use egraph::layout::{FieldRepulsionBackend, PacmapGraphConfig};
use egraph::linalg::{
    AmgPcgConfig, EffectiveResistanceConfig, LaplacianKind, ResistanceCurvatureConfig,
};
use egraph::neighborhood::DogVaradhanConfig;
use egraph::spatial::InterpolationOrder;
use wasm_bindgen::{JsCast, prelude::*};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum JsTargetKind {
    Layout2d,
    Layout3d,
    LayoutNd,
    Manifold2d,
}

pub(crate) fn parse_target_kind(raw: &str) -> Result<JsTargetKind, JsValue> {
    match normalize_enum_token(raw)?.as_str() {
        "layout2d" => Ok(JsTargetKind::Layout2d),
        "layout3d" => Ok(JsTargetKind::Layout3d),
        "layoutnd" => Ok(JsTargetKind::LayoutNd),
        "manifold2d" => Ok(JsTargetKind::Manifold2d),
        _ => Err(invalid_input_err(format!(
            "unknown target {raw:?}; expected 'layout2d', 'layout3d', 'layoutNd', or 'manifold2d'"
        ))),
    }
}

pub(crate) fn opt_target_prop(obj: &JsValue) -> Result<JsTargetKind, JsValue> {
    match opt_string_prop(obj, "target")? {
        None => Ok(JsTargetKind::Layout2d),
        Some(raw) => parse_target_kind(&raw),
    }
}

pub(crate) fn normalize_enum_token(raw: &str) -> Result<String, JsValue> {
    if raw.contains('_') {
        return Err(invalid_input_err(format!(
            "snake_case enum token {raw:?} is not supported; use camelCase"
        )));
    }
    Ok(raw
        .chars()
        .filter(|ch| !matches!(ch, '-' | ' '))
        .map(|ch| ch.to_ascii_lowercase())
        .collect())
}

pub(crate) fn solver_status_str(status: SolverStatus) -> &'static str {
    match status {
        SolverStatus::Initialized => "initialized",
        SolverStatus::Running => "running",
        SolverStatus::Converged => "converged",
        SolverStatus::Completed => "completed",
        SolverStatus::Failed => "failed",
        _ => "failed",
    }
}

pub(crate) fn parse_ibfft_order(val: &JsValue) -> Result<InterpolationOrder, JsValue> {
    if let Some(num) = val.as_f64() {
        let ord = parse_non_negative_usize(num, "interpolationOrder")?;
        return InterpolationOrder::from_order(ord).map_err(core_err_to_js);
    }
    if let Some(raw) = val.as_string() {
        return match normalize_enum_token(&raw)?.as_str() {
            "bilinear" | "1" => Ok(InterpolationOrder::Bilinear),
            "cubic" | "3" => Ok(InterpolationOrder::Cubic),
            _ => Err(invalid_input_err(format!(
                "unknown interpolationOrder {raw:?}; expected 'bilinear' (1) or 'cubic' (3)"
            ))),
        };
    }
    Err(invalid_input_err(
        "interpolationOrder must be 'bilinear', 'cubic', 1, or 3",
    ))
}

pub(crate) fn is_plain_js_object(val: &JsValue) -> bool {
    val.is_object()
        && !js_sys::Array::is_array(val)
        && !js_sys::ArrayBuffer::is_view(val)
        && !val.is_function()
}

pub(crate) fn check_options_arg(options: &JsValue, ctx: &str) -> Result<bool, JsValue> {
    if options.is_undefined() || options.is_null() {
        return Ok(false);
    }
    if is_plain_js_object(options) {
        return Ok(true);
    }
    Err(invalid_input_err(format!(
        "{ctx}: options must be a plain object, null, or undefined"
    )))
}

pub(crate) fn get_prop(obj: &JsValue, key: &str) -> Result<JsValue, JsValue> {
    js_sys::Reflect::get(obj, &JsValue::from_str(key))
        .map_err(|err| invalid_input_err(format!("failed to read options.{key}: {err:?}")))
}

pub(crate) fn opt_usize_prop(obj: &JsValue, key: &str, default: usize) -> Result<usize, JsValue> {
    let val = get_prop(obj, key)?;
    if val.is_undefined() || val.is_null() {
        return Ok(default);
    }
    let num = val
        .as_f64()
        .ok_or_else(|| invalid_input_err(format!("options.{key} must be an integer number")))?;
    parse_non_negative_usize(num, key)
}

pub(crate) fn opt_maybe_usize_prop(obj: &JsValue, key: &str) -> Result<Option<usize>, JsValue> {
    let val = get_prop(obj, key)?;
    if val.is_undefined() || val.is_null() {
        return Ok(None);
    }
    let num = val
        .as_f64()
        .ok_or_else(|| invalid_input_err(format!("options.{key} must be an integer number")))?;
    parse_non_negative_usize(num, key).map(Some)
}

pub(crate) fn opt_f64_prop(obj: &JsValue, key: &str, default: f64) -> Result<f64, JsValue> {
    let val = get_prop(obj, key)?;
    if val.is_undefined() || val.is_null() {
        return Ok(default);
    }
    val.as_f64()
        .ok_or_else(|| invalid_input_err(format!("options.{key} must be a number")))
}

pub(crate) fn opt_maybe_f64_prop(obj: &JsValue, key: &str) -> Result<Option<f64>, JsValue> {
    let val = get_prop(obj, key)?;
    if val.is_undefined() || val.is_null() {
        return Ok(None);
    }
    val.as_f64()
        .map(Some)
        .ok_or_else(|| invalid_input_err(format!("options.{key} must be a number")))
}

pub(crate) fn opt_bool_prop(obj: &JsValue, key: &str, default: bool) -> Result<bool, JsValue> {
    let val = get_prop(obj, key)?;
    if val.is_undefined() || val.is_null() {
        return Ok(default);
    }
    val.as_bool()
        .ok_or_else(|| invalid_input_err(format!("options.{key} must be a boolean")))
}

pub(crate) fn opt_string_prop(obj: &JsValue, key: &str) -> Result<Option<String>, JsValue> {
    let val = get_prop(obj, key)?;
    if val.is_undefined() || val.is_null() {
        return Ok(None);
    }
    val.as_string()
        .map(Some)
        .ok_or_else(|| invalid_input_err(format!("options.{key} must be a string")))
}

pub(crate) fn opt_seed_prop(obj: &JsValue) -> Result<u64, JsValue> {
    let val = get_prop(obj, "seed")?;
    parse_seed_u64(&val)
}

pub(crate) fn opt_maybe_bool_prop(obj: &JsValue, key: &str) -> Result<Option<bool>, JsValue> {
    let val = get_prop(obj, key)?;
    if val.is_undefined() || val.is_null() {
        return Ok(None);
    }
    val.as_bool()
        .map(Some)
        .ok_or_else(|| invalid_input_err(format!("options.{key} must be a boolean")))
}

pub(crate) fn parse_dog_config_from_options(
    options: &JsValue,
    ctx: &str,
) -> Result<DogVaradhanConfig, JsValue> {
    let mut cfg = DogVaradhanConfig::default().with_seed(DEFAULT_SEED);
    if check_options_arg(options, ctx)? {
        cfg.seed = opt_seed_prop(options)?;

        let deg_primary = opt_maybe_usize_prop(options, "degree")?;
        let deg_alt = opt_maybe_usize_prop(options, "dogDegree")?;
        cfg.degree = deg_primary.or(deg_alt);

        let lm_primary = opt_maybe_usize_prop(options, "numLandmarks")?;
        let lm_alt = opt_maybe_usize_prop(options, "dogNumLandmarks")?;
        cfg.n_landmarks = lm_primary.or(lm_alt).unwrap_or(cfg.n_landmarks);

        cfg.min_degree = opt_usize_prop(options, "minDegree", cfg.min_degree)?;
        cfg.jl_dim = opt_usize_prop(options, "jlDim", cfg.jl_dim)?;

        let dn_val = opt_maybe_bool_prop(options, "degreeNormalize")?;
        let dn_alt = opt_maybe_bool_prop(options, "dogDegreeNormalize")?;
        if let Some(dn) = dn_val.or(dn_alt) {
            cfg.degree_normalize = dn;
        }

        let p_primary = opt_maybe_f64_prop(options, "pExp")?;
        let p_alt = opt_maybe_f64_prop(options, "dogPExp")?;
        cfg.p_exp = p_primary.or(p_alt).unwrap_or(cfg.p_exp);

        let psd_primary = opt_maybe_f64_prop(options, "epsPsd")?;
        let psd_alt = opt_maybe_f64_prop(options, "dogEpsPsd")?;
        cfg.eps_psd = psd_primary.or(psd_alt).unwrap_or(cfg.eps_psd);

        let tc_primary = opt_maybe_f64_prop(options, "tCoarse")?;
        let tc_alt = opt_maybe_f64_prop(options, "dogTCoarse")?;
        cfg.t_coarse = tc_primary.or(tc_alt).unwrap_or(cfg.t_coarse);

        cfg.t1 = opt_maybe_f64_prop(options, "t1")?.or(opt_maybe_f64_prop(options, "dogT1")?);
        cfg.t2 = opt_maybe_f64_prop(options, "t2")?.or(opt_maybe_f64_prop(options, "dogT2")?);
        cfg.alpha =
            opt_maybe_f64_prop(options, "alpha")?.or(opt_maybe_f64_prop(options, "dogAlpha")?);
        cfg.beta_floor = opt_maybe_f64_prop(options, "betaFloor")?
            .or(opt_maybe_f64_prop(options, "dogBetaFloor")?);
    }
    Ok(cfg)
}

pub(crate) fn parse_pacmap_graph_config(
    options: &JsValue,
    ctx: &str,
) -> Result<PacmapGraphConfig, JsValue> {
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
                        invalid_input_err(format!(
                            "failed to iterate options.phaseIterations: {err:?}"
                        ))
                    })?
                    .ok_or_else(|| {
                        invalid_input_err(
                            "options.phaseIterations must be a 3-element integer array",
                        )
                    })?;
                for item_res in iter {
                    let item = item_res.map_err(|err| {
                        invalid_input_err(format!(
                            "failed to read options.phaseIterations: {err:?}"
                        ))
                    })?;
                    let num = item.as_f64().ok_or_else(|| {
                        invalid_input_err("options.phaseIterations entries must be integers")
                    })?;
                    vals.push(parse_non_negative_usize(num, "phaseIterations")?);
                }
            } else {
                return Err(invalid_input_err(
                    "options.phaseIterations must be a 3-element integer array",
                ));
            }
            if vals.len() != 3 {
                return Err(invalid_input_err(format!(
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
) -> Result<(AmgPcgConfig, Option<Vec<f64>>), JsValue> {
    let mut cfg = AmgPcgConfig::default();
    let mut near_nullspace: Option<Vec<f64>> = None;
    if check_options_arg(options, "AmgPcgSolver")? {
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
                        invalid_input_err(format!("failed to iterate nearNullspace: {err:?}"))
                    })?
                    .ok_or_else(|| {
                        invalid_input_err(
                            "options.nearNullspace must be a Float64Array or number array",
                        )
                    })?;
                let mut vec = Vec::new();
                for item_res in iter {
                    let item = item_res.map_err(|err| {
                        invalid_input_err(format!("failed to read nearNullspace: {err:?}"))
                    })?;
                    let num = item.as_f64().ok_or_else(|| {
                        invalid_input_err("options.nearNullspace entries must be numbers")
                    })?;
                    vec.push(num);
                }
                near_nullspace = Some(vec);
            } else {
                return Err(invalid_input_err(
                    "options.nearNullspace must be a Float64Array or number array",
                ));
            }
        }
    }
    cfg.validate().map_err(core_err_to_js)?;
    Ok((cfg, near_nullspace))
}

pub(crate) fn parse_amg_laplacian_kind(raw: &str) -> Result<LaplacianKind, JsValue> {
    match normalize_enum_token(raw)?.as_str() {
        "combinatorial" | "unnormalized" | "standard" => Ok(LaplacianKind::Combinatorial),
        "symmetricnormalized" | "symnorm" | "normalized" | "sym" => {
            Ok(LaplacianKind::SymmetricNormalized)
        }
        _ => Err(invalid_input_err(format!(
            "unknown Laplacian kind {raw:?}; expected 'combinatorial' or 'symmetricNormalized'"
        ))),
    }
}

pub(crate) fn parse_effective_resistance_options(
    options: &JsValue,
) -> Result<EffectiveResistanceConfig, JsValue> {
    let mut cfg = EffectiveResistanceConfig::default();
    if check_options_arg(options, "computeEffectiveResistances")? {
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
) -> Result<ResistanceCurvatureConfig, JsValue> {
    let mut cfg = ResistanceCurvatureConfig::default();
    if check_options_arg(options, "resistanceCurvatureFlow")? {
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

pub(crate) fn parse_field_repulsion_backend(raw: &str) -> Result<FieldRepulsionBackend, JsValue> {
    match normalize_enum_token(raw)?.as_str() {
        "ibfft" | "fft" => Ok(FieldRepulsionBackend::IbFft),
        "barneshut" | "bh" => Ok(FieldRepulsionBackend::BarnesHut),
        "exact" => Ok(FieldRepulsionBackend::Exact),
        _ => Err(invalid_input_err(format!(
            "unknown FieldRepulsionBackend {raw:?}; expected 'ibFft', 'barnesHut', or 'exact'"
        ))),
    }
}
