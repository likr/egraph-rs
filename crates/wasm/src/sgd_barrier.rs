//! 3-tier Sparse Barrier SGD constructors and diagnostics (`SparseBarrierStats`, `DogStats`).

use egraph::layout::{BarrierSymmetry, SgdSolver, SparseBarrierConfig, SparseBarrierStats};
use egraph::neighborhood::DogVaradhanStats;
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::dog_varadhan::JsDogVaradhanLandmarks;
use crate::js_convert::{
    core_err_to_js, parse_non_negative_usize, parse_opt_usize_param, parse_optional_usize,
    parse_seed_u64,
};
use crate::js_options::{
    is_plain_js_object, normalize_enum_token, opt_bool_prop, opt_f64_prop, opt_maybe_usize_prop,
    opt_seed_prop, opt_string_prop, opt_usize_prop, parse_dog_config_from_options_or_args,
    reject_extra_args_with_options,
};
use crate::linalg_matrix::JsSpectralMomentStats;
use crate::sgd::{JsSgdSolver, parse_sgd_schedule_kind};

fn parse_barrier_symmetry(raw: &str) -> Result<BarrierSymmetry, JsError> {
    match normalize_enum_token(raw).as_str() {
        "symmax" => Ok(BarrierSymmetry::SymMax),
        "symavg" => Ok(BarrierSymmetry::SymAvg),
        _ => Err(JsError::new(&format!(
            "unknown BarrierSymmetry {raw:?}; expected 'sym_max' or 'sym_avg'"
        ))),
    }
}

/// Diagnostics emitted by `SgdSolver.sparseBarrierFromLandmarks` and `SgdSolver.sparseBarrierDog`.
#[wasm_bindgen(js_name = SparseBarrierStats)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsSparseBarrierStats {
    inner: SparseBarrierStats,
}

#[wasm_bindgen(js_class = SparseBarrierStats)]
impl JsSparseBarrierStats {
    /// Returns the count of Tier 1 1-hop pairs.
    #[wasm_bindgen(getter, js_name = nPairs1hop)]
    pub fn n_pairs_1hop(&self) -> usize {
        self.inner.n_pairs_1hop
    }

    /// Returns the count of Tier 2 2-hop pairs.
    #[wasm_bindgen(getter, js_name = nPairs2hop)]
    pub fn n_pairs_2hop(&self) -> usize {
        self.inner.n_pairs_2hop
    }

    /// Returns the count of Tier 3 non-2-hop barrier pairs.
    #[wasm_bindgen(getter, js_name = nPairsNon2)]
    pub fn n_pairs_non2(&self) -> usize {
        self.inner.n_pairs_non2
    }

    /// Returns the count of one-sided hinge pairs.
    #[wasm_bindgen(getter, js_name = nPairsHinge)]
    pub fn n_pairs_hinge(&self) -> usize {
        self.inner.n_pairs_hinge
    }

    /// Returns the count of two-sided spring pairs.
    #[wasm_bindgen(getter, js_name = nPairsTwoSided)]
    pub fn n_pairs_two_sided(&self) -> usize {
        self.inner.n_pairs_two_sided
    }

    /// Returns the adaptive or explicit perplexity value.
    #[wasm_bindgen(getter, js_name = perplexityVal)]
    pub fn perplexity_val(&self) -> f64 {
        self.inner.perplexity_val
    }

    /// Returns the per-node negative sample budget used.
    #[wasm_bindgen(getter, js_name = perNodeNegUsed)]
    pub fn per_node_neg_used(&self) -> usize {
        self.inner.per_node_neg_used
    }

    /// Returns `"adaptive_2hop"` or `"explicit"`.
    #[wasm_bindgen(getter, js_name = perplexitySource)]
    pub fn perplexity_source(&self) -> String {
        self.inner.perplexity_source.to_owned()
    }

    /// Returns the applied relative distance floor.
    #[wasm_bindgen(getter, js_name = dFloor)]
    pub fn d_floor(&self) -> f64 {
        self.inner.d_floor
    }

    /// Returns the effective non-2-hop repulsion multiplier $\gamma_{\mathrm{eff}}$.
    #[wasm_bindgen(getter, js_name = gammaEff)]
    pub fn gamma_eff(&self) -> f64 {
        self.inner.gamma_eff
    }
}

/// Regularized DoG Varadhan landmark diagnostics returned by `SgdSolver.dogStats()`.
#[wasm_bindgen(js_name = DogStats)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsDogStats {
    inner: DogVaradhanStats,
}

#[wasm_bindgen(js_class = DogStats)]
impl JsDogStats {
    #[wasm_bindgen(getter, js_name = momentStats)]
    pub fn moment_stats(&self) -> JsSpectralMomentStats {
        JsSpectralMomentStats::from_inner(self.inner.moment_stats)
    }

    #[wasm_bindgen(getter)]
    pub fn t1(&self) -> f64 {
        self.inner.t1
    }

    #[wasm_bindgen(getter)]
    pub fn t2(&self) -> f64 {
        self.inner.t2
    }

    #[wasm_bindgen(getter)]
    pub fn alpha(&self) -> f64 {
        self.inner.alpha
    }

    #[wasm_bindgen(getter)]
    pub fn degree(&self) -> usize {
        self.inner.degree
    }

    #[wasm_bindgen(getter, js_name = deltaPsd)]
    pub fn delta_psd(&self) -> f64 {
        self.inner.delta_psd
    }

    #[wasm_bindgen(getter, js_name = hMin)]
    pub fn h_min(&self) -> f64 {
        self.inner.h_min
    }

    #[wasm_bindgen(getter, js_name = kMin)]
    pub fn k_min(&self) -> f64 {
        self.inner.k_min
    }

    #[wasm_bindgen(getter, js_name = kMed)]
    pub fn k_med(&self) -> f64 {
        self.inner.k_med
    }

    #[wasm_bindgen(getter, js_name = betaAuto)]
    pub fn beta_auto(&self) -> f64 {
        self.inner.beta_auto
    }

    #[wasm_bindgen(getter, js_name = betaEff)]
    pub fn beta_eff(&self) -> f64 {
        self.inner.beta_eff
    }

    #[wasm_bindgen(getter, js_name = numLandmarks)]
    pub fn num_landmarks(&self) -> usize {
        self.inner.n_landmarks
    }

    #[wasm_bindgen(getter, js_name = tCoarse)]
    pub fn t_coarse(&self) -> f64 {
        self.inner.t_coarse
    }

    #[wasm_bindgen(getter, js_name = usedCoarseScaffold)]
    pub fn used_coarse_scaffold(&self) -> bool {
        self.inner.used_coarse_scaffold
    }
}

#[allow(clippy::too_many_arguments)]
fn parse_sparse_barrier_config(
    max_2hop_or_options: &JsValue,
    hinge_repulsion: Option<bool>,
    delta_rel: Option<f64>,
    barrier_sym: Option<String>,
    omega_1: Option<f64>,
    omega_2: Option<f64>,
    gamma_non2: Option<f64>,
    ht_repulsion: Option<bool>,
    min_ratio: Option<f64>,
    eps_dist: Option<f64>,
    neg_per_node: Option<f64>,
    num_pivots: Option<f64>,
    core_pivots: Option<bool>,
    t_max: Option<f64>,
    eps: Option<f64>,
    kind: Option<String>,
    step_scale: Option<f64>,
    seed: Option<f64>,
    has_dog_extra: bool,
) -> Result<SparseBarrierConfig, JsError> {
    let mut cfg = SparseBarrierConfig::default();
    if is_plain_js_object(max_2hop_or_options) {
        let has_extra = has_dog_extra
            || hinge_repulsion.is_some()
            || delta_rel.is_some()
            || barrier_sym.is_some()
            || omega_1.is_some()
            || omega_2.is_some()
            || gamma_non2.is_some()
            || ht_repulsion.is_some()
            || min_ratio.is_some()
            || eps_dist.is_some()
            || neg_per_node.is_some()
            || num_pivots.is_some()
            || core_pivots.is_some()
            || t_max.is_some()
            || eps.is_some()
            || kind.is_some()
            || step_scale.is_some()
            || seed.is_some();
        reject_extra_args_with_options(has_extra)?;

        cfg.max_2hop_per_node =
            opt_usize_prop(max_2hop_or_options, "max2hopPerNode", cfg.max_2hop_per_node)?;
        cfg.hinge_repulsion =
            opt_bool_prop(max_2hop_or_options, "hingeRepulsion", cfg.hinge_repulsion)?;
        cfg.delta_rel = opt_f64_prop(max_2hop_or_options, "deltaRel", cfg.delta_rel)?;
        if let Some(sym_str) = opt_string_prop(max_2hop_or_options, "barrierSym")? {
            cfg.barrier_sym = parse_barrier_symmetry(&sym_str)?;
        }
        cfg.omega_1 = opt_f64_prop(max_2hop_or_options, "omega1", cfg.omega_1)?;
        cfg.omega_2 = opt_f64_prop(max_2hop_or_options, "omega2", cfg.omega_2)?;
        cfg.gamma_non2 = opt_f64_prop(max_2hop_or_options, "gammaNon2", cfg.gamma_non2)?;
        cfg.ht_repulsion = opt_bool_prop(max_2hop_or_options, "htRepulsion", cfg.ht_repulsion)?;
        cfg.min_ratio = opt_f64_prop(max_2hop_or_options, "minRatio", cfg.min_ratio)?;
        cfg.eps_dist = opt_f64_prop(max_2hop_or_options, "epsDist", cfg.eps_dist)?;
        cfg.neg_per_node = opt_maybe_usize_prop(max_2hop_or_options, "negPerNode")?;
        cfg.num_pivots = opt_usize_prop(max_2hop_or_options, "numPivots", cfg.num_pivots)?;
        cfg.core_pivots = opt_bool_prop(max_2hop_or_options, "corePivots", cfg.core_pivots)?;
        cfg.degree_weight_beta = opt_f64_prop(
            max_2hop_or_options,
            "degreeWeightBeta",
            cfg.degree_weight_beta,
        )?;
        cfg.schedule.t_max = opt_usize_prop(max_2hop_or_options, "tMax", cfg.schedule.t_max)?;
        cfg.schedule.eps = opt_f64_prop(max_2hop_or_options, "eps", cfg.schedule.eps)?;
        if let Some(k_str) = opt_string_prop(max_2hop_or_options, "kind")? {
            cfg.schedule.kind = parse_sgd_schedule_kind(&k_str)?;
        }
        cfg.schedule.step_scale =
            opt_f64_prop(max_2hop_or_options, "stepScale", cfg.schedule.step_scale)?;
        cfg.seed = opt_seed_prop(max_2hop_or_options)?;
        return Ok(cfg);
    }

    if !max_2hop_or_options.is_undefined() && !max_2hop_or_options.is_null() {
        let v = max_2hop_or_options.as_f64().ok_or_else(|| {
            JsError::new("max2hopOrOptions must be a non-negative integer, options object, null, or undefined")
        })?;
        cfg.max_2hop_per_node = parse_non_negative_usize(v, "max2hopPerNode")?;
    }
    cfg.hinge_repulsion = hinge_repulsion.unwrap_or(cfg.hinge_repulsion);
    cfg.delta_rel = delta_rel.unwrap_or(cfg.delta_rel);
    if let Some(sym_str) = barrier_sym {
        cfg.barrier_sym = parse_barrier_symmetry(&sym_str)?;
    }
    cfg.omega_1 = omega_1.unwrap_or(cfg.omega_1);
    cfg.omega_2 = omega_2.unwrap_or(cfg.omega_2);
    cfg.gamma_non2 = gamma_non2.unwrap_or(cfg.gamma_non2);
    cfg.ht_repulsion = ht_repulsion.unwrap_or(cfg.ht_repulsion);
    cfg.min_ratio = min_ratio.unwrap_or(cfg.min_ratio);
    cfg.eps_dist = eps_dist.unwrap_or(cfg.eps_dist);
    cfg.neg_per_node = parse_opt_usize_param(neg_per_node, "negPerNode")?;
    cfg.num_pivots = parse_optional_usize(num_pivots, cfg.num_pivots, "numPivots")?;
    cfg.core_pivots = core_pivots.unwrap_or(cfg.core_pivots);
    cfg.schedule.t_max = parse_optional_usize(t_max, cfg.schedule.t_max, "tMax")?;
    cfg.schedule.eps = eps.unwrap_or(cfg.schedule.eps);
    if let Some(k_str) = kind {
        cfg.schedule.kind = parse_sgd_schedule_kind(&k_str)?;
    }
    cfg.schedule.step_scale = step_scale.unwrap_or(cfg.schedule.step_scale);
    cfg.seed = parse_seed_u64(seed)?;
    Ok(cfg)
}

#[wasm_bindgen(js_class = SgdSolver)]
impl JsSgdSolver {
    /// Constructs a 3-tier Sparse Barrier SGD solver from precomputed `DogVaradhanLandmarks`.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = sparseBarrierFromLandmarks)]
    pub fn sparse_barrier_from_landmarks(
        graph: &JsCsrGraph,
        landmarks: &JsDogVaradhanLandmarks,
        pivots: Option<Vec<u32>>,
        max_2hop_or_options: &JsValue,
        hinge_repulsion: Option<bool>,
        delta_rel: Option<f64>,
        barrier_sym: Option<String>,
        omega_1: Option<f64>,
        omega_2: Option<f64>,
        gamma_non2: Option<f64>,
        ht_repulsion: Option<bool>,
        min_ratio: Option<f64>,
        eps_dist: Option<f64>,
        neg_per_node: Option<f64>,
        num_pivots: Option<f64>,
        core_pivots: Option<bool>,
        t_max: Option<f64>,
        eps: Option<f64>,
        kind: Option<String>,
        step_scale: Option<f64>,
        seed: Option<f64>,
    ) -> Result<JsSgdSolver, JsError> {
        let cfg = parse_sparse_barrier_config(
            max_2hop_or_options,
            hinge_repulsion,
            delta_rel,
            barrier_sym,
            omega_1,
            omega_2,
            gamma_non2,
            ht_repulsion,
            min_ratio,
            eps_dist,
            neg_per_node,
            num_pivots,
            core_pivots,
            t_max,
            eps,
            kind,
            step_scale,
            seed,
            false,
        )?;
        let (inner, stats) = SgdSolver::<f32>::sparse_barrier_from_landmarks(
            graph.inner(),
            landmarks.inner(),
            pivots.as_deref(),
            &cfg,
        )
        .map_err(core_err_to_js)?;
        Ok(Self::from_barrier_parts(inner, Some(stats), None))
    }

    /// Computes regularized Chebyshev DoG Varadhan landmarks and constructs a 3-tier Sparse Barrier SGD solver.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = sparseBarrierDog)]
    pub fn sparse_barrier_dog(
        graph: &JsCsrGraph,
        max_2hop_or_options: &JsValue,
        hinge_repulsion: Option<bool>,
        delta_rel: Option<f64>,
        barrier_sym: Option<String>,
        omega_1: Option<f64>,
        omega_2: Option<f64>,
        gamma_non2: Option<f64>,
        ht_repulsion: Option<bool>,
        min_ratio: Option<f64>,
        eps_dist: Option<f64>,
        neg_per_node: Option<f64>,
        num_pivots: Option<f64>,
        core_pivots: Option<bool>,
        t_max: Option<f64>,
        eps: Option<f64>,
        kind: Option<String>,
        step_scale: Option<f64>,
        seed: Option<f64>,
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
    ) -> Result<JsSgdSolver, JsError> {
        let has_dog_extra = degree.is_some()
            || num_landmarks.is_some()
            || degree_normalize.is_some()
            || p_exp.is_some()
            || eps_psd.is_some()
            || t_coarse.is_some()
            || t1.is_some()
            || t2.is_some()
            || alpha.is_some()
            || beta_floor.is_some()
            || min_degree.is_some()
            || jl_dim.is_some();
        let barrier_cfg = parse_sparse_barrier_config(
            max_2hop_or_options,
            hinge_repulsion,
            delta_rel,
            barrier_sym,
            omega_1,
            omega_2,
            gamma_non2,
            ht_repulsion,
            min_ratio,
            eps_dist,
            neg_per_node,
            num_pivots,
            core_pivots,
            t_max,
            eps,
            kind,
            step_scale,
            seed,
            has_dog_extra,
        )?;
        let dog_cfg = parse_dog_config_from_options_or_args(
            max_2hop_or_options,
            barrier_cfg.seed,
            degree,
            num_landmarks,
            degree_normalize,
            p_exp,
            eps_psd,
            t_coarse,
            t1,
            t2,
            alpha,
            beta_floor,
            min_degree,
            jl_dim,
        )?;
        let (inner, stats, dog_stats) =
            SgdSolver::<f32>::sparse_barrier_dog(graph.inner(), &dog_cfg, &barrier_cfg)
                .map_err(core_err_to_js)?;
        Ok(Self::from_barrier_parts(
            inner,
            Some(stats),
            Some(dog_stats),
        ))
    }

    /// Returns `SparseBarrierStats` when constructed via `sparseBarrierFromLandmarks` or
    /// `sparseBarrierDog`, or `undefined` otherwise.
    pub fn stats(&self) -> Option<JsSparseBarrierStats> {
        self.barrier_stats_inner()
            .map(|inner| JsSparseBarrierStats { inner })
    }

    /// Returns regularized DoG Varadhan landmark diagnostics `DogStats` when constructed via
    /// `sparseBarrierDog`, or `undefined` otherwise.
    #[wasm_bindgen(js_name = dogStats)]
    pub fn dog_stats(&self) -> Option<JsDogStats> {
        self.dog_stats_inner().map(|inner| JsDogStats { inner })
    }
}
