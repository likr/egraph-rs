//! WebAssembly bindings for 1+2-hop neighborhood extraction, adaptive perplexity, $k$-NN graphs,
//! and sparse probability symmetrization from `egraph-neighborhood`.

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    core_err_to_js, invalid_input_err, parse_non_negative_usize, parse_optional_usize,
    parse_seed_u64,
};
use crate::js_options::{
    check_options_arg, normalize_enum_token, opt_bool_prop, opt_f64_prop, opt_maybe_usize_prop,
    opt_seed_prop, opt_string_prop, opt_usize_prop,
};
use crate::neighborhood_types::{
    JsHybridKnnResult, JsKnnGraph, JsSparse12HopPairs, JsSparseProbabilities, JsTwoHopCsr,
    JsTwoHopKnnResult,
};
use egraph::core::CsrGraph;
use egraph::neighborhood::{
    DEFAULT_MAX_PERPLEXITY, DEFAULT_MEDIAN_MIN_PERPLEXITY, DEFAULT_MEDIAN_PERPLEXITY_C,
    DEFAULT_NODEWISE_MIN_PERPLEXITY, DEFAULT_NODEWISE_PERPLEXITY_C, HybridKnnConfig,
    PerplexityConfig, SymmetrizationMode, TwoHopSamplingMode, apply_relative_distance_floor,
    build_capped_12hop_knn, build_gumap_12hop_priority_csr, build_hybrid_adaptive_knn,
    build_pure_2hop_knn, build_rerank_knn, compute_2hop_neighborhood_sizes,
    compute_fuzzy_simplicial_set, compute_horvitz_thompson_gamma, compute_knn_probabilities,
    compute_median_2hop_perplexity, compute_nodewise_2hop_perplexities, extract_1hop_2hop_csr,
    extract_sparse_1hop_2hop_pairs, extract_sparse_1hop_2hop_pairs_with_mode, find_ab_params,
    graph_12hop_row_normalized_probabilities_with_degree_decay,
    graph_12hop_row_normalized_probabilities_with_options, graph_row_normalized_probabilities,
    partial_bfs_knn, symmetrize_row_normalized,
};
use wasm_bindgen::prelude::*;

const DEFAULT_K_FACTOR: f64 = 3.0;
const DEFAULT_MARGIN_GAP: f64 = 0.1;
const DEFAULT_COARSE_DIM: usize = 10;
const DEFAULT_RERANK_FACTOR: usize = 16;
const DEFAULT_PARTIAL_BFS_K: usize = 30;
const DEFAULT_POWER: f64 = 2.0;
const DEFAULT_PERPLEXITY_SIGMA_ITERS: usize = 50;
const DEFAULT_FUZZY_SIGMA_ITERS: usize = 64;
const DEFAULT_SIGMA_TOLERANCE: f64 = 1e-5;
const DEFAULT_MIN_DIST: f64 = 0.1;
const DEFAULT_SPREAD: f64 = 1.0;
const DEFAULT_SPARSE_MAX_2HOP: usize = 32;

fn unweighted_f64_csr_to_f32(graph: &CsrGraph<f64>) -> Result<CsrGraph<f32>, JsValue> {
    CsrGraph::<f32>::from_csr(
        graph.indptr(),
        graph.indices(),
        None,
        graph.is_directed(),
        false,
    )
    .map_err(core_err_to_js)
}

/// Extracts canonical unweighted 1-hop and disjoint pure 2-hop CSR graphs.
#[wasm_bindgen(js_name = extract1hop2hopCsr)]
pub fn js_extract_1hop_2hop_csr(graph: &JsCsrGraph) -> Result<JsTwoHopCsr, JsValue> {
    let (adj1, adj2) = extract_1hop_2hop_csr(graph.inner()).map_err(core_err_to_js)?;
    let hop1 = unweighted_f64_csr_to_f32(&adj1)?;
    let hop2 = unweighted_f64_csr_to_f32(&adj2)?;
    Ok(JsTwoHopCsr::from_parts(hop1, hop2))
}

fn parse_two_hop_sampling_mode(raw: Option<String>) -> Result<TwoHopSamplingMode, JsValue> {
    match raw {
        None => Ok(TwoHopSamplingMode::DegreeOrderedHubSuppressed),
        Some(s) => match normalize_enum_token(&s)?.as_str() {
            "degreeorderedhubsuppressed" | "degreeordered" | "hubsuppressed" => {
                Ok(TwoHopSamplingMode::DegreeOrderedHubSuppressed)
            }
            "twosteprandomwalk" | "randomwalk" => Ok(TwoHopSamplingMode::TwoStepRandomWalk),
            _ => Err(invalid_input_err(format!(
                "unknown TwoHopSamplingMode {s:?}; expected 'degreeOrderedHubSuppressed' or 'twoStepRandomWalk'"
            ))),
        },
    }
}

/// Extracts canonical undirected 1-hop and hub-safe capped 2-hop vertex pair arrays `Sparse12HopPairs`.
#[wasm_bindgen(js_name = extractSparse1hop2hopPairs)]
pub fn js_extract_sparse_1hop_2hop_pairs(
    graph: &JsCsrGraph,
    max_2hop_per_node: Option<f64>,
    seed: &JsValue,
) -> Result<JsSparse12HopPairs, JsValue> {
    let cap = parse_optional_usize(max_2hop_per_node, DEFAULT_SPARSE_MAX_2HOP, "max2hopPerNode")?;
    let seed_u64 = parse_seed_u64(seed)?;
    let inner =
        extract_sparse_1hop_2hop_pairs(graph.inner(), cap, seed_u64).map_err(core_err_to_js)?;
    Ok(JsSparse12HopPairs::from_inner(inner))
}

/// Extracts canonical 1-hop and 2-hop pairs using an explicit `TwoHopSamplingMode` and optional `maxBridgeDegree`.
#[wasm_bindgen(js_name = extractSparse1hop2hopPairsWithMode)]
pub fn js_extract_sparse_1hop_2hop_pairs_with_mode(
    graph: &JsCsrGraph,
    options: &JsValue,
) -> Result<JsSparse12HopPairs, JsValue> {
    let mut cap = DEFAULT_SPARSE_MAX_2HOP;
    let mut mode_raw = None;
    let mut bridge = None;
    let mut seed_u64 = crate::js_convert::DEFAULT_SEED;
    if check_options_arg(options, "extractSparse1hop2hopPairsWithMode")? {
        cap = opt_usize_prop(options, "max2hopPerNode", cap)?;
        mode_raw = opt_string_prop(options, "mode")?;
        bridge = opt_maybe_usize_prop(options, "maxBridgeDegree")?;
        seed_u64 = opt_seed_prop(options)?;
    }
    let m = parse_two_hop_sampling_mode(mode_raw)?;
    let inner = extract_sparse_1hop_2hop_pairs_with_mode(graph.inner(), cap, m, bridge, seed_u64)
        .map_err(core_err_to_js)?;
    Ok(JsSparse12HopPairs::from_inner(inner))
}

/// Builds a pure 1+2-hop CSR `KnnGraph` with `{1.0, 2.0}` distances and degree statistics.
#[wasm_bindgen(js_name = buildPure2hopKnn)]
pub fn js_build_pure_2hop_knn(graph: &JsCsrGraph) -> Result<JsTwoHopKnnResult, JsValue> {
    let (knn, k2_deg, stats) = build_pure_2hop_knn(graph.inner()).map_err(core_err_to_js)?;
    Ok(JsTwoHopKnnResult::from_parts(knn, k2_deg, stats))
}

/// Builds a hub-safe capped 1+2-hop CSR `KnnGraph` with `{1.0, 2.0}` distances and degree statistics.
#[wasm_bindgen(js_name = buildCapped12hopKnn)]
pub fn js_build_capped_12hop_knn(
    graph: &JsCsrGraph,
    max_2hop_per_node: Option<f64>,
    seed: &JsValue,
) -> Result<JsTwoHopKnnResult, JsValue> {
    let cap = parse_optional_usize(max_2hop_per_node, DEFAULT_SPARSE_MAX_2HOP, "max2hopPerNode")?;
    let seed_u64 = parse_seed_u64(seed)?;
    let (knn, k2_deg, stats) =
        build_capped_12hop_knn(graph.inner(), cap, seed_u64).map_err(core_err_to_js)?;
    Ok(JsTwoHopKnnResult::from_parts(knn, k2_deg, stats))
}

/// Computes exact distinct 1-hop + 2-hop neighborhood sizes $|B_2(u) \setminus \{u\}|$.
#[wasm_bindgen(js_name = compute2hopNeighborhoodSizes)]
pub fn js_compute_2hop_neighborhood_sizes(graph: &JsCsrGraph) -> Vec<u32> {
    compute_2hop_neighborhood_sizes(graph.inner())
}

/// Computes per-node 2-hop adaptive perplexities.
#[wasm_bindgen(js_name = computeNodewise2hopPerplexities)]
pub fn js_compute_nodewise_2hop_perplexities(
    graph: &JsCsrGraph,
    options: &JsValue,
) -> Result<Vec<f64>, JsValue> {
    let mut c = DEFAULT_NODEWISE_PERPLEXITY_C;
    let mut min_p = DEFAULT_NODEWISE_MIN_PERPLEXITY;
    let mut max_p = DEFAULT_MAX_PERPLEXITY;
    if check_options_arg(options, "computeNodewise2hopPerplexities")? {
        c = opt_f64_prop(options, "c", c)?;
        min_p = opt_f64_prop(options, "minPerplexity", min_p)?;
        max_p = opt_f64_prop(options, "maxPerplexity", max_p)?;
    }
    compute_nodewise_2hop_perplexities(graph.inner(), c, min_p, max_p).map_err(core_err_to_js)
}

/// Computes scalar median 2-hop adaptive perplexity.
#[wasm_bindgen(js_name = computeMedian2hopPerplexity)]
pub fn js_compute_median_2hop_perplexity(
    graph: &JsCsrGraph,
    options: &JsValue,
) -> Result<f64, JsValue> {
    let mut c = DEFAULT_MEDIAN_PERPLEXITY_C;
    let mut min_p = DEFAULT_MEDIAN_MIN_PERPLEXITY;
    let mut max_p = DEFAULT_MAX_PERPLEXITY;
    if check_options_arg(options, "computeMedian2hopPerplexity")? {
        c = opt_f64_prop(options, "c", c)?;
        min_p = opt_f64_prop(options, "minPerplexity", min_p)?;
        max_p = opt_f64_prop(options, "maxPerplexity", max_p)?;
    }
    compute_median_2hop_perplexity(graph.inner(), c, min_p, max_p).map_err(core_err_to_js)
}

/// Builds a hybrid 1+2-hop + two-step coarse `StaticKdTree` + full-$D$ reranked `KnnGraph`.
#[wasm_bindgen(js_name = buildHybridAdaptiveKnn)]
pub fn js_build_hybrid_adaptive_knn(
    graph: &JsCsrGraph,
    coords: &[f64],
    dim: f64,
    perplexities: &[f64],
    options: &JsValue,
) -> Result<JsHybridKnnResult, JsValue> {
    let dim_val = parse_non_negative_usize(dim, "dim")?;
    let mut cfg = HybridKnnConfig::default();
    cfg.k_factor = DEFAULT_K_FACTOR;
    cfg.margin_gap = DEFAULT_MARGIN_GAP;
    cfg.coarse_dim = DEFAULT_COARSE_DIM;
    cfg.rerank_factor = DEFAULT_RERANK_FACTOR;
    if check_options_arg(options, "buildHybridAdaptiveKnn")? {
        cfg.k_factor = opt_f64_prop(options, "kFactor", cfg.k_factor)?;
        cfg.margin_gap = opt_f64_prop(options, "marginGap", cfg.margin_gap)?;
        cfg.coarse_dim = opt_usize_prop(options, "coarseDim", cfg.coarse_dim)?;
        cfg.rerank_factor = opt_usize_prop(options, "rerankFactor", cfg.rerank_factor)?;
        cfg.k_override = opt_maybe_usize_prop(options, "k")?;
        cfg.max_row_k = opt_maybe_usize_prop(options, "maxRowK")?;
    }
    let (knn, stats) =
        build_hybrid_adaptive_knn(graph.inner(), coords, dim_val, perplexities, &cfg)
            .map_err(core_err_to_js)?;
    Ok(JsHybridKnnResult::from_parts(knn, stats))
}

/// Computes a $k$-NN graph via Partial BFS with deterministic Fisher-Yates tie-breaking.
#[wasm_bindgen(js_name = partialBfsKnn)]
pub fn js_partial_bfs_knn(
    graph: &JsCsrGraph,
    k: Option<f64>,
    seed: &JsValue,
) -> Result<JsKnnGraph, JsValue> {
    let k_val = parse_optional_usize(k, DEFAULT_PARTIAL_BFS_K, "k")?;
    let seed_u64 = parse_seed_u64(seed)?;
    let inner = partial_bfs_knn(graph.inner(), k_val, seed_u64).map_err(core_err_to_js)?;
    Ok(JsKnnGraph::from_inner(inner))
}

/// Computes sparse symmetric joint probabilities from a `KnnGraph` and per-node perplexities.
#[wasm_bindgen(js_name = computeKnnProbabilities)]
pub fn js_compute_knn_probabilities(
    knn: &JsKnnGraph,
    perplexities: &[f64],
    options: &JsValue,
) -> Result<JsSparseProbabilities, JsValue> {
    let mut power = DEFAULT_POWER;
    let mut sigma_iters = DEFAULT_PERPLEXITY_SIGMA_ITERS;
    let mut sigma_tolerance = DEFAULT_SIGMA_TOLERANCE;
    let mut row_normalize = false;
    if check_options_arg(options, "computeKnnProbabilities")? {
        power = opt_f64_prop(options, "power", power)?;
        sigma_iters = opt_usize_prop(options, "sigmaIters", sigma_iters)?;
        sigma_tolerance = opt_f64_prop(options, "sigmaTolerance", sigma_tolerance)?;
        row_normalize = opt_bool_prop(options, "rowNormalize", row_normalize)?;
    }
    let mut cfg = PerplexityConfig::default();
    cfg.power = power;
    cfg.sigma_iters = sigma_iters;
    cfg.sigma_tolerance = sigma_tolerance;
    cfg.symmetrization = if row_normalize {
        SymmetrizationMode::RowNormalized
    } else {
        SymmetrizationMode::StandardTsne
    };
    let inner =
        compute_knn_probabilities(knn.inner(), perplexities, &cfg).map_err(core_err_to_js)?;
    Ok(JsSparseProbabilities::from_inner(inner))
}

/// Computes the UMAP / GuMap fuzzy simplicial set probabilities from a `KnnGraph`.
#[wasm_bindgen(js_name = computeFuzzySimplicialSet)]
pub fn js_compute_fuzzy_simplicial_set(
    knn: &JsKnnGraph,
    sigma_iters: Option<f64>,
    sigma_tolerance: Option<f64>,
) -> Result<JsSparseProbabilities, JsValue> {
    let iters = parse_optional_usize(sigma_iters, DEFAULT_FUZZY_SIGMA_ITERS, "sigmaIters")?;
    let tol = sigma_tolerance.unwrap_or(DEFAULT_SIGMA_TOLERANCE);
    let inner = compute_fuzzy_simplicial_set(knn.inner(), iters, tol).map_err(core_err_to_js)?;
    Ok(JsSparseProbabilities::from_inner(inner))
}

/// Fits low-dimensional UMAP parameters `[a, b]` from `minDist` and `spread`.
#[wasm_bindgen(js_name = findAbParams)]
pub fn js_find_ab_params(min_dist: Option<f64>, spread: Option<f64>) -> Result<Vec<f64>, JsValue> {
    let (a, b) = find_ab_params(
        min_dist.unwrap_or(DEFAULT_MIN_DIST),
        spread.unwrap_or(DEFAULT_SPREAD),
    )
    .map_err(core_err_to_js)?;
    Ok(vec![a, b])
}

/// Computes parameter-free Horvitz-Thompson repulsion scaling $\gamma_{\mathrm{eff}}$.
#[wasm_bindgen(js_name = computeHorvitzThompsonGamma)]
pub fn js_compute_horvitz_thompson_gamma(
    gamma_non2: f64,
    num_nodes: f64,
    neg_per_node: f64,
) -> Result<f64, JsValue> {
    let n = parse_non_negative_usize(num_nodes, "numNodes")?;
    let neg = parse_non_negative_usize(neg_per_node, "negPerNode")?;
    compute_horvitz_thompson_gamma(gamma_non2, n, neg).map_err(core_err_to_js)
}

/// Constructs a two-step coarse `StaticKdTree` + full-`dim` reranked `KnnGraph` from row-major
/// `(numNodes, dim)` coordinates.
#[wasm_bindgen(js_name = buildRerankKnn)]
pub fn js_build_rerank_knn(
    coords: &[f64],
    num_nodes: f64,
    dim: f64,
    k: f64,
    options: &JsValue,
) -> Result<JsKnnGraph, JsValue> {
    let n = parse_non_negative_usize(num_nodes, "numNodes")?;
    let d = parse_non_negative_usize(dim, "dim")?;
    let k_val = parse_non_negative_usize(k, "k")?;
    let mut c_dim = DEFAULT_COARSE_DIM;
    let mut r_factor = DEFAULT_RERANK_FACTOR;
    if check_options_arg(options, "buildRerankKnn")? {
        c_dim = opt_usize_prop(options, "coarseDim", c_dim)?;
        r_factor = opt_usize_prop(options, "rerankFactor", r_factor)?;
    }
    let inner = build_rerank_knn(coords, n, d, k_val, c_dim, r_factor).map_err(core_err_to_js)?;
    Ok(JsKnnGraph::from_inner(inner))
}

/// Computes row-normalized symmetrized graph probabilities from a `CsrGraph`.
#[wasm_bindgen(js_name = graphRowNormalizedProbabilities)]
pub fn js_graph_row_normalized_probabilities(
    graph: &JsCsrGraph,
) -> Result<JsSparseProbabilities, JsValue> {
    let inner = graph_row_normalized_probabilities(graph.inner()).map_err(core_err_to_js)?;
    Ok(JsSparseProbabilities::from_inner(inner))
}

/// Computes row-normalized symmetrized 1+2-hop probabilities with optional capping and `(d_i d_j)^(-degreeBeta)` decay.
#[wasm_bindgen(js_name = graph12hopRowNormalizedProbabilities)]
pub fn js_graph_12hop_row_normalized_probabilities(
    graph: &JsCsrGraph,
    options: &JsValue,
) -> Result<JsSparseProbabilities, JsValue> {
    let mut w2_val = 0.5;
    let mut cap_opt = None;
    let mut beta_val = 0.0;
    let mut seed_u64 = crate::js_convert::DEFAULT_SEED;
    if check_options_arg(options, "graph12hopRowNormalizedProbabilities")? {
        w2_val = opt_f64_prop(options, "w2", w2_val)?;
        cap_opt = opt_maybe_usize_prop(options, "max2hopPerNode")?;
        beta_val = opt_f64_prop(options, "degreeBeta", beta_val)?;
        seed_u64 = opt_seed_prop(options)?;
    }
    let inner = graph_12hop_row_normalized_probabilities_with_degree_decay(
        graph.inner(),
        w2_val,
        cap_opt,
        beta_val,
        seed_u64,
    )
    .map_err(core_err_to_js)?;
    Ok(JsSparseProbabilities::from_inner(inner))
}

/// Computes row-normalized symmetrized 1+2-hop probabilities with explicit `TwoHopSamplingMode` and `maxBridgeDegree`.
#[wasm_bindgen(js_name = graph12hopRowNormalizedProbabilitiesWithOptions)]
pub fn js_graph_12hop_row_normalized_probabilities_with_options(
    graph: &JsCsrGraph,
    options: &JsValue,
) -> Result<JsSparseProbabilities, JsValue> {
    let mut w2_val = 0.5;
    let mut cap_opt = None;
    let mut beta_val = 0.0;
    let mut mode_raw = None;
    let mut bridge = None;
    let mut seed_u64 = crate::js_convert::DEFAULT_SEED;
    if check_options_arg(options, "graph12hopRowNormalizedProbabilitiesWithOptions")? {
        w2_val = opt_f64_prop(options, "w2", w2_val)?;
        cap_opt = opt_maybe_usize_prop(options, "max2hopPerNode")?;
        beta_val = opt_f64_prop(options, "degreeBeta", beta_val)?;
        mode_raw = opt_string_prop(options, "mode")?;
        bridge = opt_maybe_usize_prop(options, "maxBridgeDegree")?;
        seed_u64 = opt_seed_prop(options)?;
    }
    let m = parse_two_hop_sampling_mode(mode_raw)?;
    let inner = graph_12hop_row_normalized_probabilities_with_options(
        graph.inner(),
        w2_val,
        cap_opt,
        beta_val,
        m,
        bridge,
        seed_u64,
    )
    .map_err(core_err_to_js)?;
    Ok(JsSparseProbabilities::from_inner(inner))
}

/// Computes row-normalized symmetrized joint probabilities from arbitrary CSR directed affinities.
#[wasm_bindgen(js_name = symmetrizeRowNormalized)]
pub fn js_symmetrize_row_normalized(
    num_nodes: f64,
    indptr: &[u32],
    indices: &[u32],
    weights: &[f64],
) -> Result<JsSparseProbabilities, JsValue> {
    let n = parse_non_negative_usize(num_nodes, "numNodes")?;
    let inner = symmetrize_row_normalized(n, indptr, indices, weights).map_err(core_err_to_js)?;
    Ok(JsSparseProbabilities::from_inner(inner))
}

/// Builds a sorted symmetric CSR `KnnGraph` from 1-hop and 2-hop pair distances enforcing
/// strict 1-hop priority ($d_1 < \mathrm{tier2Margin} \cdot d_2$ and $d_1 < d_2$).
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = buildGumap12hopPriorityCsr)]
pub fn js_build_gumap_12hop_priority_csr(
    num_nodes: f64,
    u1: &[u32],
    v1: &[u32],
    d1: &[f64],
    u2: &[u32],
    v2: &[u32],
    d2: &[f64],
    options: &JsValue,
) -> Result<JsKnnGraph, JsValue> {
    let n = parse_non_negative_usize(num_nodes, "numNodes")?;
    let margin = if check_options_arg(options, "buildGumap12hopPriorityCsr")? {
        opt_f64_prop(options, "tier2Margin", 1.1)?
    } else {
        1.1
    };
    let inner = build_gumap_12hop_priority_csr(n, u1, v1, d1, u2, v2, d2, margin)
        .map_err(core_err_to_js)?;
    Ok(JsKnnGraph::from_inner(inner))
}

/// Clamps pair distances `d1`, `d2`, and `d3` in place using `max(epsDist, minRatio * median_pos)`,
/// returning the applied distance floor.
#[wasm_bindgen(js_name = applyRelativeDistanceFloor)]
pub fn js_apply_relative_distance_floor(
    d1: &mut [f64],
    d2: &mut [f64],
    d3: &mut [f64],
    options: &JsValue,
) -> Result<f64, JsValue> {
    let mut eps_dist = 1e-3;
    let mut min_ratio = 0.1;
    if check_options_arg(options, "applyRelativeDistanceFloor")? {
        eps_dist = opt_f64_prop(options, "epsDist", eps_dist)?;
        min_ratio = opt_f64_prop(options, "minRatio", min_ratio)?;
    }
    apply_relative_distance_floor(d1, d2, d3, eps_dist, min_ratio).map_err(core_err_to_js)
}
