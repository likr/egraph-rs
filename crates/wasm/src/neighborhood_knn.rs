//! WebAssembly bindings for 1+2-hop neighborhood extraction, adaptive perplexity, $k$-NN graphs,
//! and sparse probability symmetrization from `egraph-neighborhood`.

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    core_err_to_js, parse_non_negative_usize, parse_opt_usize_param, parse_optional_usize,
    parse_seed_u64,
};
use crate::js_options::normalize_enum_token;
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

fn unweighted_f64_csr_to_f32(graph: &CsrGraph<f64>) -> Result<CsrGraph<f32>, JsError> {
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
pub fn js_extract_1hop_2hop_csr(graph: &JsCsrGraph) -> Result<JsTwoHopCsr, JsError> {
    let (adj1, adj2) = extract_1hop_2hop_csr(graph.inner()).map_err(core_err_to_js)?;
    let hop1 = unweighted_f64_csr_to_f32(&adj1)?;
    let hop2 = unweighted_f64_csr_to_f32(&adj2)?;
    Ok(JsTwoHopCsr::from_parts(hop1, hop2))
}

fn parse_two_hop_sampling_mode(raw: Option<String>) -> Result<TwoHopSamplingMode, JsError> {
    match raw {
        None => Ok(TwoHopSamplingMode::DegreeOrderedHubSuppressed),
        Some(s) => match normalize_enum_token(&s).as_str() {
            "degreeorderedhubsuppressed" | "degreeordered" | "hubsuppressed" => {
                Ok(TwoHopSamplingMode::DegreeOrderedHubSuppressed)
            }
            "twosteprandomwalk" | "randomwalk" => Ok(TwoHopSamplingMode::TwoStepRandomWalk),
            _ => Err(JsError::new(&format!(
                "unknown TwoHopSamplingMode {s:?}; expected 'degree_ordered_hub_suppressed' or 'two_step_random_walk'"
            ))),
        },
    }
}

/// Extracts canonical undirected 1-hop and hub-safe capped 2-hop vertex pair arrays `Sparse12HopPairs`.
#[wasm_bindgen(js_name = extractSparse1hop2hopPairs)]
pub fn js_extract_sparse_1hop_2hop_pairs(
    graph: &JsCsrGraph,
    max_2hop_per_node: Option<f64>,
    seed: Option<f64>,
) -> Result<JsSparse12HopPairs, JsError> {
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
    max_2hop_per_node: Option<f64>,
    mode: Option<String>,
    max_bridge_degree: Option<f64>,
    seed: Option<f64>,
) -> Result<JsSparse12HopPairs, JsError> {
    let cap = parse_optional_usize(max_2hop_per_node, DEFAULT_SPARSE_MAX_2HOP, "max2hopPerNode")?;
    let m = parse_two_hop_sampling_mode(mode)?;
    let bridge = parse_opt_usize_param(max_bridge_degree, "maxBridgeDegree")?;
    let seed_u64 = parse_seed_u64(seed)?;
    let inner = extract_sparse_1hop_2hop_pairs_with_mode(graph.inner(), cap, m, bridge, seed_u64)
        .map_err(core_err_to_js)?;
    Ok(JsSparse12HopPairs::from_inner(inner))
}

/// Builds a pure 1+2-hop CSR `KnnGraph` with `{1.0, 2.0}` distances and degree statistics.
#[wasm_bindgen(js_name = buildPure2hopKnn)]
pub fn js_build_pure_2hop_knn(graph: &JsCsrGraph) -> Result<JsTwoHopKnnResult, JsError> {
    let (knn, k2_deg, stats) = build_pure_2hop_knn(graph.inner()).map_err(core_err_to_js)?;
    Ok(JsTwoHopKnnResult::from_parts(knn, k2_deg, stats))
}

/// Builds a hub-safe capped 1+2-hop CSR `KnnGraph` with `{1.0, 2.0}` distances and degree statistics.
#[wasm_bindgen(js_name = buildCapped12hopKnn)]
pub fn js_build_capped_12hop_knn(
    graph: &JsCsrGraph,
    max_2hop_per_node: Option<f64>,
    seed: Option<f64>,
) -> Result<JsTwoHopKnnResult, JsError> {
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
    c: Option<f64>,
    min_perplexity: Option<f64>,
    max_perplexity: Option<f64>,
) -> Result<Vec<f64>, JsError> {
    compute_nodewise_2hop_perplexities(
        graph.inner(),
        c.unwrap_or(DEFAULT_NODEWISE_PERPLEXITY_C),
        min_perplexity.unwrap_or(DEFAULT_NODEWISE_MIN_PERPLEXITY),
        max_perplexity.unwrap_or(DEFAULT_MAX_PERPLEXITY),
    )
    .map_err(core_err_to_js)
}

/// Computes scalar median 2-hop adaptive perplexity.
#[wasm_bindgen(js_name = computeMedian2hopPerplexity)]
pub fn js_compute_median_2hop_perplexity(
    graph: &JsCsrGraph,
    c: Option<f64>,
    min_perplexity: Option<f64>,
    max_perplexity: Option<f64>,
) -> Result<f64, JsError> {
    compute_median_2hop_perplexity(
        graph.inner(),
        c.unwrap_or(DEFAULT_MEDIAN_PERPLEXITY_C),
        min_perplexity.unwrap_or(DEFAULT_MEDIAN_MIN_PERPLEXITY),
        max_perplexity.unwrap_or(DEFAULT_MAX_PERPLEXITY),
    )
    .map_err(core_err_to_js)
}

/// Builds a hybrid 1+2-hop + two-step coarse `StaticKdTree` + full-$D$ reranked `KnnGraph`.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = buildHybridAdaptiveKnn)]
pub fn js_build_hybrid_adaptive_knn(
    graph: &JsCsrGraph,
    coords: &[f64],
    dim: f64,
    perplexities: &[f64],
    k_factor: Option<f64>,
    margin_gap: Option<f64>,
    coarse_dim: Option<f64>,
    rerank_factor: Option<f64>,
    k: Option<f64>,
    max_row_k: Option<f64>,
) -> Result<JsHybridKnnResult, JsError> {
    let dim_val = parse_non_negative_usize(dim, "dim")?;
    let cfg = HybridKnnConfig {
        k_factor: k_factor.unwrap_or(DEFAULT_K_FACTOR),
        k_override: parse_opt_usize_param(k, "k")?,
        margin_gap: margin_gap.unwrap_or(DEFAULT_MARGIN_GAP),
        coarse_dim: parse_optional_usize(coarse_dim, DEFAULT_COARSE_DIM, "coarseDim")?,
        rerank_factor: parse_optional_usize(rerank_factor, DEFAULT_RERANK_FACTOR, "rerankFactor")?,
        max_row_k: parse_opt_usize_param(max_row_k, "maxRowK")?,
    };
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
    seed: Option<f64>,
) -> Result<JsKnnGraph, JsError> {
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
    power: Option<f64>,
    sigma_iters: Option<f64>,
    sigma_tolerance: Option<f64>,
    row_normalize: Option<bool>,
) -> Result<JsSparseProbabilities, JsError> {
    let cfg = PerplexityConfig {
        power: power.unwrap_or(DEFAULT_POWER),
        sigma_iters: parse_optional_usize(
            sigma_iters,
            DEFAULT_PERPLEXITY_SIGMA_ITERS,
            "sigmaIters",
        )?,
        sigma_tolerance: sigma_tolerance.unwrap_or(DEFAULT_SIGMA_TOLERANCE),
        symmetrization: if row_normalize.unwrap_or(false) {
            SymmetrizationMode::RowNormalized
        } else {
            SymmetrizationMode::StandardTsne
        },
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
) -> Result<JsSparseProbabilities, JsError> {
    let iters = parse_optional_usize(sigma_iters, DEFAULT_FUZZY_SIGMA_ITERS, "sigmaIters")?;
    let tol = sigma_tolerance.unwrap_or(DEFAULT_SIGMA_TOLERANCE);
    let inner = compute_fuzzy_simplicial_set(knn.inner(), iters, tol).map_err(core_err_to_js)?;
    Ok(JsSparseProbabilities::from_inner(inner))
}

/// Fits low-dimensional UMAP parameters `[a, b]` from `minDist` and `spread`.
#[wasm_bindgen(js_name = findAbParams)]
pub fn js_find_ab_params(min_dist: Option<f64>, spread: Option<f64>) -> Result<Vec<f64>, JsError> {
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
) -> Result<f64, JsError> {
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
    coarse_dim: Option<f64>,
    rerank_factor: Option<f64>,
) -> Result<JsKnnGraph, JsError> {
    let n = parse_non_negative_usize(num_nodes, "numNodes")?;
    let d = parse_non_negative_usize(dim, "dim")?;
    let k_val = parse_non_negative_usize(k, "k")?;
    let c_dim = parse_optional_usize(coarse_dim, DEFAULT_COARSE_DIM, "coarseDim")?;
    let r_factor = parse_optional_usize(rerank_factor, DEFAULT_RERANK_FACTOR, "rerankFactor")?;
    let inner = build_rerank_knn(coords, n, d, k_val, c_dim, r_factor).map_err(core_err_to_js)?;
    Ok(JsKnnGraph::from_inner(inner))
}

/// Computes row-normalized symmetrized graph probabilities from a `CsrGraph`.
#[wasm_bindgen(js_name = graphRowNormalizedProbabilities)]
pub fn js_graph_row_normalized_probabilities(
    graph: &JsCsrGraph,
) -> Result<JsSparseProbabilities, JsError> {
    let inner = graph_row_normalized_probabilities(graph.inner()).map_err(core_err_to_js)?;
    Ok(JsSparseProbabilities::from_inner(inner))
}

/// Computes row-normalized symmetrized 1+2-hop probabilities with optional capping and `(d_i d_j)^(-degreeBeta)` decay.
#[wasm_bindgen(js_name = graph12hopRowNormalizedProbabilities)]
pub fn js_graph_12hop_row_normalized_probabilities(
    graph: &JsCsrGraph,
    w2: Option<f64>,
    max_2hop_per_node: Option<f64>,
    degree_beta: Option<f64>,
    seed: Option<f64>,
) -> Result<JsSparseProbabilities, JsError> {
    let w2_val = w2.unwrap_or(0.5);
    let cap_opt = parse_opt_usize_param(max_2hop_per_node, "max2hopPerNode")?;
    let beta_val = degree_beta.unwrap_or(0.0);
    let seed_u64 = parse_seed_u64(seed)?;
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
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = graph12hopRowNormalizedProbabilitiesWithOptions)]
pub fn js_graph_12hop_row_normalized_probabilities_with_options(
    graph: &JsCsrGraph,
    w2: Option<f64>,
    max_2hop_per_node: Option<f64>,
    degree_beta: Option<f64>,
    mode: Option<String>,
    max_bridge_degree: Option<f64>,
    seed: Option<f64>,
) -> Result<JsSparseProbabilities, JsError> {
    let w2_val = w2.unwrap_or(0.5);
    let cap_opt = parse_opt_usize_param(max_2hop_per_node, "max2hopPerNode")?;
    let beta_val = degree_beta.unwrap_or(0.0);
    let m = parse_two_hop_sampling_mode(mode)?;
    let bridge = parse_opt_usize_param(max_bridge_degree, "maxBridgeDegree")?;
    let seed_u64 = parse_seed_u64(seed)?;
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
) -> Result<JsSparseProbabilities, JsError> {
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
    tier2_margin: Option<f64>,
) -> Result<JsKnnGraph, JsError> {
    let n = parse_non_negative_usize(num_nodes, "numNodes")?;
    let margin = tier2_margin.unwrap_or(1.1);
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
    eps_dist: Option<f64>,
    min_ratio: Option<f64>,
) -> Result<f64, JsError> {
    apply_relative_distance_floor(
        d1,
        d2,
        d3,
        eps_dist.unwrap_or(1e-3),
        min_ratio.unwrap_or(0.1),
    )
    .map_err(core_err_to_js)
}
