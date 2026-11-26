//! Multi-level Louvain coarsening, centroid prolongation, and area-reserved coarse backbone bindings (`f32`).

use egraph::clustering::{
    CoarseLouvainConfig, align_clustered_centroids, louvain_coarsen,
    project_clustered_layout_with_radii,
};
use egraph::core::CsrGraph;
use egraph::layout::{
    CoarseBackboneConfig, CoarseBackboneResult, CoarseBackboneStats, CoarseDistanceBackend,
    CoarseEdgeLengthMode, coarse_backbone_init,
};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, csr_f64_to_f32, ffi_err_to_js, parse_seed_u64};
use crate::js_options::{
    check_options_arg, get_prop, normalize_enum_token, opt_f64_prop, opt_maybe_f64_prop,
    opt_seed_prop, opt_string_prop, opt_usize_prop,
};
use crate::layout_buffers::JsLayout2d;

const DEFAULT_JITTER_SCALE: f64 = 0.5;
const DEFAULT_GAMMA_TETHER: f64 = 0.5;

fn parse_coarse_edge_length_mode(raw: &str) -> Result<CoarseEdgeLengthMode, JsError> {
    match normalize_enum_token(raw).as_str() {
        "conductanceinverse" | "conductance" | "inverseconductance" => {
            Ok(CoarseEdgeLengthMode::ConductanceInverse)
        }
        "resistancecurvatureflow" | "resistancecurvature" | "rcf" => {
            Ok(CoarseEdgeLengthMode::ResistanceCurvatureFlow)
        }
        _ => Err(JsError::new(&format!(
            "unknown edgeLengthMode {raw:?}; expected 'conductanceInverse' or 'resistanceCurvatureFlow'"
        ))),
    }
}

fn parse_coarse_distance_backend(raw: &str) -> Result<CoarseDistanceBackend, JsError> {
    match normalize_enum_token(raw).as_str() {
        "rhgh" => Ok(CoarseDistanceBackend::Rhgh),
        "chebyshevdog" | "dog" => Ok(CoarseDistanceBackend::ChebyshevDog),
        "shortestpath" | "sp" => Ok(CoarseDistanceBackend::ShortestPath),
        _ => Err(JsError::new(&format!(
            "unknown distanceBackend {raw:?}; expected 'rhgh', 'chebyshevDog', or 'shortestPath'"
        ))),
    }
}

fn parse_coarse_louvain_config(
    options: &JsValue,
    ctx: &str,
) -> Result<CoarseLouvainConfig, JsError> {
    let mut cfg = CoarseLouvainConfig::default();
    if check_options_arg(options, ctx)? {
        cfg.max_coarse_nodes = opt_usize_prop(options, "maxCoarseNodes", cfg.max_coarse_nodes)?;
        cfg.resolution = opt_f64_prop(options, "resolution", cfg.resolution)?;
        cfg.resolution_decay = opt_f64_prop(options, "resolutionDecay", cfg.resolution_decay)?;
        cfg.max_passes = opt_usize_prop(options, "maxPasses", cfg.max_passes)?;
        cfg.min_modularity_gain =
            opt_f64_prop(options, "minModularityGain", cfg.min_modularity_gain)?;
        cfg.max_outer_rounds = opt_usize_prop(options, "maxOuterRounds", cfg.max_outer_rounds)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    cfg.validate().map_err(core_err_to_js)?;
    Ok(cfg)
}

pub(crate) fn parse_coarse_backbone_config(
    options: &JsValue,
    ctx: &str,
) -> Result<CoarseBackboneConfig, JsError> {
    let mut cfg = CoarseBackboneConfig::default();
    if check_options_arg(options, ctx)? {
        cfg.louvain = parse_coarse_louvain_config(options, ctx)?;
        if let Some(m_str) = opt_string_prop(options, "edgeLengthMode")? {
            cfg.edge_length_mode = parse_coarse_edge_length_mode(&m_str)?;
        }
        if let Some(b_str) = opt_string_prop(options, "distanceBackend")? {
            cfg.distance_backend = parse_coarse_distance_backend(&b_str)?;
        }
        cfg.alpha = opt_f64_prop(options, "alpha", cfg.alpha)?;
        cfg.c0 = opt_f64_prop(options, "c0", cfg.c0)?;
        cfg.kappa = opt_f64_prop(options, "kappa", cfg.kappa)?;
        cfg.gamma_w = opt_f64_prop(options, "gammaW", cfg.gamma_w)?;
        cfg.gamma_m = opt_f64_prop(options, "gammaM", cfg.gamma_m)?;
        cfg.mu_ov = opt_f64_prop(options, "muOv", cfg.mu_ov)?;
        cfg.m_gap = opt_f64_prop(options, "mGap", cfg.m_gap)?;
        cfg.coarse_pivots = opt_usize_prop(options, "coarsePivots", cfg.coarse_pivots)?;
        cfg.coarse_neg_per_node =
            opt_usize_prop(options, "coarseNegPerNode", cfg.coarse_neg_per_node)?;
        cfg.coarse_t_max = opt_usize_prop(options, "coarseTMax", cfg.coarse_t_max)?;
        cfg.rhgh_gamma = opt_maybe_f64_prop(options, "rhghGamma")?;
        cfg.rhgh_p = opt_maybe_f64_prop(options, "rhghP")?;
        cfg.rcf_steps = opt_usize_prop(options, "rcfSteps", cfg.rcf_steps)?;
        cfg.rcf_eta = opt_f64_prop(options, "rcfEta", cfg.rcf_eta)?;
        cfg.jitter_scale = opt_f64_prop(options, "jitterScale", cfg.jitter_scale)?;
        cfg.gamma_tether = opt_f64_prop(options, "gammaTether", cfg.gamma_tether)?;
        let ts_val = get_prop(options, "targetStd")?;
        if ts_val.is_null() {
            cfg.target_std = None;
        } else if !ts_val.is_undefined() {
            cfg.target_std = Some(
                ts_val
                    .as_f64()
                    .ok_or_else(|| JsError::new("options.targetStd must be a number or null"))?,
            );
        }
        cfg.robust_quantile = opt_f64_prop(options, "robustQuantile", cfg.robust_quantile)?;
        cfg.seed = cfg.louvain.seed;
    }
    cfg.validate().map_err(core_err_to_js)?;
    Ok(cfg)
}

/// Output of `louvainCoarsen`.
#[wasm_bindgen(js_name = CoarsenedHierarchyResult)]
#[derive(Debug, Clone)]
pub struct JsCoarsenedHierarchyResult {
    coarse_graph: CsrGraph<f32>,
    node_to_coarse: Vec<u32>,
    coarse_sizes: Vec<u32>,
    num_clusters: usize,
    level_num_clusters: Vec<usize>,
    level_resolutions: Vec<f64>,
    level_modularities: Vec<f64>,
}

#[wasm_bindgen(js_class = CoarsenedHierarchyResult)]
impl JsCoarsenedHierarchyResult {
    /// Returns a cloned handle to the coarsest super-graph `CsrGraph`.
    #[wasm_bindgen(js_name = coarseGraph)]
    pub fn coarse_graph(&self) -> JsCsrGraph {
        JsCsrGraph::from_inner(self.coarse_graph.clone())
    }

    /// Returns the original-to-coarsest cluster assignment (`Uint32Array`).
    #[wasm_bindgen(js_name = nodeToCoarse)]
    pub fn node_to_coarse(&self) -> Vec<u32> {
        self.node_to_coarse.clone()
    }

    /// Returns the original vertex counts per coarsest cluster (`Uint32Array`).
    #[wasm_bindgen(js_name = coarseSizes)]
    pub fn coarse_sizes(&self) -> Vec<u32> {
        self.coarse_sizes.clone()
    }

    /// Returns the number of coarsest super-nodes.
    #[wasm_bindgen(getter, js_name = numClusters)]
    pub fn num_clusters(&self) -> usize {
        self.num_clusters
    }

    /// Returns the number of outer Louvain contraction levels.
    #[wasm_bindgen(getter, js_name = numLevels)]
    pub fn num_levels(&self) -> usize {
        self.level_num_clusters.len()
    }

    /// Returns the number of clusters at each contraction level.
    #[wasm_bindgen(js_name = levelNumClusters)]
    pub fn level_num_clusters(&self) -> Vec<usize> {
        self.level_num_clusters.clone()
    }

    /// Returns the resolution parameter used at each contraction level.
    #[wasm_bindgen(js_name = levelResolutions)]
    pub fn level_resolutions(&self) -> Vec<f64> {
        self.level_resolutions.clone()
    }

    /// Returns the modularity $Q_\gamma$ achieved at each contraction level.
    #[wasm_bindgen(js_name = levelModularities)]
    pub fn level_modularities(&self) -> Vec<f64> {
        self.level_modularities.clone()
    }
}

/// Runs multi-level Louvain coarsening with resolution decay up to `maxCoarseNodes`.
#[wasm_bindgen(js_name = louvainCoarsen)]
pub fn js_louvain_coarsen(
    graph: &JsCsrGraph,
    options: &JsValue,
) -> Result<JsCoarsenedHierarchyResult, JsError> {
    let cfg = parse_coarse_louvain_config(options, "louvainCoarsen")?;
    let hierarchy = louvain_coarsen(graph.inner(), &cfg).map_err(core_err_to_js)?;
    let coarse_graph = csr_f64_to_f32(&hierarchy.coarsest.graph)?;
    Ok(JsCoarsenedHierarchyResult {
        coarse_graph,
        node_to_coarse: hierarchy.coarsest.node_to_coarse,
        coarse_sizes: hierarchy.coarsest.coarse_sizes,
        num_clusters: hierarchy.coarsest.num_clusters,
        level_num_clusters: hierarchy
            .levels
            .iter()
            .map(|l| l.coarse.num_clusters)
            .collect(),
        level_resolutions: hierarchy.levels.iter().map(|l| l.resolution).collect(),
        level_modularities: hierarchy.levels.iter().map(|l| l.modularity).collect(),
    })
}

/// Projects a coarse 2D `Layout2d` (`f32`) onto fine vertices using per-cluster `clusterRadii`.
#[wasm_bindgen(js_name = projectClusteredLayoutWithRadii)]
pub fn js_project_clustered_layout_with_radii(
    coarse_layout: &JsLayout2d,
    node_to_coarse: &[u32],
    cluster_radii: &[f64],
    jitter_scale: Option<f64>,
    seed: Option<f64>,
) -> Result<JsLayout2d, JsError> {
    let j_scale = jitter_scale.unwrap_or(DEFAULT_JITTER_SCALE);
    let seed_u64 = parse_seed_u64(seed)?;
    let coarse_soa = coarse_layout.inner().to_layout2d();
    let fine = project_clustered_layout_with_radii(
        &coarse_soa,
        node_to_coarse,
        cluster_radii,
        j_scale,
        seed_u64,
    )
    .map_err(core_err_to_js)?;
    Ok(JsLayout2d::from_layout2d(&fine))
}

/// Aligns fine-level cluster centroids toward `coarseLayout` via rigid per-cluster translation
/// in place without rigid disk containment.
#[wasm_bindgen(js_name = alignClusteredCentroids)]
pub fn js_align_clustered_centroids(
    fine_layout: &mut JsLayout2d,
    coarse_layout: &JsLayout2d,
    node_to_coarse: &[u32],
    gamma_tether: Option<f64>,
) -> Result<(), JsError> {
    let tether = gamma_tether.unwrap_or(DEFAULT_GAMMA_TETHER);
    let mut fine_soa = fine_layout.inner().to_layout2d();
    let coarse_soa = coarse_layout.inner().to_layout2d();
    align_clustered_centroids(&mut fine_soa, &coarse_soa, node_to_coarse, tether)
        .map_err(core_err_to_js)?;
    fine_layout
        .inner_mut()
        .copy_from_layout2d(&fine_soa)
        .map_err(ffi_err_to_js)?;
    Ok(())
}

/// Output of `coarseBackboneInit`.
#[wasm_bindgen(js_name = CoarseBackboneResult)]
#[derive(Debug, Clone)]
pub struct JsCoarseBackboneResult {
    fine_layout: JsLayout2d,
    coarse_layout: JsLayout2d,
    coarse_graph: CsrGraph<f32>,
    node_to_coarse: Vec<u32>,
    coarse_sizes: Vec<u32>,
    radii: Vec<f64>,
    stats: CoarseBackboneStats,
}

impl JsCoarseBackboneResult {
    pub(crate) fn from_core(res: CoarseBackboneResult<f32>) -> Result<Self, JsError> {
        let coarse_graph = csr_f64_to_f32(&res.hierarchy.coarsest.graph)?;
        Ok(Self {
            fine_layout: JsLayout2d::from_layout2d(&res.fine_layout),
            coarse_layout: JsLayout2d::from_layout2d(&res.coarse_layout),
            coarse_graph,
            node_to_coarse: res.hierarchy.coarsest.node_to_coarse,
            coarse_sizes: res.hierarchy.coarsest.coarse_sizes,
            radii: res.radii,
            stats: res.stats,
        })
    }
}

#[wasm_bindgen(js_class = CoarseBackboneResult)]
impl JsCoarseBackboneResult {
    /// Returns the prolonged fine-level `Layout2d` (`f32`).
    #[wasm_bindgen(js_name = fineLayout)]
    pub fn fine_layout(&self) -> JsLayout2d {
        self.fine_layout.clone()
    }

    /// Returns the coarse super-node `Layout2d` (`f32`) scaled consistently with `fineLayout`.
    #[wasm_bindgen(js_name = coarseLayout)]
    pub fn coarse_layout(&self) -> JsLayout2d {
        self.coarse_layout.clone()
    }

    /// Returns the coarsest `CsrGraph` (`f32`).
    #[wasm_bindgen(js_name = coarseGraph)]
    pub fn coarse_graph(&self) -> JsCsrGraph {
        JsCsrGraph::from_inner(self.coarse_graph.clone())
    }

    /// Returns the original-to-coarsest cluster IDs (`Uint32Array`).
    #[wasm_bindgen(js_name = nodeToCoarse)]
    pub fn node_to_coarse(&self) -> Vec<u32> {
        self.node_to_coarse.clone()
    }

    /// Returns the original vertex counts per coarse cluster (`Uint32Array`).
    #[wasm_bindgen(js_name = coarseSizes)]
    pub fn coarse_sizes(&self) -> Vec<u32> {
        self.coarse_sizes.clone()
    }

    /// Returns the reserved cluster radii (`Float64Array`).
    pub fn radii(&self) -> Vec<f64> {
        self.radii.clone()
    }

    #[wasm_bindgen(getter, js_name = numFineNodes)]
    pub fn num_fine_nodes(&self) -> usize {
        self.stats.num_fine_nodes
    }

    #[wasm_bindgen(getter, js_name = numCoarseNodes)]
    pub fn num_coarse_nodes(&self) -> usize {
        self.stats.num_coarse_nodes
    }

    #[wasm_bindgen(getter, js_name = numCoarseEdges)]
    pub fn num_coarse_edges(&self) -> usize {
        self.stats.num_coarse_edges
    }

    #[wasm_bindgen(getter, js_name = louvainLevels)]
    pub fn louvain_levels(&self) -> usize {
        self.stats.louvain_levels
    }

    #[wasm_bindgen(getter, js_name = meanClusterSize)]
    pub fn mean_cluster_size(&self) -> f64 {
        self.stats.mean_cluster_size
    }

    #[wasm_bindgen(getter, js_name = maxClusterSize)]
    pub fn max_cluster_size(&self) -> usize {
        self.stats.max_cluster_size
    }

    #[wasm_bindgen(getter, js_name = medianLEdge)]
    pub fn median_l_edge(&self) -> f64 {
        self.stats.median_l_edge
    }

    #[wasm_bindgen(getter, js_name = maxRadius)]
    pub fn max_radius(&self) -> f64 {
        self.stats.max_radius
    }

    #[wasm_bindgen(getter, js_name = coarseOverlapPairsInitial)]
    pub fn coarse_overlap_pairs_initial(&self) -> usize {
        self.stats.coarse_overlap_pairs_initial
    }

    #[wasm_bindgen(getter, js_name = coarseOverlapPairsFinal)]
    pub fn coarse_overlap_pairs_final(&self) -> usize {
        self.stats.coarse_overlap_pairs_final
    }

    #[wasm_bindgen(getter, js_name = scaleFactorApplied)]
    pub fn scale_factor_applied(&self) -> f64 {
        self.stats.scale_factor_applied
    }
}

/// Runs coarse Louvain edge-length-first area reservation and non-SPD backbone initialization.
#[wasm_bindgen(js_name = coarseBackboneInit)]
pub fn js_coarse_backbone_init(
    graph: &JsCsrGraph,
    options: &JsValue,
) -> Result<JsCoarseBackboneResult, JsError> {
    let cfg = parse_coarse_backbone_config(options, "coarseBackboneInit")?;
    let res = coarse_backbone_init::<f32, f32>(graph.inner(), &cfg).map_err(core_err_to_js)?;
    JsCoarseBackboneResult::from_core(res)
}
