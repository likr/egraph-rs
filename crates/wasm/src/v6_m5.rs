//! M5 v6 WebAssembly bindings for Louvain coarsening, centroid prolongation,
//! pure-Rust pivoted `RHGH` landmarks, coarse edge-length-first area reservation,
//! and `PaCMAP-Graph` coarse backbone initialization (`f32`-backed for minimal `.wasm` footprint).

use egraph::clustering::{
    CoarseLouvainConfig, align_clustered_centroids, louvain_coarsen,
    project_clustered_layout_with_radii,
};
use egraph::core::{CsrGraph, IterativeSolver};
use egraph::layout::{
    CoarseBackboneConfig, CoarseBackboneResult, CoarseBackboneStats, CoarseDistanceBackend,
    CoarseEdgeLengthMode, ScreenedPoissonRefiner, ScreenedPoissonRefinerConfig, SgdSolver,
    coarse_backbone_init,
};
use egraph::neighborhood::{RhghLandmarkConfig, RhghLandmarkStats, compute_pivoted_rhgh_landmarks};
use egraph::spatial::{
    DensityFieldKernel, ScreenedPoissonConfig, ScreenedPoissonField2d, ScreenedPoissonOutput,
    ScreenedPoissonStats,
};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    core_err_to_js, csr_f64_to_f32, ffi_err_to_js, parse_non_negative_usize, parse_seed_u64,
};
use crate::js_options::{
    check_options_arg, get_prop, normalize_enum_token, opt_bool_prop, opt_f64_prop,
    opt_maybe_f64_prop, opt_seed_prop, opt_string_prop, opt_usize_prop, parse_pacmap_graph_config,
    solver_status_str,
};
use crate::layout_buffers::JsLayout2d;
use crate::v6_m2::JsDogVaradhanLandmarks;
use crate::v6_m3::{
    JsSgdSolver, run_layout_in_place, run_steps_layout_in_place, step_layout_in_place,
};

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

fn parse_coarse_backbone_config(
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

/// Output of `computePivotedRhghLandmarks`.
#[wasm_bindgen(js_name = RhghLandmarksResult)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsRhghLandmarksResult {
    landmarks: JsDogVaradhanLandmarks,
    stats: RhghLandmarkStats,
}

#[wasm_bindgen(js_class = RhghLandmarksResult)]
impl JsRhghLandmarksResult {
    /// Returns the `DogVaradhanLandmarks` handle usable with `SgdSolver` and `GuMapSolver`.
    pub fn landmarks(&self) -> JsDogVaradhanLandmarks {
        self.landmarks.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn gamma(&self) -> f64 {
        self.stats.gamma
    }

    #[wasm_bindgen(getter, js_name = pExp)]
    pub fn p_exp(&self) -> f64 {
        self.stats.p_exp
    }

    #[wasm_bindgen(getter, js_name = degCv)]
    pub fn deg_cv(&self) -> f64 {
        self.stats.deg_cv
    }

    #[wasm_bindgen(getter, js_name = leafFraction)]
    pub fn leaf_fraction(&self) -> f64 {
        self.stats.leaf_fraction
    }

    #[wasm_bindgen(getter, js_name = meanDegree)]
    pub fn mean_degree(&self) -> f64 {
        self.stats.mean_degree
    }

    #[wasm_bindgen(getter)]
    pub fn exp2(&self) -> f64 {
        self.stats.exp2
    }

    #[wasm_bindgen(getter, js_name = sSw)]
    pub fn s_sw(&self) -> f64 {
        self.stats.s_sw
    }

    #[wasm_bindgen(getter, js_name = numLandmarks)]
    pub fn num_landmarks(&self) -> usize {
        self.stats.n_landmarks
    }

    #[wasm_bindgen(getter, js_name = numProbes)]
    pub fn num_probes(&self) -> usize {
        self.stats.num_probes
    }

    #[wasm_bindgen(getter, js_name = maxPcgIters)]
    pub fn max_pcg_iters(&self) -> usize {
        self.stats.max_pcg_iters
    }

    #[wasm_bindgen(getter, js_name = totalPcgIters)]
    pub fn total_pcg_iters(&self) -> usize {
        self.stats.total_pcg_iters
    }

    #[wasm_bindgen(getter, js_name = allConverged)]
    pub fn all_converged(&self) -> bool {
        self.stats.all_converged
    }

    #[wasm_bindgen(getter, js_name = edgeMedScale)]
    pub fn edge_med_scale(&self) -> f64 {
        self.stats.edge_med_scale
    }
}

/// Computes pure-Rust pivoted `RHGH` landmark tables via `AmgPcgSolver` + JL diagonal probes.
#[wasm_bindgen(js_name = computePivotedRhghLandmarks)]
pub fn js_compute_pivoted_rhgh_landmarks(
    graph: &JsCsrGraph,
    options: &JsValue,
) -> Result<JsRhghLandmarksResult, JsError> {
    let mut cfg = RhghLandmarkConfig::default();
    if check_options_arg(options, "computePivotedRhghLandmarks")? {
        cfg.gamma = opt_maybe_f64_prop(options, "gamma")?;
        cfg.p_exp = opt_maybe_f64_prop(options, "pExp")?;
        cfg.n_landmarks = opt_usize_prop(options, "numLandmarks", cfg.n_landmarks)?;
        cfg.min_degree = opt_usize_prop(options, "minDegree", cfg.min_degree)?;
        cfg.num_probes = opt_usize_prop(options, "numProbes", cfg.num_probes)?;
        cfg.rel_tol = opt_f64_prop(options, "relTol", cfg.rel_tol)?;
        cfg.max_iterations = opt_usize_prop(options, "maxIterations", cfg.max_iterations)?;
        cfg.normalize_1hop = opt_bool_prop(options, "normalize1hop", cfg.normalize_1hop)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    let (inner, stats) =
        compute_pivoted_rhgh_landmarks(graph.inner(), &cfg).map_err(core_err_to_js)?;
    Ok(JsRhghLandmarksResult {
        landmarks: JsDogVaradhanLandmarks::from_rhgh_parts(inner, &stats),
        stats,
    })
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
    fn from_core(res: CoarseBackboneResult<f32>) -> Result<Self, JsError> {
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

/// Output of `SgdSolver.pacmapGraphWithBackbone`.
#[wasm_bindgen(js_name = PacmapBackboneResult)]
#[derive(Debug, Clone)]
pub struct JsPacmapBackboneResult {
    solver: JsSgdSolver,
    fine_layout: JsLayout2d,
    backbone: JsCoarseBackboneResult,
}

#[wasm_bindgen(js_class = PacmapBackboneResult)]
impl JsPacmapBackboneResult {
    /// Returns the initialized cooperative `SgdSolver` (`PaCMAP-Graph` mode).
    pub fn solver(&self) -> JsSgdSolver {
        self.solver.clone()
    }

    /// Returns the coarse-backbone initial `Layout2d` (`f32`).
    #[wasm_bindgen(js_name = fineLayout)]
    pub fn fine_layout(&self) -> JsLayout2d {
        self.fine_layout.clone()
    }

    /// Returns the underlying `CoarseBackboneResult` handle.
    pub fn backbone(&self) -> JsCoarseBackboneResult {
        self.backbone.clone()
    }
}

pub(crate) fn sgd_pacmap_graph_with_backbone(
    graph: &JsCsrGraph,
    options: &JsValue,
) -> Result<JsPacmapBackboneResult, JsError> {
    let pm_cfg = parse_pacmap_graph_config(options, "SgdSolver.pacmapGraphWithBackbone")?;
    let bb_cfg = parse_coarse_backbone_config(options, "SgdSolver.pacmapGraphWithBackbone")?;
    let (solver, fine_layout, bb) =
        SgdSolver::<f32>::pacmap_graph_with_backbone(graph.inner(), pm_cfg, &bb_cfg)
            .map_err(core_err_to_js)?;
    Ok(JsPacmapBackboneResult {
        solver: JsSgdSolver::from_inner(solver),
        fine_layout: JsLayout2d::from_layout2d(&fine_layout),
        backbone: JsCoarseBackboneResult::from_core(bb)?,
    })
}

// ============================================================================
// M5.2 Screened Poisson Unilateral Upper-Bound Density Field & Refiner (f32)
// ============================================================================

fn parse_density_field_kernel(raw: &str) -> Result<DensityFieldKernel, JsError> {
    match normalize_enum_token(raw).as_str() {
        "screenedpoisson" | "sp" => Ok(DensityFieldKernel::ScreenedPoisson),
        "singlegaussian" | "gaussian" | "sg" => Ok(DensityFieldKernel::SingleGaussian),
        _ => Err(JsError::new(&format!(
            "unknown density field kernel {raw:?}; expected 'screenedPoisson' or 'singleGaussian'"
        ))),
    }
}

fn density_field_kernel_js_str(kernel: DensityFieldKernel) -> &'static str {
    match kernel {
        DensityFieldKernel::ScreenedPoisson => "screenedPoisson",
        DensityFieldKernel::SingleGaussian => "singleGaussian",
    }
}

fn parse_screened_poisson_config(
    options: &JsValue,
    ctx: &str,
) -> Result<ScreenedPoissonConfig, JsError> {
    let mut cfg = ScreenedPoissonConfig::default();
    if check_options_arg(options, ctx)? {
        if let Some(k_str) = opt_string_prop(options, "kernel")? {
            cfg.kernel = parse_density_field_kernel(&k_str)?;
        }
        cfg.grid_size = opt_usize_prop(options, "gridSize", cfg.grid_size)?;
        cfg.padding_factor = opt_usize_prop(options, "paddingFactor", cfg.padding_factor)?;
        cfg.margin_ratio = opt_f64_prop(options, "marginRatio", cfg.margin_ratio)?;
        cfg.sigma_ratio = opt_f64_prop(options, "sigmaRatio", cfg.sigma_ratio)?;
        cfg.sigma = opt_maybe_f64_prop(options, "sigma")?;
        cfg.screening_length_ratio =
            opt_f64_prop(options, "screeningLengthRatio", cfg.screening_length_ratio)?;
        cfg.kappa = opt_maybe_f64_prop(options, "kappa")?;
        cfg.rho_max_factor = opt_f64_prop(options, "rhoMaxFactor", cfg.rho_max_factor)?;
        cfg.rho_max = opt_maybe_f64_prop(options, "rhoMax")?;
        cfg.exact_adjoint = opt_bool_prop(options, "exactAdjoint", cfg.exact_adjoint)?;
    }
    cfg.validate().map_err(core_err_to_js)?;
    Ok(cfg)
}

fn parse_screened_poisson_refiner_config(
    options: &JsValue,
    ctx: &str,
) -> Result<ScreenedPoissonRefinerConfig, JsError> {
    let mut cfg = ScreenedPoissonRefinerConfig {
        field: parse_screened_poisson_config(options, ctx)?,
        ..ScreenedPoissonRefinerConfig::default()
    };
    if check_options_arg(options, ctx)? {
        cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
        cfg.step_size = opt_f64_prop(options, "stepSize", cfg.step_size)?;
        cfg.momentum = opt_f64_prop(options, "momentum", cfg.momentum)?;
        cfg.max_step_ratio = opt_f64_prop(options, "maxStepRatio", cfg.max_step_ratio)?;
        cfg.anchor_weight = opt_f64_prop(options, "anchorWeight", cfg.anchor_weight)?;
        cfg.gamma_tether = opt_f64_prop(options, "gammaTether", cfg.gamma_tether)?;
        cfg.recenter = opt_bool_prop(options, "recenter", cfg.recenter)?;
        cfg.energy_tol = opt_f64_prop(options, "energyTol", cfg.energy_tol)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    cfg.validate().map_err(core_err_to_js)?;
    Ok(cfg)
}

/// Output forces and telemetry produced by `ScreenedPoissonField2d` and `ScreenedPoissonRefiner`.
#[wasm_bindgen(js_name = ScreenedPoissonResult)]
#[derive(Debug, Clone)]
pub struct JsScreenedPoissonResult {
    force_x: Vec<f32>,
    force_y: Vec<f32>,
    stats: ScreenedPoissonStats,
}

impl JsScreenedPoissonResult {
    fn from_output(out: ScreenedPoissonOutput<f32>) -> Self {
        Self {
            force_x: out.force_x,
            force_y: out.force_y,
            stats: out.stats,
        }
    }

    fn from_stats(stats: ScreenedPoissonStats) -> Self {
        Self {
            force_x: Vec::new(),
            force_y: Vec::new(),
            stats,
        }
    }
}

#[wasm_bindgen(js_class = ScreenedPoissonResult)]
impl JsScreenedPoissonResult {
    /// Returns the per-node $x$-component descent forces (`Float32Array`).
    #[wasm_bindgen(js_name = forceX)]
    pub fn force_x(&self) -> Vec<f32> {
        self.force_x.clone()
    }

    /// Returns the per-node $y$-component descent forces (`Float32Array`).
    #[wasm_bindgen(js_name = forceY)]
    pub fn force_y(&self) -> Vec<f32> {
        self.force_y.clone()
    }

    /// Discrete Parseval variational energy $\mathcal{L}_{\mathrm{dens}} \ge 0$.
    #[wasm_bindgen(getter)]
    pub fn energy(&self) -> f64 {
        self.stats.energy
    }

    /// Peak smoothed density $\max_p \rho_V(p)$ on the grid.
    #[wasm_bindgen(getter, js_name = maxDensity)]
    pub fn max_density(&self) -> f64 {
        self.stats.max_density
    }

    /// Fraction of active interior grid cells where $\rho_V > \rho_{\max}$.
    #[wasm_bindgen(getter, js_name = excessFraction)]
    pub fn excess_fraction(&self) -> f64 {
        self.stats.excess_fraction
    }

    /// Integrated excess density mass $\int \delta\rho(p)\,dp \in [0, 1]$.
    #[wasm_bindgen(getter, js_name = excessMass)]
    pub fn excess_mass(&self) -> f64 {
        self.stats.excess_mass
    }

    /// Effective Gaussian smoothing width $\sigma_0$ used.
    #[wasm_bindgen(getter, js_name = sigmaUsed)]
    pub fn sigma_used(&self) -> f64 {
        self.stats.sigma_used
    }

    /// Effective screening wavenumber $\kappa_0$ used.
    #[wasm_bindgen(getter, js_name = kappaUsed)]
    pub fn kappa_used(&self) -> f64 {
        self.stats.kappa_used
    }

    /// Effective upper density threshold $\rho_{\max}$ used.
    #[wasm_bindgen(getter, js_name = rhoMaxUsed)]
    pub fn rho_max_used(&self) -> f64 {
        self.stats.rho_max_used
    }

    /// Active interior grid dimension $G$.
    #[wasm_bindgen(getter, js_name = gridDim)]
    pub fn grid_dim(&self) -> usize {
        self.stats.grid_dim
    }

    /// Padded FFT dimension $G_{\mathrm{pad}}$.
    #[wasm_bindgen(getter, js_name = padDim)]
    pub fn pad_dim(&self) -> usize {
        self.stats.pad_dim
    }

    /// Number of 2D R2C/C2R FFT calls executed (`0`, `2`, `4`, or `6`).
    #[wasm_bindgen(getter, js_name = fft2dCalls)]
    pub fn fft_2d_calls(&self) -> f64 {
        self.stats.fft_2d_calls as f64
    }

    /// `true` when zero grid cells exceeded $\rho_{\max}$ and the solve exited early.
    #[wasm_bindgen(getter, js_name = earlyExitNoExcess)]
    pub fn early_exit_no_excess(&self) -> bool {
        self.stats.early_exit_no_excess
    }
}

/// Reusable `f32` R2C-FFT Screened Poisson unilateral upper-bound 2D density field evaluator.
#[wasm_bindgen(js_name = ScreenedPoissonField2d)]
#[derive(Debug, Clone)]
pub struct JsScreenedPoissonField2d {
    inner: ScreenedPoissonField2d<f32>,
}

#[wasm_bindgen(js_class = ScreenedPoissonField2d)]
impl JsScreenedPoissonField2d {
    /// Allocates a `ScreenedPoissonField2d` evaluator workspace (`f32`-backed).
    #[wasm_bindgen(constructor)]
    pub fn new(options: &JsValue) -> Result<JsScreenedPoissonField2d, JsError> {
        let cfg = parse_screened_poisson_config(options, "ScreenedPoissonField2d")?;
        let inner = ScreenedPoissonField2d::<f32>::new(cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Evaluates unilateral density energy and descent forces over `layout`'s adaptive bounding box.
    pub fn evaluate(&mut self, layout: &JsLayout2d) -> Result<JsScreenedPoissonResult, JsError> {
        let soa = layout.inner().to_layout2d();
        let mut out = ScreenedPoissonOutput::new();
        self.inner
            .evaluate(&soa, &mut out)
            .map_err(core_err_to_js)?;
        Ok(JsScreenedPoissonResult::from_output(out))
    }

    /// Evaluates unilateral density energy and descent forces inside explicit `[minX, maxX, minY, maxY]`.
    #[wasm_bindgen(js_name = evaluateInBounds)]
    pub fn evaluate_in_bounds(
        &mut self,
        layout: &JsLayout2d,
        bounds: &[f64],
    ) -> Result<JsScreenedPoissonResult, JsError> {
        if bounds.len() != 4 {
            return Err(JsError::new(&format!(
                "bounds must contain exactly 4 elements [minX, maxX, minY, maxY], got {}",
                bounds.len()
            )));
        }
        let soa = layout.inner().to_layout2d();
        let mut out = ScreenedPoissonOutput::new();
        self.inner
            .evaluate_in_bounds(&soa, [bounds[0], bounds[1], bounds[2], bounds[3]], &mut out)
            .map_err(core_err_to_js)?;
        Ok(JsScreenedPoissonResult::from_output(out))
    }

    /// Returns total heap bytes allocated by internal `f32` grids, bitmasks, and FFT tables.
    #[wasm_bindgen(js_name = workspaceMemoryBytes)]
    pub fn workspace_memory_bytes(&self) -> usize {
        self.inner.workspace_memory_bytes()
    }

    /// Returns the active interior grid dimension $G$.
    #[wasm_bindgen(getter, js_name = gridSize)]
    pub fn grid_size(&self) -> usize {
        self.inner.config().grid_size
    }

    /// Returns the zero-padding multiplier (`1` or `2`).
    #[wasm_bindgen(getter, js_name = paddingFactor)]
    pub fn padding_factor(&self) -> usize {
        self.inner.config().padding_factor
    }

    /// Returns the active kernel name (`"screenedPoisson"` or `"singleGaussian"`).
    #[wasm_bindgen(getter)]
    pub fn kernel(&self) -> String {
        density_field_kernel_js_str(self.inner.config().kernel).to_owned()
    }

    /// Returns whether exact variational adjoint masking is enabled.
    #[wasm_bindgen(getter, js_name = exactAdjoint)]
    pub fn exact_adjoint(&self) -> bool {
        self.inner.config().exact_adjoint
    }
}

/// Cooperative Screened Poisson unilateral upper-bound density field refiner (`f32`-backed).
#[wasm_bindgen(js_name = ScreenedPoissonRefiner)]
#[derive(Debug, Clone)]
pub struct JsScreenedPoissonRefiner {
    inner: ScreenedPoissonRefiner<f32>,
}

#[wasm_bindgen(js_class = ScreenedPoissonRefiner)]
impl JsScreenedPoissonRefiner {
    /// Creates a standalone `ScreenedPoissonRefiner` for `numNodes` vertices.
    #[wasm_bindgen(constructor)]
    pub fn new(num_nodes: f64, options: &JsValue) -> Result<JsScreenedPoissonRefiner, JsError> {
        let n = parse_non_negative_usize(num_nodes, "numNodes")?;
        let cfg = parse_screened_poisson_refiner_config(options, "ScreenedPoissonRefiner")?;
        let inner = ScreenedPoissonRefiner::<f32>::new(n, cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Creates a `ScreenedPoissonRefiner` retaining 1-hop edges from `graph` for optional anchors.
    #[wasm_bindgen(js_name = fromGraph)]
    pub fn from_graph(
        graph: &JsCsrGraph,
        options: &JsValue,
    ) -> Result<JsScreenedPoissonRefiner, JsError> {
        let cfg =
            parse_screened_poisson_refiner_config(options, "ScreenedPoissonRefiner.fromGraph")?;
        let inner = ScreenedPoissonRefiner::<f32>::with_graph(graph.inner(), cfg)
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Attaches or replaces a coarse-level cluster centroid tether (`alignClusteredCentroids`).
    #[wasm_bindgen(js_name = setCoarseTether)]
    pub fn set_coarse_tether(
        &mut self,
        coarse_layout: &JsLayout2d,
        node_to_coarse: &[u32],
    ) -> Result<(), JsError> {
        let coarse_soa = coarse_layout.inner().to_layout2d();
        self.inner
            .set_coarse_tether(coarse_soa, node_to_coarse.to_vec())
            .map_err(core_err_to_js)
    }

    /// Executes one density refinement step on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` density refinement steps on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "ScreenedPoissonRefiner::run_steps",
        )
    }

    /// Runs density refinement to completion or early convergence on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "ScreenedPoissonRefiner::run",
        )
    }

    /// Executes a single decoupled epoch-boundary density step on `layout` in place and returns stats.
    #[wasm_bindgen(js_name = applyEpochStep)]
    pub fn apply_epoch_step(
        &mut self,
        layout: &mut JsLayout2d,
    ) -> Result<JsScreenedPoissonResult, JsError> {
        let mut soa = layout.inner().to_layout2d();
        let stats = self
            .inner
            .apply_epoch_step(&mut soa)
            .map_err(core_err_to_js)?;
        layout
            .inner_mut()
            .copy_from_layout2d(&soa)
            .map_err(ffi_err_to_js)?;
        Ok(JsScreenedPoissonResult::from_stats(stats))
    }

    /// Returns telemetry from the most recent density field evaluation.
    #[wasm_bindgen(js_name = lastStats)]
    pub fn last_stats(&self) -> JsScreenedPoissonResult {
        JsScreenedPoissonResult::from_stats(*self.inner.last_stats())
    }

    /// Most recently evaluated variational density energy.
    #[wasm_bindgen(getter, js_name = lastEnergy)]
    pub fn last_energy(&self) -> f64 {
        self.inner.last_energy()
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    /// Returns the number of refinement iterations completed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the configured maximum number of refinement iterations.
    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> usize {
        self.inner.total_iterations()
    }

    /// Returns `true` when the solver reached `"converged"` or `"completed"`.
    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        self.inner.status().is_finished()
    }

    /// Returns the expected number of vertices in `Layout2d`.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Returns total heap bytes allocated by the underlying density field workspace.
    #[wasm_bindgen(js_name = workspaceMemoryBytes)]
    pub fn workspace_memory_bytes(&self) -> usize {
        self.inner.workspace_memory_bytes()
    }
}
