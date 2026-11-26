//! WebAssembly `SgdSolver` baseline constructors (`ortmannSparseStress`, `largevis`, `drgraph`, `graphCne`).

use egraph::layout::{
    DrGraphConfig, GraphCneConfig, GraphCneObjective, LargeVisConfig, OrtmannPivotStrategy,
    OrtmannSparseStressConfig, SgdSolver,
};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::core_err_to_js;
use crate::js_options::{
    check_options_arg, normalize_enum_token, opt_bool_prop, opt_f64_prop, opt_maybe_f64_prop,
    opt_seed_prop, opt_string_prop, opt_usize_prop,
};
use crate::sgd::JsSgdSolver;

fn parse_ortmann_pivot_strategy(raw: &str) -> Result<OrtmannPivotStrategy, JsError> {
    match normalize_enum_token(raw).as_str() {
        "kmeanssp" => Ok(OrtmannPivotStrategy::KMeansSp),
        "maxminrandomsp" => Ok(OrtmannPivotStrategy::MaxMinRandomSp),
        "maxminsp" => Ok(OrtmannPivotStrategy::MaxMinSp),
        _ => Err(JsError::new(&format!(
            "unknown OrtmannPivotStrategy {raw:?}; expected 'kmeans_sp', 'maxmin_random_sp', or 'maxmin_sp'"
        ))),
    }
}

fn parse_graph_cne_objective(raw: &str) -> Result<GraphCneObjective, JsError> {
    match normalize_enum_token(raw).as_str() {
        "infonce" => Ok(GraphCneObjective::InfoNce),
        "negtsne" => Ok(GraphCneObjective::NegTSne),
        _ => Err(JsError::new(&format!(
            "unknown GraphCneObjective {raw:?}; expected 'infonce' or 'neg_tsne'"
        ))),
    }
}

#[wasm_bindgen(js_class = SgdSolver)]
impl JsSgdSolver {
    /// Constructs an Ortmann et al. (2016) Sparse Stress SGD solver.
    #[wasm_bindgen(js_name = ortmannSparseStress)]
    pub fn ortmann_sparse_stress(
        graph: &JsCsrGraph,
        options: &JsValue,
    ) -> Result<JsSgdSolver, JsError> {
        let mut cfg = OrtmannSparseStressConfig::default();
        if check_options_arg(options, "SgdSolver.ortmannSparseStress")? {
            cfg.num_pivots = opt_usize_prop(options, "numPivots", cfg.num_pivots)?;
            if let Some(strat_str) = opt_string_prop(options, "pivotStrategy")? {
                cfg.pivot_strategy = parse_ortmann_pivot_strategy(&strat_str)?;
            }
            cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
            cfg.eta_max = opt_maybe_f64_prop(options, "etaMax")?;
            cfg.eta_min = opt_maybe_f64_prop(options, "etaMin")?;
            cfg.eps = opt_f64_prop(options, "eps", cfg.eps)?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner =
            SgdSolver::<f32>::ortmann_sparse_stress(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(JsSgdSolver::from_inner(inner))
    }

    /// Constructs a single-threaded deterministic `LargeVis` (Tang et al., 2016) solver.
    pub fn largevis(graph: &JsCsrGraph, options: &JsValue) -> Result<JsSgdSolver, JsError> {
        let mut cfg = LargeVisConfig::default();
        if check_options_arg(options, "SgdSolver.largevis")? {
            cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
            cfg.negative_samples =
                opt_usize_prop(options, "negativeSamples", cfg.negative_samples)?;
            cfg.gamma = opt_f64_prop(options, "gamma", cfg.gamma)?;
            cfg.initial_lr = opt_f64_prop(options, "initialLr", cfg.initial_lr)?;
            cfg.min_lr = opt_f64_prop(options, "minLr", cfg.min_lr)?;
            cfg.degree_exponent = opt_f64_prop(options, "degreeExponent", cfg.degree_exponent)?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner = SgdSolver::<f32>::largevis(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(JsSgdSolver::from_inner(inner))
    }

    /// Constructs a multilevel negative-sampling `DRGraph` (Zhu et al., 2020) solver.
    pub fn drgraph(graph: &JsCsrGraph, options: &JsValue) -> Result<JsSgdSolver, JsError> {
        let mut cfg = DrGraphConfig::default();
        if check_options_arg(options, "SgdSolver.drgraph")? {
            cfg.k_order = opt_usize_prop(options, "kOrder", cfg.k_order)?;
            cfg.negative_samples =
                opt_usize_prop(options, "negativeSamples", cfg.negative_samples)?;
            cfg.b = opt_f64_prop(options, "b", cfg.b)?;
            cfg.gamma_coarse = opt_f64_prop(options, "gammaCoarse", cfg.gamma_coarse)?;
            cfg.gamma_fine = opt_f64_prop(options, "gammaFine", cfg.gamma_fine)?;
            cfg.coarsen_stop_ratio =
                opt_f64_prop(options, "coarsenStopRatio", cfg.coarsen_stop_ratio)?;
            cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
            cfg.initial_lr = opt_f64_prop(options, "initialLr", cfg.initial_lr)?;
            cfg.min_lr = opt_f64_prop(options, "minLr", cfg.min_lr)?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner = SgdSolver::<f32>::drgraph(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(JsSgdSolver::from_inner(inner))
    }

    /// Constructs a `Graph-CNE` (Bohm et al., 2025) contrastive neighbor embedding solver.
    #[wasm_bindgen(js_name = graphCne)]
    pub fn graph_cne(graph: &JsCsrGraph, options: &JsValue) -> Result<JsSgdSolver, JsError> {
        let mut cfg = GraphCneConfig::default();
        if check_options_arg(options, "SgdSolver.graphCne")? {
            if let Some(obj_str) = opt_string_prop(options, "objective")? {
                cfg.objective = parse_graph_cne_objective(&obj_str)?;
            }
            cfg.negative_samples =
                opt_usize_prop(options, "negativeSamples", cfg.negative_samples)?;
            cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
            cfg.initial_lr = opt_f64_prop(options, "initialLr", cfg.initial_lr)?;
            cfg.min_lr = opt_f64_prop(options, "minLr", cfg.min_lr)?;
            cfg.repulsion_weight = opt_f64_prop(options, "repulsionWeight", cfg.repulsion_weight)?;
            cfg.row_normalize = opt_bool_prop(options, "rowNormalize", cfg.row_normalize)?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner = SgdSolver::<f32>::graph_cne(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(JsSgdSolver::from_inner(inner))
    }
}
