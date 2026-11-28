//! WebAssembly `SgdSolver` Ortmann et al. (2016) Sparse Stress constructor.

use egraph::core::{Layout2d, Layout3d, LayoutNd, ManifoldLayout2d};
use egraph::layout::{OrtmannPivotStrategy, OrtmannSparseStressConfig, SgdSolver};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, invalid_input_err};
use crate::js_options::{
    JsTargetKind, check_options_arg, normalize_enum_token, opt_f64_prop, opt_maybe_f64_prop,
    opt_seed_prop, opt_string_prop, opt_target_prop, opt_usize_prop,
};
use crate::sgd::{JsSgdSolver, SgdInner};

fn parse_ortmann_pivot_strategy(raw: &str) -> Result<OrtmannPivotStrategy, JsValue> {
    match normalize_enum_token(raw)?.as_str() {
        "kmeanssp" => Ok(OrtmannPivotStrategy::KMeansSp),
        "maxminrandomsp" => Ok(OrtmannPivotStrategy::MaxMinRandomSp),
        "maxminsp" => Ok(OrtmannPivotStrategy::MaxMinSp),
        _ => Err(invalid_input_err(format!(
            "unknown OrtmannPivotStrategy {raw:?}; expected 'kmeansSp', 'maxminRandomSp', or 'maxminSp'"
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
    ) -> Result<JsSgdSolver, JsValue> {
        let mut cfg = OrtmannSparseStressConfig::default();
        let mut target = JsTargetKind::Layout2d;
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
            target = opt_target_prop(options)?;
        }
        let inner = match target {
            JsTargetKind::Layout2d => SgdInner::Dim2(
                SgdSolver::<Layout2d<f32>>::ortmann_sparse_stress(graph.inner(), cfg)
                    .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Layout3d => SgdInner::Dim3(
                SgdSolver::<Layout3d<f32>>::ortmann_sparse_stress(graph.inner(), cfg)
                    .map_err(core_err_to_js)?,
            ),
            JsTargetKind::LayoutNd => SgdInner::DimN(
                SgdSolver::<LayoutNd<f32>>::ortmann_sparse_stress(graph.inner(), cfg)
                    .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Manifold2d => SgdInner::Manifold(
                SgdSolver::<ManifoldLayout2d<f32>>::ortmann_sparse_stress(graph.inner(), cfg)
                    .map_err(core_err_to_js)?,
            ),
        };
        Ok(JsSgdSolver::from_inner(inner))
    }
}
