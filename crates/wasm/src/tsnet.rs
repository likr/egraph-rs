//! Cooperative unified `TsNetSolver` (`f32`) WebAssembly bindings supporting
//! Policy 1A + Option 1 `(target, backend)` dispatch across 2D, 3D, N-D, and Manifolds.

use egraph::core::{Layout2d, Layout3d, LayoutNd, ManifoldKind, ManifoldLayout2d};
use egraph::layout::{
    TsNetConfig, TsNetSolver, TsNetStage,
    neighbor::{Exact as NeighborExact, TsNetBackend2d, TsNetBackend3d},
};
use egraph::neighborhood::TwoHopSamplingMode;
use egraph::spatial::FitFieldStats;
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, parse_f64_sequence};
use crate::js_options::{
    check_options_arg, get_prop, opt_f64_prop, opt_maybe_usize_prop, opt_usize_prop,
    solver_status_str,
};
use crate::neighborhood_types::{JsKnnGraph, JsSparseProbabilities};
use crate::tsnet_options::{
    TsNetBackendChoice, parse_tsnet_options, tsnet_resolved_backend_2d, tsnet_resolved_backend_3d,
    tsnet_sampling_mode_str,
};

const DEFAULT_UNIFORM_PERPLEXITY: f64 = 30.0;

#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum TsNetInner {
    Dim2(TsNetSolver<Layout2d<f32>, TsNetBackend2d>),
    Dim3(TsNetSolver<Layout3d<f32>, TsNetBackend3d>),
    DimN(TsNetSolver<LayoutNd<f32>, NeighborExact>),
    Manifold(TsNetSolver<ManifoldLayout2d<f32>, NeighborExact>),
}

/// FFT field telemetry counters returned by `TsNetSolver.lastFftStats()`.
#[wasm_bindgen(js_name = FitFieldStats)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsFitFieldStats {
    inner: FitFieldStats,
}

#[wasm_bindgen(js_class = FitFieldStats)]
impl JsFitFieldStats {
    #[wasm_bindgen(getter, js_name = particleForwardFfts)]
    pub fn particle_forward_ffts(&self) -> f64 {
        self.inner.particle_forward_ffts as f64
    }

    #[wasm_bindgen(getter, js_name = kernelForwardFfts)]
    pub fn kernel_forward_ffts(&self) -> f64 {
        self.inner.kernel_forward_ffts as f64
    }

    #[wasm_bindgen(getter, js_name = inverseFfts)]
    pub fn inverse_ffts(&self) -> f64 {
        self.inner.inverse_ffts as f64
    }

    #[wasm_bindgen(getter, js_name = totalFft2dCalls)]
    pub fn total_fft_2d_calls(&self) -> f64 {
        self.inner.total_fft_2d_calls as f64
    }
}

/// Cooperative unified tsNET / BH-tsNET / FIt-tsNET / L-tsNET solver (`f32`).
#[wasm_bindgen(js_name = TsNetSolver)]
#[derive(Debug, Clone)]
pub struct JsTsNetSolver {
    inner: TsNetInner,
    sampling_mode: Option<TwoHopSamplingMode>,
    locked_manifold: Option<ManifoldKind>,
}

impl JsTsNetSolver {
    pub(crate) fn from_parts(inner: TsNetInner, sampling_mode: Option<TwoHopSamplingMode>) -> Self {
        Self {
            inner,
            sampling_mode,
            locked_manifold: None,
        }
    }

    pub(crate) fn inner_mut(&mut self) -> &mut TsNetInner {
        &mut self.inner
    }

    pub(crate) fn locked_manifold_mut(&mut self) -> &mut Option<ManifoldKind> {
        &mut self.locked_manifold
    }

    pub(crate) fn build_probabilities(
        probs: &JsSparseProbabilities,
        cfg: TsNetConfig,
        backend: TsNetBackendChoice,
    ) -> Result<TsNetInner, JsValue> {
        let p = probs.inner().clone();
        match backend {
            TsNetBackendChoice::Dim2(b) => TsNetSolver::<Layout2d<f32>, TsNetBackend2d>::from_probabilities_with_backend(p, cfg, b).map(TsNetInner::Dim2),
            TsNetBackendChoice::Dim3(b) => TsNetSolver::<Layout3d<f32>, TsNetBackend3d>::from_probabilities_with_backend(p, cfg, b).map(TsNetInner::Dim3),
            TsNetBackendChoice::DimN(b) => TsNetSolver::<LayoutNd<f32>, NeighborExact>::from_probabilities_with_backend(p, cfg, b).map(TsNetInner::DimN),
            TsNetBackendChoice::Manifold(b) => TsNetSolver::<ManifoldLayout2d<f32>, NeighborExact>::from_probabilities_with_backend(p, cfg, b).map(TsNetInner::Manifold),
        }
        .map_err(core_err_to_js)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_row_normalized(
        graph: &JsCsrGraph,
        w2: f64,
        max_2hop: Option<usize>,
        degree_beta: f64,
        sampling_mode: TwoHopSamplingMode,
        max_bridge_degree: Option<usize>,
        cfg: TsNetConfig,
        backend: TsNetBackendChoice,
    ) -> Result<TsNetInner, JsValue> {
        let g = graph.inner();
        match backend {
            TsNetBackendChoice::Dim2(b) => TsNetSolver::<Layout2d<f32>, TsNetBackend2d>::from_graph_row_normalized_with_options_with_backend(
                g, w2, max_2hop, degree_beta, sampling_mode, max_bridge_degree, cfg, b,
            ).map(TsNetInner::Dim2),
            TsNetBackendChoice::Dim3(b) => TsNetSolver::<Layout3d<f32>, TsNetBackend3d>::from_graph_row_normalized_with_options_with_backend(
                g, w2, max_2hop, degree_beta, sampling_mode, max_bridge_degree, cfg, b,
            ).map(TsNetInner::Dim3),
            TsNetBackendChoice::DimN(b) => TsNetSolver::<LayoutNd<f32>, NeighborExact>::from_graph_row_normalized_with_options_with_backend(
                g, w2, max_2hop, degree_beta, sampling_mode, max_bridge_degree, cfg, b,
            ).map(TsNetInner::DimN),
            TsNetBackendChoice::Manifold(b) => TsNetSolver::<ManifoldLayout2d<f32>, NeighborExact>::from_graph_row_normalized_with_options_with_backend(
                g, w2, max_2hop, degree_beta, sampling_mode, max_bridge_degree, cfg, b,
            ).map(TsNetInner::Manifold),
        }
        .map_err(core_err_to_js)
    }
}

#[wasm_bindgen(js_class = TsNetSolver)]
impl JsTsNetSolver {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsTsNetSolver, JsValue> {
        Self::pure_2hop(graph, options)
    }

    #[wasm_bindgen(js_name = pure2Hop)]
    pub fn pure_2hop(graph: &JsCsrGraph, options: &JsValue) -> Result<JsTsNetSolver, JsValue> {
        let (cfg, _, backend) = parse_tsnet_options(options, "TsNetSolver.pure2Hop")?;
        let g = graph.inner();
        let inner =
            match backend {
                TsNetBackendChoice::Dim2(b) => {
                    TsNetSolver::<Layout2d<f32>, TsNetBackend2d>::from_graph_pure_2hop_with_backend(
                        g, cfg, b,
                    )
                    .map(TsNetInner::Dim2)
                }
                TsNetBackendChoice::Dim3(b) => {
                    TsNetSolver::<Layout3d<f32>, TsNetBackend3d>::from_graph_pure_2hop_with_backend(
                        g, cfg, b,
                    )
                    .map(TsNetInner::Dim3)
                }
                TsNetBackendChoice::DimN(b) => {
                    TsNetSolver::<LayoutNd<f32>, NeighborExact>::from_graph_pure_2hop_with_backend(
                        g, cfg, b,
                    )
                    .map(TsNetInner::DimN)
                }
                TsNetBackendChoice::Manifold(b) => TsNetSolver::<
                    ManifoldLayout2d<f32>,
                    NeighborExact,
                >::from_graph_pure_2hop_with_backend(
                    g, cfg, b
                )
                .map(TsNetInner::Manifold),
            }
            .map_err(core_err_to_js)?;
        Ok(Self::from_parts(inner, None))
    }

    #[wasm_bindgen(js_name = fromKnn)]
    pub fn from_knn(knn: &JsKnnGraph, options: &JsValue) -> Result<JsTsNetSolver, JsValue> {
        let (cfg, _, backend) = parse_tsnet_options(options, "TsNetSolver.fromKnn")?;
        let has_opts = check_options_arg(options, "TsNetSolver.fromKnn")?;
        let uni = if has_opts {
            opt_f64_prop(options, "uniformPerplexity", DEFAULT_UNIFORM_PERPLEXITY)?
        } else {
            DEFAULT_UNIFORM_PERPLEXITY
        };
        let perplexities: Option<Vec<f64>> = if has_opts {
            let p_val = get_prop(options, "perplexities")?;
            if p_val.is_undefined() || p_val.is_null() {
                None
            } else {
                Some(parse_f64_sequence(&p_val, "perplexities")?)
            }
        } else {
            None
        };
        let p_ref = perplexities.as_deref();
        let k = knn.inner();
        let inner = match backend {
            TsNetBackendChoice::Dim2(b) => {
                TsNetSolver::<Layout2d<f32>, TsNetBackend2d>::from_knn_with_backend(
                    k, p_ref, uni, cfg, b,
                )
                .map(TsNetInner::Dim2)
            }
            TsNetBackendChoice::Dim3(b) => {
                TsNetSolver::<Layout3d<f32>, TsNetBackend3d>::from_knn_with_backend(
                    k, p_ref, uni, cfg, b,
                )
                .map(TsNetInner::Dim3)
            }
            TsNetBackendChoice::DimN(b) => {
                TsNetSolver::<LayoutNd<f32>, NeighborExact>::from_knn_with_backend(
                    k, p_ref, uni, cfg, b,
                )
                .map(TsNetInner::DimN)
            }
            TsNetBackendChoice::Manifold(b) => {
                TsNetSolver::<ManifoldLayout2d<f32>, NeighborExact>::from_knn_with_backend(
                    k, p_ref, uni, cfg, b,
                )
                .map(TsNetInner::Manifold)
            }
        }
        .map_err(core_err_to_js)?;
        Ok(Self::from_parts(inner, None))
    }

    #[wasm_bindgen(js_name = fromProbabilities)]
    pub fn from_probabilities(
        probs: &JsSparseProbabilities,
        options: &JsValue,
    ) -> Result<JsTsNetSolver, JsValue> {
        let (cfg, _, backend) = parse_tsnet_options(options, "TsNetSolver.fromProbabilities")?;
        let inner = Self::build_probabilities(probs, cfg, backend)?;
        Ok(Self::from_parts(inner, None))
    }

    #[wasm_bindgen(js_name = fromGraphPartialBfs)]
    pub fn from_graph_partial_bfs(
        graph: &JsCsrGraph,
        options: &JsValue,
    ) -> Result<JsTsNetSolver, JsValue> {
        let (cfg, _, backend) = parse_tsnet_options(options, "TsNetSolver.fromGraphPartialBfs")?;
        let has_opts = check_options_arg(options, "TsNetSolver.fromGraphPartialBfs")?;
        let k_val = if has_opts {
            opt_usize_prop(options, "k", 30)?
        } else {
            30
        };
        let uni = if has_opts {
            opt_f64_prop(options, "uniformPerplexity", DEFAULT_UNIFORM_PERPLEXITY)?
        } else {
            DEFAULT_UNIFORM_PERPLEXITY
        };
        let g = graph.inner();
        let inner = match backend {
            TsNetBackendChoice::Dim2(b) => TsNetSolver::<Layout2d<f32>, TsNetBackend2d>::from_graph_partial_bfs_with_backend(g, k_val, uni, cfg, b).map(TsNetInner::Dim2),
            TsNetBackendChoice::Dim3(b) => TsNetSolver::<Layout3d<f32>, TsNetBackend3d>::from_graph_partial_bfs_with_backend(g, k_val, uni, cfg, b).map(TsNetInner::Dim3),
            TsNetBackendChoice::DimN(b) => TsNetSolver::<LayoutNd<f32>, NeighborExact>::from_graph_partial_bfs_with_backend(g, k_val, uni, cfg, b).map(TsNetInner::DimN),
            TsNetBackendChoice::Manifold(b) => TsNetSolver::<ManifoldLayout2d<f32>, NeighborExact>::from_graph_partial_bfs_with_backend(g, k_val, uni, cfg, b).map(TsNetInner::Manifold),
        }.map_err(core_err_to_js)?;
        Ok(Self::from_parts(inner, None))
    }

    #[wasm_bindgen(js_name = fromGraphRowNormalized)]
    pub fn from_graph_row_normalized(
        graph: &JsCsrGraph,
        options: &JsValue,
    ) -> Result<JsTsNetSolver, JsValue> {
        let mut w2 = 0.5_f64;
        let mut max_2hop = None;
        let mut degree_beta = 0.0_f64;
        let mut sampling_mode = TwoHopSamplingMode::TwoStepRandomWalk;
        let mut max_bridge_degree = None;
        if check_options_arg(options, "TsNetSolver.fromGraphRowNormalized")? {
            w2 = opt_f64_prop(options, "w2", w2)?;
            max_2hop = opt_maybe_usize_prop(options, "max2hopPerNode")?;
            degree_beta = opt_f64_prop(options, "degreeBeta", degree_beta)?;
            sampling_mode =
                crate::tsnet_options::parse_sampling_mode_from_options(options, sampling_mode)?;
            max_bridge_degree = opt_maybe_usize_prop(options, "maxBridgeDegree")?;
        }
        let (cfg, _, backend) = parse_tsnet_options(options, "TsNetSolver.fromGraphRowNormalized")?;
        let inner = Self::build_row_normalized(
            graph,
            w2,
            max_2hop,
            degree_beta,
            sampling_mode,
            max_bridge_degree,
            cfg,
            backend,
        )?;
        Ok(Self::from_parts(inner, Some(sampling_mode)))
    }

    #[wasm_bindgen(getter, js_name = samplingMode)]
    pub fn sampling_mode(&self) -> Option<String> {
        self.sampling_mode
            .map(|mode| tsnet_sampling_mode_str(mode).to_owned())
    }

    #[wasm_bindgen(js_name = lastFftStats)]
    pub fn last_fft_stats(&self) -> Option<JsFitFieldStats> {
        let st = match &self.inner {
            TsNetInner::Dim2(s) => s.last_fft_stats().cloned(),
            TsNetInner::Dim3(s) => s.last_fft_stats().cloned(),
            TsNetInner::DimN(s) => s.last_fft_stats().cloned(),
            TsNetInner::Manifold(s) => s.last_fft_stats().cloned(),
        };
        st.map(|inner| JsFitFieldStats { inner })
    }

    #[wasm_bindgen(getter)]
    pub fn target(&self) -> String {
        match &self.inner {
            TsNetInner::Dim2(_) => "layout2d",
            TsNetInner::Dim3(_) => "layout3d",
            TsNetInner::DimN(_) => "layoutNd",
            TsNetInner::Manifold(_) => "manifold2d",
        }
        .to_owned()
    }

    #[wasm_bindgen(getter, js_name = resolvedBackend)]
    pub fn resolved_backend(&self) -> String {
        match &self.inner {
            TsNetInner::Dim2(s) => tsnet_resolved_backend_2d(s.backend(), s.num_nodes()),
            TsNetInner::Dim3(s) => tsnet_resolved_backend_3d(s.backend(), s.num_nodes()),
            TsNetInner::DimN(_) | TsNetInner::Manifold(_) => "exact",
        }
        .to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn backend(&self) -> String {
        match &self.inner {
            TsNetInner::Dim2(s) => match s.backend() {
                TsNetBackend2d::Exact(_) => "exact",
                TsNetBackend2d::BarnesHut(_) => "barnesHut",
                TsNetBackend2d::FitFft(_) => "fitFft",
                TsNetBackend2d::LinearFft(_) => "linearFft",
                TsNetBackend2d::Auto(_) => "auto2d",
                _ => "auto2d",
            },
            TsNetInner::Dim3(s) => match s.backend() {
                TsNetBackend3d::Exact(_) => "exact",
                TsNetBackend3d::BarnesHut(_) => "barnesHut",
                TsNetBackend3d::Auto(_) => "auto3d",
                _ => "auto3d",
            },
            TsNetInner::DimN(_) | TsNetInner::Manifold(_) => "exact",
        }
        .to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        let st = match &self.inner {
            TsNetInner::Dim2(s) => s.status(),
            TsNetInner::Dim3(s) => s.status(),
            TsNetInner::DimN(s) => s.status(),
            TsNetInner::Manifold(s) => s.status(),
        };
        solver_status_str(st).to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        match &self.inner {
            TsNetInner::Dim2(s) => s.iteration(),
            TsNetInner::Dim3(s) => s.iteration(),
            TsNetInner::DimN(s) => s.iteration(),
            TsNetInner::Manifold(s) => s.iteration(),
        }
    }

    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> Option<usize> {
        match &self.inner {
            TsNetInner::Dim2(s) => s.total_iterations(),
            TsNetInner::Dim3(s) => s.total_iterations(),
            TsNetInner::DimN(s) => s.total_iterations(),
            TsNetInner::Manifold(s) => s.total_iterations(),
        }
    }

    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        let st = match &self.inner {
            TsNetInner::Dim2(s) => s.status(),
            TsNetInner::Dim3(s) => s.status(),
            TsNetInner::DimN(s) => s.status(),
            TsNetInner::Manifold(s) => s.status(),
        };
        st.is_finished()
    }

    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        match &self.inner {
            TsNetInner::Dim2(s) => s.num_nodes(),
            TsNetInner::Dim3(s) => s.num_nodes(),
            TsNetInner::DimN(s) => s.num_nodes(),
            TsNetInner::Manifold(s) => s.num_nodes(),
        }
    }

    #[wasm_bindgen(getter, js_name = stageIndex)]
    pub fn stage_index(&self) -> usize {
        match &self.inner {
            TsNetInner::Dim2(s) => s.stage_index(),
            TsNetInner::Dim3(s) => s.stage_index(),
            TsNetInner::DimN(s) => s.stage_index(),
            TsNetInner::Manifold(s) => s.stage_index(),
        }
    }

    #[wasm_bindgen(getter, js_name = totalFft2dCalls)]
    pub fn total_fft_2d_calls(&self) -> f64 {
        match &self.inner {
            TsNetInner::Dim2(s) => s.total_fft_2d_calls() as f64,
            TsNetInner::Dim3(s) => s.total_fft_2d_calls() as f64,
            TsNetInner::DimN(s) => s.total_fft_2d_calls() as f64,
            TsNetInner::Manifold(s) => s.total_fft_2d_calls() as f64,
        }
    }

    #[wasm_bindgen(getter, js_name = currentStageLr)]
    pub fn current_stage_lr(&self) -> f64 {
        match &self.inner {
            TsNetInner::Dim2(s) => s.current_stage_lr(),
            TsNetInner::Dim3(s) => s.current_stage_lr(),
            TsNetInner::DimN(s) => s.current_stage_lr(),
            TsNetInner::Manifold(s) => s.current_stage_lr(),
        }
    }
}

pub(crate) fn _keep_stage_import(_: TsNetStage) {}
