//! Cooperative Graph UMAP (`GuMap`) layout solver (`f32`) WebAssembly bindings
//! supporting `Layout2d`, `Layout3d`, `LayoutNd`, and `ManifoldLayout2d`.

use egraph::core::{Layout2d, Layout3d, LayoutNd, ManifoldKind, ManifoldLayout2d};
use egraph::layout::{GuMapConfig, GuMapSolver};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::dog_varadhan::JsDogVaradhanLandmarks;
use crate::js_convert::core_err_to_js;
use crate::js_options::{
    JsTargetKind, check_options_arg, opt_bool_prop, opt_f64_prop, opt_maybe_f64_prop,
    opt_seed_prop, opt_target_prop, opt_usize_prop, parse_dog_config_from_options,
};
use crate::neighborhood_types::{JsKnnGraph, JsSparseProbabilities};

const DEFAULT_GUMAP_MAX_2HOP: usize = 32;

#[derive(Debug, Clone)]
pub(crate) enum GuMapInner {
    Dim2(GuMapSolver<Layout2d<f32>>),
    Dim3(GuMapSolver<Layout3d<f32>>),
    DimN(GuMapSolver<LayoutNd<f32>>),
    Manifold(GuMapSolver<ManifoldLayout2d<f32>>),
}

fn parse_gumap_options(
    options: &JsValue,
    ctx: &str,
) -> Result<(GuMapConfig, usize, JsTargetKind), JsValue> {
    let mut cfg = GuMapConfig::default();
    let mut max_2hop = DEFAULT_GUMAP_MAX_2HOP;
    let has_opts = check_options_arg(options, ctx)?;
    let target = if has_opts {
        opt_target_prop(options)?
    } else {
        JsTargetKind::Layout2d
    };
    if has_opts {
        max_2hop = opt_usize_prop(options, "max2hopPerNode", max_2hop)?;
        cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
        cfg.negative_sample_rate =
            opt_usize_prop(options, "negativeSampleRate", cfg.negative_sample_rate)?;
        cfg.learning_rate = opt_f64_prop(options, "learningRate", cfg.learning_rate)?;
        cfg.min_dist = opt_f64_prop(options, "minDist", cfg.min_dist)?;
        cfg.spread = opt_f64_prop(options, "spread", cfg.spread)?;
        cfg.a = opt_maybe_f64_prop(options, "a")?;
        cfg.b = opt_maybe_f64_prop(options, "b")?;
        cfg.sigma_iters = opt_usize_prop(options, "sigmaIters", cfg.sigma_iters)?;
        cfg.sigma_tolerance = opt_f64_prop(options, "sigmaTolerance", cfg.sigma_tolerance)?;
        cfg.symmetric_negative =
            opt_bool_prop(options, "symmetricNegative", cfg.symmetric_negative)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    Ok((cfg, max_2hop, target))
}

/// Cooperative Graph UMAP (`GuMap`) layout solver (`f32`) across 2D, 3D, N-D, and Manifolds.
#[wasm_bindgen(js_name = GuMapSolver)]
#[derive(Debug, Clone)]
pub struct JsGuMapSolver {
    inner: GuMapInner,
    locked_manifold: Option<ManifoldKind>,
}

impl JsGuMapSolver {
    pub(crate) fn inner(&self) -> &GuMapInner {
        &self.inner
    }

    pub(crate) fn inner_mut(&mut self) -> &mut GuMapInner {
        &mut self.inner
    }

    pub(crate) fn locked_manifold_mut(&mut self) -> &mut Option<ManifoldKind> {
        &mut self.locked_manifold
    }
}

#[wasm_bindgen(js_class = GuMapSolver)]
impl JsGuMapSolver {
    /// Constructs a `GuMapSolver` from `graph` using 1+2-hop pairs and DoG Varadhan landmarks.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsGuMapSolver, JsValue> {
        Self::from_graph_dog(graph, options)
    }

    /// Constructs a `GuMapSolver` directly from a fuzzy simplicial set `SparseProbabilities`.
    #[wasm_bindgen(js_name = fromFuzzySet)]
    pub fn from_fuzzy_set(
        probs: &JsSparseProbabilities,
        options: &JsValue,
    ) -> Result<JsGuMapSolver, JsValue> {
        let (cfg, _, target) = parse_gumap_options(options, "GuMapSolver.fromFuzzySet")?;
        let inner = match target {
            JsTargetKind::Layout2d => GuMapInner::Dim2(
                GuMapSolver::<Layout2d<f32>>::from_fuzzy_set_ref(probs.inner(), cfg)
                    .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Layout3d => GuMapInner::Dim3(
                GuMapSolver::<Layout3d<f32>>::from_fuzzy_set_ref(probs.inner(), cfg)
                    .map_err(core_err_to_js)?,
            ),
            JsTargetKind::LayoutNd => GuMapInner::DimN(
                GuMapSolver::<LayoutNd<f32>>::from_fuzzy_set_ref(probs.inner(), cfg)
                    .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Manifold2d => GuMapInner::Manifold(
                GuMapSolver::<ManifoldLayout2d<f32>>::from_fuzzy_set_ref(probs.inner(), cfg)
                    .map_err(core_err_to_js)?,
            ),
        };
        Ok(Self {
            inner,
            locked_manifold: None,
        })
    }

    /// Constructs a `GuMapSolver` from a `KnnGraph` via `computeFuzzySimplicialSet`.
    #[wasm_bindgen(js_name = fromKnn)]
    pub fn from_knn(knn: &JsKnnGraph, options: &JsValue) -> Result<JsGuMapSolver, JsValue> {
        let (cfg, _, target) = parse_gumap_options(options, "GuMapSolver.fromKnn")?;
        let inner = match target {
            JsTargetKind::Layout2d => GuMapInner::Dim2(
                GuMapSolver::<Layout2d<f32>>::from_knn(knn.inner(), cfg).map_err(core_err_to_js)?,
            ),
            JsTargetKind::Layout3d => GuMapInner::Dim3(
                GuMapSolver::<Layout3d<f32>>::from_knn(knn.inner(), cfg).map_err(core_err_to_js)?,
            ),
            JsTargetKind::LayoutNd => GuMapInner::DimN(
                GuMapSolver::<LayoutNd<f32>>::from_knn(knn.inner(), cfg).map_err(core_err_to_js)?,
            ),
            JsTargetKind::Manifold2d => GuMapInner::Manifold(
                GuMapSolver::<ManifoldLayout2d<f32>>::from_knn(knn.inner(), cfg)
                    .map_err(core_err_to_js)?,
            ),
        };
        Ok(Self {
            inner,
            locked_manifold: None,
        })
    }

    /// Constructs a `GuMapSolver` from a `CsrGraph` using 1+2-hop pairs and DoG Varadhan landmarks.
    #[wasm_bindgen(js_name = fromGraphDog)]
    pub fn from_graph_dog(graph: &JsCsrGraph, options: &JsValue) -> Result<JsGuMapSolver, JsValue> {
        let (cfg, max_2hop, target) = parse_gumap_options(options, "GuMapSolver.fromGraphDog")?;
        let mut dog_cfg = parse_dog_config_from_options(options, "GuMapSolver.fromGraphDog")?;
        dog_cfg.seed = cfg.seed;
        let inner = match target {
            JsTargetKind::Layout2d => GuMapInner::Dim2(
                GuMapSolver::<Layout2d<f32>>::from_graph_dog(
                    graph.inner(),
                    max_2hop,
                    &dog_cfg,
                    cfg,
                )
                .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Layout3d => GuMapInner::Dim3(
                GuMapSolver::<Layout3d<f32>>::from_graph_dog(
                    graph.inner(),
                    max_2hop,
                    &dog_cfg,
                    cfg,
                )
                .map_err(core_err_to_js)?,
            ),
            JsTargetKind::LayoutNd => GuMapInner::DimN(
                GuMapSolver::<LayoutNd<f32>>::from_graph_dog(
                    graph.inner(),
                    max_2hop,
                    &dog_cfg,
                    cfg,
                )
                .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Manifold2d => GuMapInner::Manifold(
                GuMapSolver::<ManifoldLayout2d<f32>>::from_graph_dog(
                    graph.inner(),
                    max_2hop,
                    &dog_cfg,
                    cfg,
                )
                .map_err(core_err_to_js)?,
            ),
        };
        Ok(Self {
            inner,
            locked_manifold: None,
        })
    }

    /// Constructs a `GuMapSolver` from precomputed `DogVaradhanLandmarks` and 1+2-hop priority CSR.
    #[wasm_bindgen(js_name = fromLandmarks)]
    pub fn from_landmarks(
        graph: &JsCsrGraph,
        landmarks: &JsDogVaradhanLandmarks,
        options: &JsValue,
    ) -> Result<JsGuMapSolver, JsValue> {
        let (cfg, max_2hop, target) = parse_gumap_options(options, "GuMapSolver.fromLandmarks")?;
        let inner = match target {
            JsTargetKind::Layout2d => GuMapInner::Dim2(
                GuMapSolver::<Layout2d<f32>>::from_landmarks(
                    graph.inner(),
                    landmarks.inner(),
                    max_2hop,
                    cfg,
                )
                .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Layout3d => GuMapInner::Dim3(
                GuMapSolver::<Layout3d<f32>>::from_landmarks(
                    graph.inner(),
                    landmarks.inner(),
                    max_2hop,
                    cfg,
                )
                .map_err(core_err_to_js)?,
            ),
            JsTargetKind::LayoutNd => GuMapInner::DimN(
                GuMapSolver::<LayoutNd<f32>>::from_landmarks(
                    graph.inner(),
                    landmarks.inner(),
                    max_2hop,
                    cfg,
                )
                .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Manifold2d => GuMapInner::Manifold(
                GuMapSolver::<ManifoldLayout2d<f32>>::from_landmarks(
                    graph.inner(),
                    landmarks.inner(),
                    max_2hop,
                    cfg,
                )
                .map_err(core_err_to_js)?,
            ),
        };
        Ok(Self {
            inner,
            locked_manifold: None,
        })
    }
}
