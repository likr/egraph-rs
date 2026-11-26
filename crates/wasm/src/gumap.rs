//! Cooperative Graph UMAP (`GuMap`) layout solver (`f32`) WebAssembly bindings.

use egraph::layout::{GuMapConfig, GuMapSolver};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::dog_varadhan::JsDogVaradhanLandmarks;
use crate::js_convert::{
    core_err_to_js, parse_non_negative_usize, parse_optional_usize, parse_seed_u64,
};
use crate::js_options::{
    is_plain_js_object, opt_bool_prop, opt_f64_prop, opt_maybe_f64_prop, opt_seed_prop,
    opt_usize_prop, parse_dog_config_from_options_or_args, reject_extra_args_with_options,
    solver_status_str,
};
use crate::layout_buffers::JsLayout2d;
use crate::neighborhood_types::{JsKnnGraph, JsSparseProbabilities};
use crate::solver_runner::{run_layout_in_place, run_steps_layout_in_place, step_layout_in_place};

const DEFAULT_GUMAP_MAX_2HOP: usize = 32;

#[allow(clippy::too_many_arguments)]
fn parse_gumap_config(
    n_epochs_or_options: &JsValue,
    negative_sample_rate: Option<f64>,
    learning_rate: Option<f64>,
    min_dist: Option<f64>,
    spread: Option<f64>,
    a: Option<f64>,
    b: Option<f64>,
    sigma_iters: Option<f64>,
    sigma_tolerance: Option<f64>,
    seed: Option<f64>,
    has_extra_args: bool,
) -> Result<GuMapConfig, JsError> {
    let mut cfg = GuMapConfig::default();
    if is_plain_js_object(n_epochs_or_options) {
        let has_extra = has_extra_args
            || negative_sample_rate.is_some()
            || learning_rate.is_some()
            || min_dist.is_some()
            || spread.is_some()
            || a.is_some()
            || b.is_some()
            || sigma_iters.is_some()
            || sigma_tolerance.is_some()
            || seed.is_some();
        reject_extra_args_with_options(has_extra)?;

        cfg.iterations = opt_usize_prop(n_epochs_or_options, "nEpochs", cfg.iterations)?;
        cfg.negative_sample_rate = opt_usize_prop(
            n_epochs_or_options,
            "negativeSampleRate",
            cfg.negative_sample_rate,
        )?;
        cfg.learning_rate = opt_f64_prop(n_epochs_or_options, "learningRate", cfg.learning_rate)?;
        cfg.min_dist = opt_f64_prop(n_epochs_or_options, "minDist", cfg.min_dist)?;
        cfg.spread = opt_f64_prop(n_epochs_or_options, "spread", cfg.spread)?;
        cfg.a = opt_maybe_f64_prop(n_epochs_or_options, "a")?;
        cfg.b = opt_maybe_f64_prop(n_epochs_or_options, "b")?;
        cfg.sigma_iters = opt_usize_prop(n_epochs_or_options, "sigmaIters", cfg.sigma_iters)?;
        cfg.sigma_tolerance =
            opt_f64_prop(n_epochs_or_options, "sigmaTolerance", cfg.sigma_tolerance)?;
        cfg.symmetric_negative = opt_bool_prop(
            n_epochs_or_options,
            "symmetricNegative",
            cfg.symmetric_negative,
        )?;
        cfg.seed = opt_seed_prop(n_epochs_or_options)?;
        return Ok(cfg);
    }

    if !n_epochs_or_options.is_undefined() && !n_epochs_or_options.is_null() {
        let v = n_epochs_or_options.as_f64().ok_or_else(|| {
            JsError::new("nEpochsOrOptions must be a non-negative integer, options object, null, or undefined")
        })?;
        cfg.iterations = parse_non_negative_usize(v, "nEpochs")?;
    }
    cfg.negative_sample_rate = parse_optional_usize(
        negative_sample_rate,
        cfg.negative_sample_rate,
        "negativeSampleRate",
    )?;
    cfg.learning_rate = learning_rate.unwrap_or(cfg.learning_rate);
    cfg.min_dist = min_dist.unwrap_or(cfg.min_dist);
    cfg.spread = spread.unwrap_or(cfg.spread);
    cfg.a = a;
    cfg.b = b;
    cfg.sigma_iters = parse_optional_usize(sigma_iters, cfg.sigma_iters, "sigmaIters")?;
    cfg.sigma_tolerance = sigma_tolerance.unwrap_or(cfg.sigma_tolerance);
    cfg.seed = parse_seed_u64(seed)?;
    Ok(cfg)
}

/// Cooperative Graph UMAP (`GuMap`) layout solver (`f32`).
#[wasm_bindgen(js_name = GuMapSolver)]
#[derive(Debug, Clone)]
pub struct JsGuMapSolver {
    inner: GuMapSolver<f32>,
}

#[wasm_bindgen(js_class = GuMapSolver)]
impl JsGuMapSolver {
    /// Constructs a `GuMapSolver` directly from a fuzzy simplicial set `SparseProbabilities`.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = fromFuzzySet)]
    pub fn from_fuzzy_set(
        probs: &JsSparseProbabilities,
        n_epochs_or_options: &JsValue,
        negative_sample_rate: Option<f64>,
        learning_rate: Option<f64>,
        min_dist: Option<f64>,
        spread: Option<f64>,
        a: Option<f64>,
        b: Option<f64>,
        seed: Option<f64>,
        sigma_iters: Option<f64>,
        sigma_tolerance: Option<f64>,
    ) -> Result<JsGuMapSolver, JsError> {
        let cfg = parse_gumap_config(
            n_epochs_or_options,
            negative_sample_rate,
            learning_rate,
            min_dist,
            spread,
            a,
            b,
            sigma_iters,
            sigma_tolerance,
            seed,
            false,
        )?;
        let inner =
            GuMapSolver::<f32>::from_fuzzy_set_ref(probs.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `GuMapSolver` from a `KnnGraph` via `computeFuzzySimplicialSet`.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = fromKnn)]
    pub fn from_knn(
        knn: &JsKnnGraph,
        n_epochs_or_options: &JsValue,
        negative_sample_rate: Option<f64>,
        learning_rate: Option<f64>,
        min_dist: Option<f64>,
        spread: Option<f64>,
        a: Option<f64>,
        b: Option<f64>,
        sigma_iters: Option<f64>,
        sigma_tolerance: Option<f64>,
        seed: Option<f64>,
    ) -> Result<JsGuMapSolver, JsError> {
        let cfg = parse_gumap_config(
            n_epochs_or_options,
            negative_sample_rate,
            learning_rate,
            min_dist,
            spread,
            a,
            b,
            sigma_iters,
            sigma_tolerance,
            seed,
            false,
        )?;
        let inner = GuMapSolver::<f32>::from_knn(knn.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `GuMapSolver` from a `CsrGraph` using 1+2-hop pairs and DoG Varadhan landmarks.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = fromGraphDog)]
    pub fn from_graph_dog(
        graph: &JsCsrGraph,
        max_2hop_or_options: &JsValue,
        n_epochs_or_options: &JsValue,
        negative_sample_rate: Option<f64>,
        learning_rate: Option<f64>,
        min_dist: Option<f64>,
        spread: Option<f64>,
        a: Option<f64>,
        b: Option<f64>,
        sigma_iters: Option<f64>,
        sigma_tolerance: Option<f64>,
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
    ) -> Result<JsGuMapSolver, JsError> {
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

        let (max_2hop, gumap_cfg, dog_cfg) = if is_plain_js_object(max_2hop_or_options) {
            let has_extra = has_dog_extra
                || (!n_epochs_or_options.is_undefined() && !n_epochs_or_options.is_null());
            let max_2hop_val = opt_usize_prop(
                max_2hop_or_options,
                "max2hopPerNode",
                DEFAULT_GUMAP_MAX_2HOP,
            )?;
            let g_cfg = parse_gumap_config(
                max_2hop_or_options,
                negative_sample_rate,
                learning_rate,
                min_dist,
                spread,
                a,
                b,
                sigma_iters,
                sigma_tolerance,
                seed,
                has_extra,
            )?;
            let d_cfg = parse_dog_config_from_options_or_args(
                max_2hop_or_options,
                g_cfg.seed,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )?;
            (max_2hop_val, g_cfg, d_cfg)
        } else {
            let max_2hop_val = if max_2hop_or_options.is_undefined()
                || max_2hop_or_options.is_null()
            {
                DEFAULT_GUMAP_MAX_2HOP
            } else {
                let v = max_2hop_or_options.as_f64().ok_or_else(|| {
                    JsError::new("max2hopOrOptions must be a non-negative integer, options object, null, or undefined")
                })?;
                parse_non_negative_usize(v, "max2hopPerNode")?
            };
            let g_cfg = parse_gumap_config(
                n_epochs_or_options,
                negative_sample_rate,
                learning_rate,
                min_dist,
                spread,
                a,
                b,
                sigma_iters,
                sigma_tolerance,
                seed,
                has_dog_extra && is_plain_js_object(n_epochs_or_options),
            )?;
            let d_cfg = parse_dog_config_from_options_or_args(
                n_epochs_or_options,
                g_cfg.seed,
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
            (max_2hop_val, g_cfg, d_cfg)
        };

        let inner =
            GuMapSolver::<f32>::from_graph_dog(graph.inner(), max_2hop, &dog_cfg, gumap_cfg)
                .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `GuMapSolver` from precomputed `DogVaradhanLandmarks` and 1+2-hop priority CSR.
    #[wasm_bindgen(js_name = fromLandmarks)]
    pub fn from_landmarks(
        graph: &JsCsrGraph,
        landmarks: &JsDogVaradhanLandmarks,
        max_2hop_or_options: &JsValue,
        options: &JsValue,
    ) -> Result<JsGuMapSolver, JsError> {
        let (max_2hop, gumap_cfg) = if is_plain_js_object(max_2hop_or_options) {
            reject_extra_args_with_options(!options.is_undefined() && !options.is_null())?;
            let max_2hop_val = opt_usize_prop(
                max_2hop_or_options,
                "max2hopPerNode",
                DEFAULT_GUMAP_MAX_2HOP,
            )?;
            let g_cfg = parse_gumap_config(
                max_2hop_or_options,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                false,
            )?;
            (max_2hop_val, g_cfg)
        } else {
            let max_2hop_val = if max_2hop_or_options.is_undefined()
                || max_2hop_or_options.is_null()
            {
                DEFAULT_GUMAP_MAX_2HOP
            } else {
                let v = max_2hop_or_options.as_f64().ok_or_else(|| {
                    JsError::new("max2hopOrOptions must be a non-negative integer, options object, null, or undefined")
                })?;
                parse_non_negative_usize(v, "max2hopPerNode")?
            };
            if !options.is_undefined() && !options.is_null() && !is_plain_js_object(options) {
                return Err(JsError::new(
                    "GuMapSolver.fromLandmarks: options must be a plain object, null, or undefined",
                ));
            }
            let g_cfg = parse_gumap_config(
                options, None, None, None, None, None, None, None, None, None, false,
            )?;
            (max_2hop_val, g_cfg)
        };
        let inner = GuMapSolver::<f32>::from_landmarks(
            graph.inner(),
            landmarks.inner(),
            max_2hop,
            gumap_cfg,
        )
        .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one epoch on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` epochs on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "GuMapSolver::run_steps",
        )
    }

    /// Runs GuMap optimization to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "GuMapSolver::run",
        )
    }

    /// Returns the number of directed positive attractive edges stored.
    #[wasm_bindgen(getter, js_name = numEdges)]
    pub fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    /// Returns the resolved UMAP curve parameter `a`.
    #[wasm_bindgen(getter)]
    pub fn a(&self) -> f64 {
        self.inner.ab().0
    }

    /// Returns the resolved UMAP curve parameter `b`.
    #[wasm_bindgen(getter)]
    pub fn b(&self) -> f64 {
        self.inner.ab().1
    }

    /// Returns the resolved UMAP curve parameters `[a, b]`.
    #[wasm_bindgen(getter)]
    pub fn ab(&self) -> Vec<f64> {
        let (a, b) = self.inner.ab();
        vec![a, b]
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    /// Returns the number of epochs executed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the total scheduled epoch count (`nEpochs`).
    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> usize {
        self.inner.total_iterations()
    }

    /// Returns `true` when the solver has converged or completed.
    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        self.inner.status().is_finished()
    }

    /// Returns the expected vertex count.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }
}
