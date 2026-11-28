//! Cooperative interpolation-based batch Graph UMAP (`ibUMAP`) WebAssembly bindings (`f32`).
//!
//! `IbUmapSolver` targets `"layout2d"` exclusively (`"layout3d"`, `"layoutNd"`, and
//! `"manifold2d"` fail fast with `UnsupportedOperation`).

use egraph::layout::{FieldRepulsionBackend, IbUmapConfig, IbUmapSolver};
use egraph::neighborhood::DogVaradhanConfig;
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::dog_varadhan::JsDogVaradhanLandmarks;
use crate::js_convert::{
    DEFAULT_SEED, core_err_to_js, invalid_input_err, parse_non_negative_usize, parse_u32_sequence,
    unsupported_err,
};
use crate::js_options::{
    JsTargetKind, check_options_arg, get_prop, is_plain_js_object, opt_bool_prop, opt_f64_prop,
    opt_maybe_f64_prop, opt_seed_prop, opt_string_prop, opt_target_prop, opt_usize_prop,
    parse_dog_config_from_options, parse_field_repulsion_backend, parse_ibfft_order,
    solver_status_str,
};
use crate::layout_buffers::JsLayout2d;
use crate::neighborhood_types::{JsKnnGraph, JsSparseProbabilities};
use crate::solver_runner::{run_layout_in_place, run_steps_layout_in_place, step_layout_in_place};

const DEFAULT_IBUMAP_MAX_2HOP: usize = 30;
const DEFAULT_IBUMAP_LANDMARKS_MAX_2HOP: usize = 32;
const DEFAULT_IBUMAP_SIGMA_ITERS: usize = 64;
const DEFAULT_IBUMAP_SIGMA_TOL: f64 = 1e-5;

fn parse_ib_umap_config(options: &JsValue, ctx: &str) -> Result<IbUmapConfig, JsValue> {
    let mut cfg = IbUmapConfig::default();
    if check_options_arg(options, ctx)? {
        let target = opt_target_prop(options)?;
        if target != JsTargetKind::Layout2d {
            return Err(unsupported_err(
                "IbUmapSolver only supports target 'layout2d'",
            ));
        }
        if let Some(b_str) = opt_string_prop(options, "backend")? {
            cfg.backend = parse_field_repulsion_backend(&b_str)?;
        }
        cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
        cfg.learning_rate = opt_f64_prop(options, "learningRate", cfg.learning_rate)?;
        cfg.min_dist = opt_f64_prop(options, "minDist", cfg.min_dist)?;
        cfg.spread = opt_f64_prop(options, "spread", cfg.spread)?;
        cfg.a = opt_maybe_f64_prop(options, "a")?;
        cfg.b = opt_maybe_f64_prop(options, "b")?;
        cfg.c_k = opt_f64_prop(options, "cK", cfg.c_k)?;
        cfg.tau_scale = opt_f64_prop(options, "tauScale", cfg.tau_scale)?;
        cfg.hub_damping = opt_bool_prop(options, "hubDamping", cfg.hub_damping)?;
        cfg.theta = opt_f64_prop(options, "theta", cfg.theta)?;
        let ord_val = get_prop(options, "interpolationOrder")?;
        if !ord_val.is_undefined() && !ord_val.is_null() {
            cfg.ibfft.order = parse_ibfft_order(&ord_val)?;
        }
        cfg.ibfft.intervals = opt_usize_prop(options, "intervals", cfg.ibfft.intervals)?;
        cfg.ibfft.min_grid_dim = opt_usize_prop(options, "minGridDim", cfg.ibfft.min_grid_dim)?;
        cfg.ibfft.max_grid_dim = opt_usize_prop(options, "maxGridDim", cfg.ibfft.max_grid_dim)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    Ok(cfg)
}

/// Cooperative interpolation-based batch Graph UMAP (`ibUMAP`) solver (`f32`, `layout2d`).
#[wasm_bindgen(js_name = IbUmapSolver)]
#[derive(Debug, Clone)]
pub struct JsIbUmapSolver {
    inner: IbUmapSolver<f32>,
}

#[wasm_bindgen(js_class = IbUmapSolver)]
impl JsIbUmapSolver {
    /// Constructs an `IbUmapSolver` from a `CsrGraph` using $O(|E|)$ 1+2-hop pairs and
    /// regularized DoG Varadhan landmark ALT distances.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsIbUmapSolver, JsValue> {
        let cfg = parse_ib_umap_config(options, "IbUmapSolver")?;
        let (max_2hop, mut dog_cfg) = if is_plain_js_object(options) {
            let max_2hop_val = opt_usize_prop(options, "max2hopPerNode", DEFAULT_IBUMAP_MAX_2HOP)?;
            let d_cfg = parse_dog_config_from_options(options, "IbUmapSolver")?;
            (max_2hop_val, d_cfg)
        } else {
            (
                DEFAULT_IBUMAP_MAX_2HOP,
                DogVaradhanConfig::default().with_seed(DEFAULT_SEED),
            )
        };
        dog_cfg.seed = cfg.seed;
        let inner = IbUmapSolver::<f32>::from_graph_dog(graph.inner(), max_2hop, &dog_cfg, cfg)
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs an `IbUmapSolver` from precomputed `DogVaradhanLandmarks` and 1+2-hop pairs.
    #[wasm_bindgen(js_name = fromLandmarks)]
    pub fn from_landmarks(
        graph: &JsCsrGraph,
        landmarks: &JsDogVaradhanLandmarks,
        max_2hop_or_options: &JsValue,
        options: &JsValue,
    ) -> Result<JsIbUmapSolver, JsValue> {
        let (max_2hop, opts_val) = if is_plain_js_object(max_2hop_or_options) {
            if !options.is_undefined() && !options.is_null() {
                return Err(invalid_input_err(
                    "IbUmapSolver.fromLandmarks: unexpected trailing argument after options object",
                ));
            }
            let max_2hop_val = opt_usize_prop(
                max_2hop_or_options,
                "max2hopPerNode",
                DEFAULT_IBUMAP_LANDMARKS_MAX_2HOP,
            )?;
            (max_2hop_val, max_2hop_or_options)
        } else {
            let max_2hop_val = if max_2hop_or_options.is_undefined()
                || max_2hop_or_options.is_null()
            {
                if is_plain_js_object(options) {
                    opt_usize_prop(options, "max2hopPerNode", DEFAULT_IBUMAP_LANDMARKS_MAX_2HOP)?
                } else {
                    DEFAULT_IBUMAP_LANDMARKS_MAX_2HOP
                }
            } else {
                let v = max_2hop_or_options.as_f64().ok_or_else(|| {
                    invalid_input_err(
                        "max2hopOrOptions must be a non-negative integer, options object, null, or undefined",
                    )
                })?;
                parse_non_negative_usize(v, "max2hopPerNode")?
            };
            (max_2hop_val, options)
        };
        let cfg = parse_ib_umap_config(opts_val, "IbUmapSolver.fromLandmarks")?;
        let inner =
            IbUmapSolver::<f32>::from_landmarks(graph.inner(), landmarks.inner(), max_2hop, cfg)
                .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs an `IbUmapSolver` from a `KnnGraph` via `computeFuzzySimplicialSet`.
    #[wasm_bindgen(js_name = fromKnn)]
    pub fn from_knn(knn: &JsKnnGraph, options: &JsValue) -> Result<JsIbUmapSolver, JsValue> {
        let cfg = parse_ib_umap_config(options, "IbUmapSolver.fromKnn")?;
        let (sigma_iters, sigma_tol) = if is_plain_js_object(options) {
            (
                opt_usize_prop(options, "sigmaIters", DEFAULT_IBUMAP_SIGMA_ITERS)?,
                opt_f64_prop(options, "sigmaTolerance", DEFAULT_IBUMAP_SIGMA_TOL)?,
            )
        } else {
            (DEFAULT_IBUMAP_SIGMA_ITERS, DEFAULT_IBUMAP_SIGMA_TOL)
        };
        let inner = IbUmapSolver::<f32>::from_knn(knn.inner(), sigma_iters, sigma_tol, cfg)
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs an `IbUmapSolver` directly from a fuzzy simplicial set `SparseProbabilities`
    /// and optional per-node `degrees` for inverse-degree hub damping.
    #[wasm_bindgen(js_name = fromFuzzySet)]
    pub fn from_fuzzy_set(
        probs: &JsSparseProbabilities,
        degrees_or_options: &JsValue,
        options: &JsValue,
    ) -> Result<JsIbUmapSolver, JsValue> {
        let (pos_degrees, opts_val) = if is_plain_js_object(degrees_or_options) {
            if !options.is_undefined() && !options.is_null() {
                return Err(invalid_input_err(
                    "IbUmapSolver.fromFuzzySet: unexpected trailing argument after options object",
                ));
            }
            (None, degrees_or_options)
        } else if degrees_or_options.is_undefined() || degrees_or_options.is_null() {
            (None, options)
        } else {
            let degs = parse_u32_sequence(degrees_or_options, "degrees")?;
            (Some(degs), options)
        };

        let cfg = parse_ib_umap_config(opts_val, "IbUmapSolver.fromFuzzySet")?;
        let degrees = if let Some(degs) = pos_degrees {
            Some(degs)
        } else if is_plain_js_object(opts_val) {
            let deg_val = get_prop(opts_val, "degrees")?;
            if deg_val.is_undefined() || deg_val.is_null() {
                None
            } else {
                Some(parse_u32_sequence(&deg_val, "degrees")?)
            }
        } else {
            None
        };

        let inner =
            IbUmapSolver::<f32>::from_fuzzy_set(probs.inner().clone(), degrees.as_deref(), cfg)
                .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        step_layout_in_place(&mut self.inner, layout)
    }

    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsValue> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "IbUmapSolver::run_steps",
        )
    }

    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "IbUmapSolver::run",
        )
    }

    #[wasm_bindgen(getter)]
    pub fn target(&self) -> String {
        "layout2d".to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn backend(&self) -> String {
        match self.inner.config().backend {
            FieldRepulsionBackend::IbFft => "ibFft",
            FieldRepulsionBackend::BarnesHut => "barnesHut",
            FieldRepulsionBackend::Exact => "exact",
            _ => "ibFft",
        }
        .to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> Option<usize> {
        self.inner.total_iterations()
    }

    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        self.inner.status().is_finished()
    }

    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    #[wasm_bindgen(getter, js_name = numEdges)]
    pub fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    #[wasm_bindgen(getter, js_name = totalFft2dCalls)]
    pub fn total_fft_2d_calls(&self) -> f64 {
        self.inner.total_fft_2d_calls() as f64
    }

    #[wasm_bindgen(getter)]
    pub fn a(&self) -> f64 {
        self.inner.ab().0
    }

    #[wasm_bindgen(getter)]
    pub fn b(&self) -> f64 {
        self.inner.ab().1
    }

    #[wasm_bindgen(getter)]
    pub fn ab(&self) -> Vec<f64> {
        let (a, b) = self.inner.ab();
        vec![a, b]
    }
}
