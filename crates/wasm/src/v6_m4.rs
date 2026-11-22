//! M4 v6 WebAssembly bindings for unified force models and layout post-processing.
//!
//! Exposes cooperative `f32` `TFdpSolver`, `IbUmapSolver`, `SgdSolver` M4 constructors
//! (`snapTFdp`, `pacmapGraph`, `sgd2`), `ForceAtlas2Solver`, `YifanHuSolver`,
//! `PrismOverlapRemoval`, `SeparationConstraint` projection utilities, `CsrGraph`
//! cycle breaking / layering, and reference-passing `FdebSolver` without `serde`,
//! `getrandom`, or threading dependencies.

use egraph::core::{CoreError, IterativeSolver};
use egraph::layout::{
    ConstraintProjector, FdebConfig, FdebSolver, FieldLayoutSolver, FieldRepulsionBackend,
    ForceAtlas2Config, ForceAtlas2Solver, IbUmapConfig, LayeringMethod, PacmapGraphConfig,
    PrismConfig, PrismOverlapRemoval, SeparationConstraint, Sgd2Config, SgdSolver, SnapTFdpConfig,
    TFdpConfig, YifanHuConfig, YifanHuSolver, assign_layers, cycle_edges,
    generate_layered_constraints, project_clustered_rectangle_no_overlap_2d,
    project_rectangle_no_overlap_2d, remove_cycles,
};
use egraph::neighborhood::{DogVaradhanConfig, KnnGraph};
use wasm_bindgen::{JsCast, prelude::*};

use crate::v6::{JsCsrGraph, JsLayout2d, core_err_to_js, ffi_err_to_js, parse_non_negative_usize};
use crate::v6_m2::{JsKnnGraph, JsSparseProbabilities};
use crate::v6_m3::{
    JsSgdSolver, get_prop, is_plain_js_object, normalize_enum_token, opt_bool_prop, opt_f64_prop,
    opt_maybe_f64_prop, opt_seed_prop, opt_string_prop, opt_usize_prop,
    parse_dog_config_from_options_or_args, parse_ibfft_order, reject_extra_args_with_options,
    run_layout_in_place, run_steps_layout_in_place, solver_status_str, step_layout_in_place,
};

const DEFAULT_IBUMAP_MAX_2HOP: usize = 30;
const DEFAULT_IBUMAP_SIGMA_ITERS: usize = 64;
const DEFAULT_IBUMAP_SIGMA_TOL: f64 = 1e-5;
const DEFAULT_MIN_LAYER_DISTANCE: f32 = 1.0;

fn check_options_arg(options: &JsValue, ctx: &str) -> Result<bool, JsError> {
    if options.is_undefined() || options.is_null() {
        return Ok(false);
    }
    if is_plain_js_object(options) {
        return Ok(true);
    }
    Err(JsError::new(&format!(
        "{ctx}: options must be a plain object, null, or undefined"
    )))
}

fn parse_non_negative_u32(value: f64, name: &str) -> Result<u32, JsError> {
    let v = parse_non_negative_usize(value, name)?;
    u32::try_from(v).map_err(|_| JsError::new(&format!("{name} must be <= u32::MAX, got {value}")))
}

fn parse_f64_sequence(val: &JsValue, name: &str) -> Result<Vec<f64>, JsError> {
    if let Some(arr) = val.dyn_ref::<js_sys::Float64Array>() {
        return Ok(arr.to_vec());
    }
    if let Some(arr) = val.dyn_ref::<js_sys::Float32Array>() {
        return Ok(arr.to_vec().into_iter().map(f64::from).collect());
    }
    if val.is_null() || val.is_undefined() || val.is_string() || !val.is_object() {
        return Err(JsError::new(&format!(
            "{name} must be a Float64Array, Float32Array, or numeric iterable"
        )));
    }
    let iter = js_sys::try_iter(val)
        .map_err(|err| JsError::new(&format!("failed to iterate {name}: {err:?}")))?
        .ok_or_else(|| {
            JsError::new(&format!(
                "{name} must be a Float64Array, Float32Array, or numeric iterable"
            ))
        })?;
    let mut out = Vec::new();
    for item_res in iter {
        let item =
            item_res.map_err(|err| JsError::new(&format!("failed to read {name}: {err:?}")))?;
        let num = item
            .as_f64()
            .ok_or_else(|| JsError::new(&format!("{name} entries must be numbers")))?;
        out.push(num);
    }
    Ok(out)
}

fn parse_u32_sequence(val: &JsValue, name: &str) -> Result<Vec<u32>, JsError> {
    if let Some(arr) = val.dyn_ref::<js_sys::Uint32Array>() {
        return Ok(arr.to_vec());
    }
    if val.is_null() || val.is_undefined() || val.is_string() || !val.is_object() {
        return Err(JsError::new(&format!(
            "{name} must be a Uint32Array or integer array"
        )));
    }
    let iter = js_sys::try_iter(val)
        .map_err(|err| JsError::new(&format!("failed to iterate {name}: {err:?}")))?
        .ok_or_else(|| JsError::new(&format!("{name} must be a Uint32Array or integer array")))?;
    let mut out = Vec::new();
    for item_res in iter {
        let item =
            item_res.map_err(|err| JsError::new(&format!("failed to read {name}: {err:?}")))?;
        let num = item
            .as_f64()
            .ok_or_else(|| JsError::new(&format!("{name} entries must be integers")))?;
        out.push(parse_non_negative_u32(num, name)?);
    }
    Ok(out)
}

fn parse_field_repulsion_backend(raw: &str) -> Result<FieldRepulsionBackend, JsError> {
    match normalize_enum_token(raw).as_str() {
        "ibfft" | "fft" => Ok(FieldRepulsionBackend::IbFft),
        "barneshut" | "bh" => Ok(FieldRepulsionBackend::BarnesHut),
        "exact" => Ok(FieldRepulsionBackend::Exact),
        _ => Err(JsError::new(&format!(
            "unknown FieldRepulsionBackend {raw:?}; expected 'ibfft', 'barnes_hut', or 'exact'"
        ))),
    }
}

fn parse_t_fdp_config(options: &JsValue) -> Result<TFdpConfig, JsError> {
    let mut cfg = TFdpConfig::default();
    if check_options_arg(options, "TFdpSolver")? {
        if let Some(b_str) = opt_string_prop(options, "backend")? {
            cfg.backend = parse_field_repulsion_backend(&b_str)?;
        }
        cfg.total_iterations = opt_usize_prop(options, "totalIterations", cfg.total_iterations)?;
        cfg.learning_rate = opt_f64_prop(options, "learningRate", cfg.learning_rate)?;
        cfg.momentum = opt_f64_prop(options, "momentum", cfg.momentum)?;
        cfg.theta = opt_f64_prop(options, "theta", cfg.theta)?;
        cfg.alpha = opt_f64_prop(options, "alpha", cfg.alpha)?;
        cfg.gamma = opt_f64_prop(options, "gamma", cfg.gamma)?;
        cfg.rho.start_weight = opt_f64_prop(options, "rhoStart", cfg.rho.start_weight)?;
        cfg.rho.end_weight = opt_f64_prop(options, "rhoEnd", cfg.rho.end_weight)?;
        cfg.rho.power = opt_f64_prop(options, "rhoPower", cfg.rho.power)?;
        let ord_val = get_prop(options, "ibfftOrder")?;
        if !ord_val.is_undefined() && !ord_val.is_null() {
            cfg.ibfft.order = parse_ibfft_order(&ord_val)?;
        }
        cfg.ibfft.intervals = opt_usize_prop(options, "ibfftIntervals", cfg.ibfft.intervals)?;
        cfg.ibfft.min_grid_dim =
            opt_usize_prop(options, "ibfftMinGridDim", cfg.ibfft.min_grid_dim)?;
        cfg.ibfft.max_grid_dim =
            opt_usize_prop(options, "ibfftMaxGridDim", cfg.ibfft.max_grid_dim)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    Ok(cfg)
}

fn parse_ib_umap_config(options: &JsValue) -> Result<IbUmapConfig, JsError> {
    let mut cfg = IbUmapConfig::default();
    if check_options_arg(options, "IbUmapSolver")? {
        if let Some(b_str) = opt_string_prop(options, "backend")? {
            cfg.backend = parse_field_repulsion_backend(&b_str)?;
        }
        cfg.n_epochs = opt_usize_prop(options, "nEpochs", cfg.n_epochs)?;
        cfg.learning_rate = opt_f64_prop(options, "learningRate", cfg.learning_rate)?;
        cfg.min_dist = opt_f64_prop(options, "minDist", cfg.min_dist)?;
        cfg.spread = opt_f64_prop(options, "spread", cfg.spread)?;
        cfg.a = opt_maybe_f64_prop(options, "a")?;
        cfg.b = opt_maybe_f64_prop(options, "b")?;
        cfg.c_k = opt_f64_prop(options, "cK", cfg.c_k)?;
        cfg.tau_scale = opt_f64_prop(options, "tauScale", cfg.tau_scale)?;
        cfg.hub_damping = opt_bool_prop(options, "hubDamping", cfg.hub_damping)?;
        cfg.theta = opt_f64_prop(options, "theta", cfg.theta)?;
        let ord_val = get_prop(options, "ibfftOrder")?;
        if !ord_val.is_undefined() && !ord_val.is_null() {
            cfg.ibfft.order = parse_ibfft_order(&ord_val)?;
        }
        cfg.ibfft.intervals = opt_usize_prop(options, "ibfftIntervals", cfg.ibfft.intervals)?;
        cfg.ibfft.min_grid_dim =
            opt_usize_prop(options, "ibfftMinGridDim", cfg.ibfft.min_grid_dim)?;
        cfg.ibfft.max_grid_dim =
            opt_usize_prop(options, "ibfftMaxGridDim", cfg.ibfft.max_grid_dim)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    Ok(cfg)
}

// ============================================================================
// 1. TFdpSolver & IbUmapSolver (FieldLayoutSolver<f32>)
// ============================================================================

/// Cooperative bounded Student-$t$ force-directed placement (`t-FDP`) solver (`f32`)
/// supporting `ibfft`, `barnes_hut`, and `exact` repulsion backends.
#[wasm_bindgen(js_name = TFdpSolver)]
#[derive(Debug, Clone)]
pub struct JsTFdpSolver {
    inner: FieldLayoutSolver<f32>,
}

fn try_extract_knn_graph(val: &JsValue) -> Result<Option<KnnGraph>, JsError> {
    if val.is_null()
        || val.is_undefined()
        || !val.is_object()
        || js_sys::Array::is_array(val)
        || js_sys::ArrayBuffer::is_view(val)
        || val.is_function()
    {
        return Ok(None);
    }
    let num_nodes_val = get_prop(val, "numNodes")?;
    let indptr_prop = get_prop(val, "indptr")?;
    let indices_prop = get_prop(val, "indices")?;
    let distances_prop = get_prop(val, "distances")?;
    let (Some(num_nodes_f), Some(indptr_fn), Some(indices_fn), Some(distances_fn)) = (
        num_nodes_val.as_f64(),
        indptr_prop.dyn_ref::<js_sys::Function>(),
        indices_prop.dyn_ref::<js_sys::Function>(),
        distances_prop.dyn_ref::<js_sys::Function>(),
    ) else {
        return Ok(None);
    };
    let n = parse_non_negative_usize(num_nodes_f, "knn.numNodes")?;
    let raw_indptr = indptr_fn.call0(val).map_err(|err| {
        JsError::new(&format!("TFdpSolver.fromKnn: knn.indptr() failed: {err:?}"))
    })?;
    let raw_indices = indices_fn.call0(val).map_err(|err| {
        JsError::new(&format!(
            "TFdpSolver.fromKnn: knn.indices() failed: {err:?}"
        ))
    })?;
    let raw_distances = distances_fn.call0(val).map_err(|err| {
        JsError::new(&format!(
            "TFdpSolver.fromKnn: knn.distances() failed: {err:?}"
        ))
    })?;
    let indptr = parse_u32_sequence(&raw_indptr, "knn.indptr")?;
    let indices = parse_u32_sequence(&raw_indices, "knn.indices")?;
    let distances = parse_f64_sequence(&raw_distances, "knn.distances")?;
    let indptr_usize: Vec<usize> = indptr.iter().map(|&v| v as usize).collect();
    let knn =
        KnnGraph::new_validated(n, indptr_usize, indices, distances).map_err(core_err_to_js)?;
    Ok(Some(knn))
}

#[wasm_bindgen(js_class = TFdpSolver)]
impl JsTFdpSolver {
    /// Constructs a `TFdpSolver` directly from a `CsrGraph`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsTFdpSolver, JsError> {
        let cfg = parse_t_fdp_config(options)?;
        let inner = FieldLayoutSolver::<f32>::t_fdp(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `TFdpSolver` from either a `KnnGraph` instance (`fromKnn(knn, options?)`)
    /// or raw CSR $k$-NN `indptr`, `indices`, and optional neighbor `weights`
    /// (`fromKnn(indptr, indices, weights?, options?)`, where `weights` defaults to uniform `0.0`).
    #[wasm_bindgen(js_name = fromKnn)]
    pub fn from_knn(
        knn_or_indptr: &JsValue,
        indices_or_options: &JsValue,
        weights_or_options: &JsValue,
        options: &JsValue,
    ) -> Result<JsTFdpSolver, JsError> {
        let (knn, opts_val) = if let Some(knn) = try_extract_knn_graph(knn_or_indptr)? {
            reject_extra_args_with_options(
                (!weights_or_options.is_undefined() && !weights_or_options.is_null())
                    || (!options.is_undefined() && !options.is_null()),
            )?;
            (knn, indices_or_options)
        } else {
            let indptr = parse_u32_sequence(knn_or_indptr, "indptr")?;
            if indptr.is_empty() {
                return Err(core_err_to_js(CoreError::invalid_csr(
                    "TFdpSolver.fromKnn: indptr must have length numNodes + 1 >= 1",
                )));
            }
            let indices = parse_u32_sequence(indices_or_options, "indices")?;
            let n = indptr.len() - 1;
            let (weights_vec, opts_val) = if is_plain_js_object(weights_or_options) {
                reject_extra_args_with_options(!options.is_undefined() && !options.is_null())?;
                (vec![0.0_f64; indices.len()], weights_or_options)
            } else if weights_or_options.is_undefined() || weights_or_options.is_null() {
                (vec![0.0_f64; indices.len()], options)
            } else {
                let ws = parse_f64_sequence(weights_or_options, "weights")?;
                (ws, options)
            };
            let indptr_usize: Vec<usize> = indptr.iter().map(|&v| v as usize).collect();
            let knn = KnnGraph::new_validated(n, indptr_usize, indices, weights_vec)
                .map_err(core_err_to_js)?;
            (knn, opts_val)
        };

        let cfg = parse_t_fdp_config(opts_val)?;
        let inner = FieldLayoutSolver::<f32>::t_fdp_from_knn(&knn, cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one `t-FDP` iteration on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` `t-FDP` iterations on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "TFdpSolver::run_steps",
        )
    }

    /// Runs `t-FDP` optimization to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "TFdpSolver::run",
        )
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    /// Returns the number of iterations executed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the total scheduled iteration count.
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

    /// Returns the current $\rho(t)$ repulsion schedule weight.
    #[wasm_bindgen(getter, js_name = currentRho)]
    pub fn current_rho(&self) -> f64 {
        self.inner.current_rho()
    }

    /// Returns the number of directed positive attraction edges stored.
    #[wasm_bindgen(getter, js_name = numEdges)]
    pub fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    /// Returns the cumulative number of `fft_2d` calls executed across all steps.
    #[wasm_bindgen(getter, js_name = totalFft2dCalls)]
    pub fn total_fft_2d_calls(&self) -> f64 {
        self.inner.total_fft_2d_calls() as f64
    }
}

/// Cooperative interpolation-based batch Graph UMAP (`ibUMAP`) solver (`f32`).
#[wasm_bindgen(js_name = IbUmapSolver)]
#[derive(Debug, Clone)]
pub struct JsIbUmapSolver {
    inner: FieldLayoutSolver<f32>,
}

#[wasm_bindgen(js_class = IbUmapSolver)]
impl JsIbUmapSolver {
    /// Constructs an `IbUmapSolver` from a `CsrGraph` using $O(|E|)$ 1+2-hop pairs and
    /// regularized DoG Varadhan landmark ALT distances.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsIbUmapSolver, JsError> {
        let cfg = parse_ib_umap_config(options)?;
        let (max_2hop, dog_cfg) = if is_plain_js_object(options) {
            let m2 = opt_usize_prop(options, "max2hopPerNode", DEFAULT_IBUMAP_MAX_2HOP)?;
            let d_cfg = parse_dog_config_from_options_or_args(
                options, cfg.seed, None, None, None, None, None, None, None, None, None, None,
                None, None,
            )?;
            (m2, d_cfg)
        } else {
            (
                DEFAULT_IBUMAP_MAX_2HOP,
                DogVaradhanConfig {
                    seed: cfg.seed,
                    ..DogVaradhanConfig::default()
                },
            )
        };
        let inner = FieldLayoutSolver::<f32>::ib_umap(graph.inner(), max_2hop, &dog_cfg, cfg)
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs an `IbUmapSolver` from a `KnnGraph` via `computeFuzzySimplicialSet`.
    #[wasm_bindgen(js_name = fromKnn)]
    pub fn from_knn(knn: &JsKnnGraph, options: &JsValue) -> Result<JsIbUmapSolver, JsError> {
        let cfg = parse_ib_umap_config(options)?;
        let (sigma_iters, sigma_tol) = if is_plain_js_object(options) {
            (
                opt_usize_prop(options, "sigmaIters", DEFAULT_IBUMAP_SIGMA_ITERS)?,
                opt_f64_prop(options, "sigmaTolerance", DEFAULT_IBUMAP_SIGMA_TOL)?,
            )
        } else {
            (DEFAULT_IBUMAP_SIGMA_ITERS, DEFAULT_IBUMAP_SIGMA_TOL)
        };
        let inner =
            FieldLayoutSolver::<f32>::ib_umap_from_knn(knn.inner(), sigma_iters, sigma_tol, cfg)
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
    ) -> Result<JsIbUmapSolver, JsError> {
        let (pos_degrees, opts_val) = if is_plain_js_object(degrees_or_options) {
            reject_extra_args_with_options(!options.is_undefined() && !options.is_null())?;
            (None, degrees_or_options)
        } else if degrees_or_options.is_undefined() || degrees_or_options.is_null() {
            (None, options)
        } else {
            let degs = parse_u32_sequence(degrees_or_options, "degrees")?;
            (Some(degs), options)
        };

        let cfg = parse_ib_umap_config(opts_val)?;
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

        let inner = FieldLayoutSolver::<f32>::ib_umap_from_fuzzy_set(
            probs.inner().clone(),
            degrees.as_deref(),
            cfg,
        )
        .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one `ibUMAP` epoch on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` `ibUMAP` epochs on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "IbUmapSolver::run_steps",
        )
    }

    /// Runs `ibUMAP` optimization to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "IbUmapSolver::run",
        )
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

    /// Returns `1.0` (constant unit schedule compatibility getter).
    #[wasm_bindgen(getter, js_name = currentRho)]
    pub fn current_rho(&self) -> f64 {
        self.inner.current_rho()
    }

    /// Returns the number of directed positive attraction edges stored.
    #[wasm_bindgen(getter, js_name = numEdges)]
    pub fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    /// Returns the cumulative number of `fft_2d` calls executed across all steps.
    #[wasm_bindgen(getter, js_name = totalFft2dCalls)]
    pub fn total_fft_2d_calls(&self) -> f64 {
        self.inner.total_fft_2d_calls() as f64
    }

    /// Returns the resolved UMAP curve parameter `a`.
    #[wasm_bindgen(getter)]
    pub fn a(&self) -> f64 {
        self.inner.ab().map_or(0.0, |(a, _)| a)
    }

    /// Returns the resolved UMAP curve parameter `b`.
    #[wasm_bindgen(getter)]
    pub fn b(&self) -> f64 {
        self.inner.ab().map_or(0.0, |(_, b)| b)
    }

    /// Returns the resolved UMAP curve parameters `[a, b]`.
    #[wasm_bindgen(getter)]
    pub fn ab(&self) -> Vec<f64> {
        let (a, b) = self.inner.ab().unwrap_or((0.0, 0.0));
        vec![a, b]
    }
}

// ============================================================================
// 2. SgdSolver M4 Constructors (snapTFdp, pacmapGraph, sgd2)
// ============================================================================

pub(crate) fn sgd_snap_t_fdp(
    graph: &JsCsrGraph,
    options: &JsValue,
) -> Result<JsSgdSolver, JsError> {
    let mut cfg = SnapTFdpConfig::default();
    if check_options_arg(options, "SgdSolver.snapTFdp")? {
        cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
        cfg.negatives_per_edge =
            opt_usize_prop(options, "negativesPerEdge", cfg.negatives_per_edge)?;
        cfg.learning_rate = opt_f64_prop(options, "learningRate", cfg.learning_rate)?;
        cfg.alpha = opt_f64_prop(options, "alpha", cfg.alpha)?;
        cfg.repulsion_scale = opt_f64_prop(options, "repulsionScale", cfg.repulsion_scale)?;
        cfg.rho_start = opt_f64_prop(options, "rhoStart", cfg.rho_start)?;
        cfg.rho_end = opt_f64_prop(options, "rhoEnd", cfg.rho_end)?;
        cfg.rho_power = opt_f64_prop(options, "rhoPower", cfg.rho_power)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    let inner = SgdSolver::<f32>::snap_t_fdp(graph.inner(), cfg).map_err(core_err_to_js)?;
    Ok(JsSgdSolver::from_inner(inner))
}

pub(crate) fn sgd_pacmap_graph(
    graph: &JsCsrGraph,
    options: &JsValue,
) -> Result<JsSgdSolver, JsError> {
    let mut cfg = PacmapGraphConfig::default();
    if check_options_arg(options, "SgdSolver.pacmapGraph")? {
        cfg.n_neighbors = opt_usize_prop(options, "nNeighbors", cfg.n_neighbors)?;
        cfg.mn_ratio = opt_f64_prop(options, "mnRatio", cfg.mn_ratio)?;
        cfg.fp_ratio = opt_f64_prop(options, "fpRatio", cfg.fp_ratio)?;
        cfg.phase1_iters = opt_usize_prop(options, "phase1Iters", cfg.phase1_iters)?;
        cfg.phase2_iters = opt_usize_prop(options, "phase2Iters", cfg.phase2_iters)?;
        cfg.phase3_iters = opt_usize_prop(options, "phase3Iters", cfg.phase3_iters)?;
        cfg.learning_rate = opt_f64_prop(options, "learningRate", cfg.learning_rate)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    let inner = SgdSolver::<f32>::pacmap_graph(graph.inner(), cfg).map_err(core_err_to_js)?;
    Ok(JsSgdSolver::from_inner(inner))
}

pub(crate) fn sgd_sgd2(graph: &JsCsrGraph, options: &JsValue) -> Result<JsSgdSolver, JsError> {
    let mut cfg = Sgd2Config::default();
    if check_options_arg(options, "SgdSolver.sgd2")? {
        cfg.t_max = opt_usize_prop(options, "tMax", cfg.t_max)?;
        cfg.eps = opt_f64_prop(options, "eps", cfg.eps)?;
        cfg.step_scale = opt_f64_prop(options, "stepScale", cfg.step_scale)?;
        cfg.batch_size = opt_usize_prop(options, "batchSize", cfg.batch_size)?;
        cfg.damping = opt_f64_prop(options, "damping", cfg.damping)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    let inner = SgdSolver::<f32>::sgd2(graph.inner(), cfg).map_err(core_err_to_js)?;
    Ok(JsSgdSolver::from_inner(inner))
}

// ============================================================================
// 3. ForceAtlas2Solver & YifanHuSolver (ContinuousFdp)
// ============================================================================

/// Cooperative ForceAtlas2 force-directed layout solver (`f32`).
#[wasm_bindgen(js_name = ForceAtlas2Solver)]
#[derive(Debug, Clone)]
pub struct JsForceAtlas2Solver {
    inner: ForceAtlas2Solver<f32>,
}

#[wasm_bindgen(js_class = ForceAtlas2Solver)]
impl JsForceAtlas2Solver {
    /// Constructs a `ForceAtlas2Solver` from `graph` and optional `ForceAtlas2Options`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsForceAtlas2Solver, JsError> {
        let mut cfg = ForceAtlas2Config::default();
        if check_options_arg(options, "ForceAtlas2Solver")? {
            cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
            cfg.theta = opt_f64_prop(options, "theta", cfg.theta)?;
            cfg.use_barnes_hut = opt_bool_prop(options, "useBarnesHut", cfg.use_barnes_hut)?;
            cfg.linlog_mode = opt_bool_prop(options, "linlogMode", cfg.linlog_mode)?;
            cfg.strong_gravity = opt_bool_prop(options, "strongGravity", cfg.strong_gravity)?;
            cfg.gravity = opt_f64_prop(options, "gravity", cfg.gravity)?;
            cfg.scaling_ratio = opt_f64_prop(options, "scalingRatio", cfg.scaling_ratio)?;
            cfg.edge_weight_influence =
                opt_f64_prop(options, "edgeWeightInfluence", cfg.edge_weight_influence)?;
            cfg.outbound_attraction_distribution = opt_bool_prop(
                options,
                "outboundAttractionDistribution",
                cfg.outbound_attraction_distribution,
            )?;
            cfg.jitter_tolerance = opt_f64_prop(options, "jitterTolerance", cfg.jitter_tolerance)?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner = ForceAtlas2Solver::<f32>::new(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one ForceAtlas2 iteration on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` ForceAtlas2 iterations on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "ForceAtlas2Solver::run_steps",
        )
    }

    /// Runs ForceAtlas2 to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "ForceAtlas2Solver::run",
        )
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    /// Returns the number of iterations executed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the scheduled iteration count.
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

/// Cooperative multilevel Yifan Hu (2005) force-directed layout solver (`f32`).
#[wasm_bindgen(js_name = YifanHuSolver)]
#[derive(Debug, Clone)]
pub struct JsYifanHuSolver {
    inner: YifanHuSolver<f32>,
}

#[wasm_bindgen(js_class = YifanHuSolver)]
impl JsYifanHuSolver {
    /// Constructs a `YifanHuSolver` from `graph` and optional `YifanHuOptions`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsYifanHuSolver, JsError> {
        let mut cfg = YifanHuConfig::default();
        if check_options_arg(options, "YifanHuSolver")? {
            cfg.max_iterations_per_level = opt_usize_prop(
                options,
                "maxIterationsPerLevel",
                cfg.max_iterations_per_level,
            )?;
            cfg.step_size = opt_f64_prop(options, "stepSize", cfg.step_size)?;
            cfg.step_ratio = opt_f64_prop(options, "stepRatio", cfg.step_ratio)?;
            cfg.relative_strength =
                opt_f64_prop(options, "relativeStrength", cfg.relative_strength)?;
            cfg.optimal_distance = opt_maybe_f64_prop(options, "optimalDistance")?;
            cfg.theta = opt_f64_prop(options, "theta", cfg.theta)?;
            cfg.min_coarsen_size = opt_usize_prop(options, "minCoarsenSize", cfg.min_coarsen_size)?;
            cfg.min_coarsen_ratio =
                opt_f64_prop(options, "minCoarsenRatio", cfg.min_coarsen_ratio)?;
            cfg.convergence_tolerance =
                opt_f64_prop(options, "convergenceTolerance", cfg.convergence_tolerance)?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner = YifanHuSolver::<f32>::new(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one Yifan Hu iteration on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` Yifan Hu iterations on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "YifanHuSolver::run_steps",
        )
    }

    /// Runs Yifan Hu multilevel optimization to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "YifanHuSolver::run",
        )
    }

    /// Returns the number of levels in the `EC + MIVS` coarsening hierarchy (`>= 1`).
    #[wasm_bindgen(getter, js_name = numLevels)]
    pub fn num_levels(&self) -> usize {
        self.inner.num_levels()
    }

    /// Returns the currently active level index (`numLevels - 1` coarsest down to `0` finest).
    #[wasm_bindgen(getter, js_name = currentLevel)]
    pub fn current_level(&self) -> usize {
        self.inner.current_level()
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    /// Returns the total iterations executed so far across all levels.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the maximum scheduled iterations (`numLevels * maxIterationsPerLevel`).
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

// ============================================================================
// 4. PrismOverlapRemoval (works under both `full` and `--no-default-features`)
// ============================================================================

fn parse_prism_config(options: &JsValue) -> Result<PrismConfig, JsError> {
    let mut cfg = PrismConfig::default();
    if check_options_arg(options, "PrismOverlapRemoval")? {
        cfg.max_iterations = opt_usize_prop(options, "maxIterations", cfg.max_iterations)?;
        cfg.max_scale_factor = opt_f64_prop(options, "maxScaleFactor", cfg.max_scale_factor)?;
        cfg.cg_max_iters = opt_usize_prop(options, "cgMaxIters", cfg.cg_max_iters)?;
        cfg.cg_tolerance = opt_f64_prop(options, "cgTolerance", cfg.cg_tolerance)?;
        cfg.padding = opt_f64_prop(options, "padding", cfg.padding)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    Ok(cfg)
}

/// Cooperative Gansner-Hu (2010) PRISM proximity-stress overlap removal solver (`f32`).
///
/// Uses `Delaunay2d` when compiled with `delaunay` and falls back automatically to
/// `StaticKdTree` $k$-NN proximity + `RectangleOverlapSweep` under `--no-default-features`.
#[wasm_bindgen(js_name = PrismOverlapRemoval)]
#[derive(Debug, Clone)]
pub struct JsPrismOverlapRemoval {
    inner: PrismOverlapRemoval<f32>,
}

#[wasm_bindgen(js_class = PrismOverlapRemoval)]
impl JsPrismOverlapRemoval {
    /// Constructs a `PrismOverlapRemoval` solver from per-node `halfWidths` and `halfHeights`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        half_widths: &[f32],
        half_heights: &[f32],
        options: &JsValue,
    ) -> Result<JsPrismOverlapRemoval, JsError> {
        let cfg = parse_prism_config(options)?;
        let inner =
            PrismOverlapRemoval::<f32>::new(half_widths.to_vec(), half_heights.to_vec(), cfg)
                .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `PrismOverlapRemoval` solver for circular/square nodes with equal
    /// half-width and half-height given by `radii`.
    #[wasm_bindgen(js_name = fromRadii)]
    pub fn from_radii(radii: &[f32], options: &JsValue) -> Result<JsPrismOverlapRemoval, JsError> {
        let cfg = parse_prism_config(options)?;
        let inner =
            PrismOverlapRemoval::<f32>::from_radii(radii.to_vec(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one PRISM proximity-stress round on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` PRISM rounds on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "PrismOverlapRemoval::run_steps",
        )
    }

    /// Runs PRISM overlap removal until `0` overlaps remain or `maxIterations` is reached.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "PrismOverlapRemoval::run",
        )
    }

    /// Returns the number of overlapping rectangle pairs remaining after the most recent step.
    #[wasm_bindgen(getter, js_name = lastOverlapCount)]
    pub fn last_overlap_count(&self) -> usize {
        self.inner.last_overlap_count()
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    /// Returns the number of PRISM rounds executed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the maximum scheduled PRISM rounds (`maxIterations`).
    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> usize {
        self.inner.total_iterations()
    }

    /// Returns `true` when the solver has converged (`0` overlaps) or completed.
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

// ============================================================================
// 5. SeparationConstraint & Constraint Projection Functions
// ============================================================================

/// One-dimensional separation constraint enforcing `coord[left] + gap <= coord[right]`.
#[wasm_bindgen(js_name = SeparationConstraint)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsSeparationConstraint {
    inner: SeparationConstraint<f32>,
}

#[wasm_bindgen(js_class = SeparationConstraint)]
impl JsSeparationConstraint {
    /// Creates a new `SeparationConstraint(left, right, gap)`.
    #[wasm_bindgen(constructor)]
    pub fn new(left: f64, right: f64, gap: f32) -> Result<JsSeparationConstraint, JsError> {
        let l = parse_non_negative_u32(left, "left")?;
        let r = parse_non_negative_u32(right, "right")?;
        if !gap.is_finite() {
            return Err(core_err_to_js(CoreError::non_finite(
                "SeparationConstraint::gap",
            )));
        }
        Ok(Self {
            inner: SeparationConstraint::new(l, r, gap),
        })
    }

    /// Returns the constrained left vertex index.
    #[wasm_bindgen(getter)]
    pub fn left(&self) -> u32 {
        self.inner.left
    }

    /// Returns the constrained right vertex index.
    #[wasm_bindgen(getter)]
    pub fn right(&self) -> u32 {
        self.inner.right
    }

    /// Returns the required minimum separation gap.
    #[wasm_bindgen(getter)]
    pub fn gap(&self) -> f32 {
        self.inner.gap
    }
}

fn parse_constraints(constraints: &JsValue) -> Result<Vec<SeparationConstraint<f32>>, JsError> {
    if constraints.is_null()
        || constraints.is_undefined()
        || constraints.is_string()
        || !constraints.is_object()
    {
        return Err(JsError::new(
            "constraints must be an array or iterable of SeparationConstraint objects",
        ));
    }
    let iter = js_sys::try_iter(constraints)
        .map_err(|err| JsError::new(&format!("failed to iterate constraints: {err:?}")))?
        .ok_or_else(|| {
            JsError::new("constraints must be an array or iterable of SeparationConstraint")
        })?;
    let mut out = Vec::new();
    for item_res in iter {
        let item = item_res
            .map_err(|err| JsError::new(&format!("failed to read constraint item: {err:?}")))?;
        if item.is_null() || !item.is_object() {
            return Err(JsError::new(
                "each constraint entry must be a SeparationConstraint or { left, right, gap } object",
            ));
        }
        let left_val = get_prop(&item, "left")?;
        let right_val = get_prop(&item, "right")?;
        let gap_val = get_prop(&item, "gap")?;

        let left_f64 = left_val
            .as_f64()
            .ok_or_else(|| JsError::new("constraint.left must be a non-negative integer"))?;
        let right_f64 = right_val
            .as_f64()
            .ok_or_else(|| JsError::new("constraint.right must be a non-negative integer"))?;
        let gap_f64 = gap_val
            .as_f64()
            .ok_or_else(|| JsError::new("constraint.gap must be a finite number"))?;

        let left = parse_non_negative_u32(left_f64, "constraint.left")?;
        let right = parse_non_negative_u32(right_f64, "constraint.right")?;
        let gap = gap_f64 as f32;
        if !gap.is_finite() {
            return Err(core_err_to_js(CoreError::non_finite("constraint.gap")));
        }
        out.push(SeparationConstraint::new(left, right, gap));
    }
    Ok(out)
}

fn parse_axis(axis: &JsValue) -> Result<usize, JsError> {
    if let Some(num) = axis.as_f64() {
        return parse_non_negative_usize(num, "axis");
    }
    if let Some(raw) = axis.as_string() {
        return match normalize_enum_token(&raw).as_str() {
            "x" | "0" => Ok(0),
            "y" | "1" => Ok(1),
            _ => Err(JsError::new(&format!(
                "layout axis must be 0 ('x') or 1 ('y'), got {raw:?}"
            ))),
        };
    }
    Err(JsError::new("layout axis must be 0 ('x') or 1 ('y')"))
}

/// Projects `coords` (`Float32Array`) in place onto the feasible region defined by
/// `constraints` and also returns the projected `Float32Array`.
#[wasm_bindgen(js_name = project1d)]
pub fn js_project_1d(coords: &mut [f32], constraints: &JsValue) -> Result<Vec<f32>, JsError> {
    let parsed = parse_constraints(constraints)?;
    let mut projector = ConstraintProjector::<f32>::new(coords.len());
    projector
        .project_1d(coords, &parsed)
        .map_err(core_err_to_js)?;
    Ok(coords.to_vec())
}

/// Projects axis `axis` (`0` / `"x"` or `1` / `"y"`) of `layout` in place onto `constraints`.
#[wasm_bindgen(js_name = projectLayoutAxis)]
pub fn js_project_layout_axis(
    layout: &mut JsLayout2d,
    axis: &JsValue,
    constraints: &JsValue,
) -> Result<(), JsError> {
    let axis_idx = parse_axis(axis)?;
    let parsed = parse_constraints(constraints)?;
    let mut projector = ConstraintProjector::<f32>::new(layout.len());
    let mut soa = layout.inner().to_layout2d();
    projector
        .project_layout_axis(&mut soa, axis_idx, &parsed)
        .map_err(core_err_to_js)?;
    layout
        .inner_mut()
        .copy_from_layout2d(&soa)
        .map_err(ffi_err_to_js)?;
    Ok(())
}

/// Generates hierarchical separation constraints `y[u] + gap <= y[v]` for directed edges
/// `(u, v)` spanning `layers[u] < layers[v]`.
#[wasm_bindgen(js_name = generateLayeredConstraints)]
pub fn js_generate_layered_constraints(
    graph: &JsCsrGraph,
    layers: &[u32],
    min_layer_distance: Option<f32>,
) -> Result<Vec<JsSeparationConstraint>, JsError> {
    let layers_usize: Vec<usize> = layers.iter().map(|&v| v as usize).collect();
    let dist = min_layer_distance.unwrap_or(DEFAULT_MIN_LAYER_DISTANCE);
    let raw =
        generate_layered_constraints(graph.inner(), &layers_usize, dist).map_err(core_err_to_js)?;
    Ok(raw
        .into_iter()
        .map(|inner| JsSeparationConstraint { inner })
        .collect())
}

/// Projects 2D rectangle positions in `layout` in place to eliminate all axis-aligned overlaps
/// while minimizing quadratic coordinate displacement.
#[wasm_bindgen(js_name = projectRectangleNoOverlap2d)]
pub fn js_project_rectangle_no_overlap_2d(
    layout: &mut JsLayout2d,
    half_widths: &[f32],
    half_heights: &[f32],
) -> Result<(), JsError> {
    let mut soa = layout.inner().to_layout2d();
    project_rectangle_no_overlap_2d(&mut soa, half_widths, half_heights).map_err(core_err_to_js)?;
    layout
        .inner_mut()
        .copy_from_layout2d(&soa)
        .map_err(ffi_err_to_js)?;
    Ok(())
}

/// Eliminates overlaps between axis-aligned cluster bounding boxes in `layout` in place while
/// rigidly preserving relative coordinates within each cluster.
#[wasm_bindgen(js_name = projectClusteredRectangleNoOverlap2d)]
pub fn js_project_clustered_rectangle_no_overlap_2d(
    layout: &mut JsLayout2d,
    cluster_ids: &[u32],
    half_widths: &[f32],
    half_heights: &[f32],
) -> Result<(), JsError> {
    let mut soa = layout.inner().to_layout2d();
    project_clustered_rectangle_no_overlap_2d(&mut soa, cluster_ids, half_widths, half_heights)
        .map_err(core_err_to_js)?;
    layout
        .inner_mut()
        .copy_from_layout2d(&soa)
        .map_err(ffi_err_to_js)?;
    Ok(())
}

// ============================================================================
// 6. Layering (cycleEdges, removeCycles, assignLayers)
// ============================================================================

fn parse_layering_method(method: Option<String>) -> Result<LayeringMethod, JsError> {
    match method {
        None => Ok(LayeringMethod::LongestPath),
        Some(raw) => match normalize_enum_token(&raw).as_str() {
            "longestpath" | "lp" => Ok(LayeringMethod::LongestPath),
            "networksimplex" | "ns" => Ok(LayeringMethod::NetworkSimplex),
            _ => Err(JsError::new(&format!(
                "unknown LayeringMethod {raw:?}; expected 'longestPath' or 'networkSimplex'"
            ))),
        },
    }
}

/// Identifies back-edges `(u, v)` whose reversal eliminates all directed cycles in `graph`,
/// returned as a flat interleaved `[u0, v0, u1, v1, ...]` `Uint32Array`.
#[wasm_bindgen(js_name = cycleEdges)]
pub fn js_cycle_edges(graph: &JsCsrGraph) -> Result<Vec<u32>, JsError> {
    let pairs = cycle_edges(graph.inner()).map_err(core_err_to_js)?;
    let mut flat = Vec::with_capacity(pairs.len() * 2);
    for (u, v) in pairs {
        flat.push(u);
        flat.push(v);
    }
    Ok(flat)
}

/// Constructs a directed acyclic `CsrGraph` by reversing cycle-closing back-edges.
#[wasm_bindgen(js_name = removeCycles)]
pub fn js_remove_cycles(graph: &JsCsrGraph) -> Result<JsCsrGraph, JsError> {
    let dag = remove_cycles(graph.inner()).map_err(core_err_to_js)?;
    Ok(JsCsrGraph::from_inner(dag))
}

/// Assigns hierarchical layers (`0, 1, ...`) to all vertices of a directed `CsrGraph`.
#[wasm_bindgen(js_name = assignLayers)]
pub fn js_assign_layers(graph: &JsCsrGraph, method: Option<String>) -> Result<Vec<u32>, JsError> {
    let m = parse_layering_method(method)?;
    let layers = assign_layers(graph.inner(), m).map_err(core_err_to_js)?;
    let mut out = Vec::with_capacity(layers.len());
    for layer in layers {
        let l_u32 = u32::try_from(layer)
            .map_err(|_| JsError::new(&format!("layer index {layer} exceeds u32::MAX")))?;
        out.push(l_u32);
    }
    Ok(out)
}

// ============================================================================
// 7. FdebSolver (Reference-passing Force-Directed Edge Bundling)
// ============================================================================

/// Cooperative Force-Directed Edge Bundling (`FDEB`) solver (`f32`) storing flat edge
/// polyline buffers without `serde` or consuming input `Layout2d` handles.
#[wasm_bindgen(js_name = FdebSolver)]
#[derive(Debug, Clone)]
pub struct JsFdebSolver {
    inner: FdebSolver<f32>,
}

#[wasm_bindgen(js_class = FdebSolver)]
impl JsFdebSolver {
    /// Constructs an `FdebSolver` from `graph` and `layout` by shared reference.
    #[wasm_bindgen(constructor)]
    pub fn new(
        graph: &JsCsrGraph,
        layout: &JsLayout2d,
        options: &JsValue,
    ) -> Result<JsFdebSolver, JsError> {
        let mut cfg = FdebConfig::default();
        if check_options_arg(options, "FdebSolver")? {
            cfg.cycles = opt_usize_prop(options, "cycles", cfg.cycles)?;
            cfg.s0 = opt_f64_prop(options, "s0", cfg.s0)?;
            cfg.i0 = opt_usize_prop(options, "i0", cfg.i0)?;
            cfg.s_step = opt_f64_prop(options, "sStep", cfg.s_step)?;
            cfg.i_step = opt_f64_prop(options, "iStep", cfg.i_step)?;
            cfg.spring_k = opt_f64_prop(options, "springK", cfg.spring_k)?;
            cfg.minimum_edge_compatibility = opt_f64_prop(
                options,
                "minimumEdgeCompatibility",
                cfg.minimum_edge_compatibility,
            )?;
        }
        let soa = layout.inner().to_layout2d();
        let inner = FdebSolver::<f32>::new(graph.inner(), &soa, cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one FDEB iteration and returns the updated solver status string.
    pub fn step(&mut self) -> Result<String, JsError> {
        let status = self.inner.step().map_err(core_err_to_js)?;
        Ok(solver_status_str(status).to_owned())
    }

    /// Executes up to `steps` FDEB iterations and returns the updated solver status string.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, steps: f64) -> Result<String, JsError> {
        let n_steps = parse_non_negative_usize(steps, "steps")?;
        let status = self.inner.run_steps(n_steps).map_err(core_err_to_js)?;
        Ok(solver_status_str(status).to_owned())
    }

    /// Runs FDEB edge bundling to completion and returns the final solver status string.
    pub fn run(&mut self) -> Result<String, JsError> {
        let status = self.inner.run().map_err(core_err_to_js)?;
        Ok(solver_status_str(status).to_owned())
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    /// Returns the number of iterations executed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the total scheduled iterations across all subdivision cycles.
    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> usize {
        self.inner.total_iterations()
    }

    /// Returns `true` when edge bundling has converged or completed.
    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }

    /// Returns the number of edges being bundled.
    #[wasm_bindgen(getter, js_name = numEdges)]
    pub fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    /// Returns CSR-style polyline point offsets of length `numEdges + 1` (`Uint32Array`).
    #[wasm_bindgen(js_name = edgeOffsets)]
    pub fn edge_offsets(&self) -> Vec<u32> {
        self.inner.edge_offsets().to_vec()
    }

    /// Returns all bundled polyline control points packed as flat interleaved `[x0, y0, ...]` (`Float32Array`).
    pub fn points(&self) -> Vec<f32> {
        self.inner.to_interleaved_points()
    }
}
