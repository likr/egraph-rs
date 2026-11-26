//! Cooperative bounded Student-$t$ force-directed placement (`t-FDP`) WebAssembly bindings (`f32`).

use egraph::core::{CoreError, IterativeSolver};
use egraph::layout::{FieldLayoutSolver, TFdpConfig};
use egraph::neighborhood::KnnGraph;
use wasm_bindgen::{JsCast, prelude::*};

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    core_err_to_js, parse_f64_sequence, parse_non_negative_usize, parse_u32_sequence,
};
use crate::js_options::{
    check_options_arg, get_prop, is_plain_js_object, opt_f64_prop, opt_seed_prop, opt_string_prop,
    opt_usize_prop, parse_field_repulsion_backend, parse_ibfft_order,
    reject_extra_args_with_options, solver_status_str,
};
use crate::layout_buffers::JsLayout2d;
use crate::solver_runner::{run_layout_in_place, run_steps_layout_in_place, step_layout_in_place};

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

/// Cooperative bounded Student-$t$ force-directed placement (`t-FDP`) solver (`f32`)
/// supporting `ibfft`, `barnes_hut`, and `exact` repulsion backends.
#[wasm_bindgen(js_name = TFdpSolver)]
#[derive(Debug, Clone)]
pub struct JsTFdpSolver {
    inner: FieldLayoutSolver<f32>,
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
