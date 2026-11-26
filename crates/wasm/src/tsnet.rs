//! Cooperative unified `TsNetSolver` (`f32`) WebAssembly bindings.

use egraph::layout::TsNetSolver;
use egraph::spatial::FitFieldStats;
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, parse_optional_usize};
use crate::js_options::solver_status_str;
use crate::layout_buffers::JsLayout2d;
use crate::neighborhood_types::{JsKnnGraph, JsSparseProbabilities};
use crate::solver_runner::{run_layout_in_place, run_steps_layout_in_place, step_layout_in_place};
use crate::tsnet_options::{parse_row_normalized_tsnet_args, parse_tsnet_config};

const DEFAULT_UNIFORM_PERPLEXITY: f64 = 30.0;

/// FFT field telemetry counters returned by `TsNetSolver.lastFftStats()`.
#[wasm_bindgen(js_name = FitFieldStats)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JsFitFieldStats {
    inner: FitFieldStats,
}

#[wasm_bindgen(js_class = FitFieldStats)]
impl JsFitFieldStats {
    /// Number of forward 2D FFTs executed on particle charge grids.
    #[wasm_bindgen(getter, js_name = particleForwardFfts)]
    pub fn particle_forward_ffts(&self) -> f64 {
        self.inner.particle_forward_ffts as f64
    }

    /// Number of forward 2D FFTs executed on kernel grids.
    #[wasm_bindgen(getter, js_name = kernelForwardFfts)]
    pub fn kernel_forward_ffts(&self) -> f64 {
        self.inner.kernel_forward_ffts as f64
    }

    /// Number of inverse 2D FFTs executed.
    #[wasm_bindgen(getter, js_name = inverseFfts)]
    pub fn inverse_ffts(&self) -> f64 {
        self.inner.inverse_ffts as f64
    }

    /// Total `fft_2d` calls (`particleForwardFfts + kernelForwardFfts + inverseFfts`).
    #[wasm_bindgen(getter, js_name = totalFft2dCalls)]
    pub fn total_fft_2d_calls(&self) -> f64 {
        self.inner.total_fft_2d_calls as f64
    }
}

/// Cooperative unified tsNET / BH-tsNET / FIt-tsNET / L-tsNET solver (`f32`).
#[wasm_bindgen(js_name = TsNetSolver)]
#[derive(Debug, Clone)]
pub struct JsTsNetSolver {
    inner: TsNetSolver<f32>,
}

#[wasm_bindgen(js_class = TsNetSolver)]
impl JsTsNetSolver {
    /// Constructs a `TsNetSolver` using the Pure 1+2-Hop node-wise adaptive perplexity preset.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = pure2Hop)]
    pub fn pure_2hop(
        graph: &JsCsrGraph,
        backend_or_options: &JsValue,
        iterations_stage1: Option<f64>,
        iterations_stage2: Option<f64>,
        iterations_stage3: Option<f64>,
        learning_rate: Option<f64>,
        perplexity_c: Option<f64>,
        min_perplexity: Option<f64>,
        max_perplexity: Option<f64>,
        exaggeration: Option<f64>,
        lambda_c_stage1: Option<f64>,
        lambda_c_stage2: Option<f64>,
        lambda_c_stage3: Option<f64>,
        lambda_r_stage3: Option<f64>,
        momentum: Option<f64>,
        epsilon_r: Option<f64>,
        theta: Option<f64>,
        power: Option<f64>,
        sigma_iters: Option<f64>,
        sigma_tolerance: Option<f64>,
        row_normalize: Option<bool>,
        seed: Option<f64>,
        intervals: Option<f64>,
        interpolation_order: &JsValue,
        min_grid_dim: Option<f64>,
        max_grid_dim: Option<f64>,
    ) -> Result<JsTsNetSolver, JsError> {
        let cfg = parse_tsnet_config(
            backend_or_options,
            iterations_stage1,
            iterations_stage2,
            iterations_stage3,
            learning_rate,
            perplexity_c,
            min_perplexity,
            max_perplexity,
            exaggeration,
            lambda_c_stage1,
            lambda_c_stage2,
            lambda_c_stage3,
            lambda_r_stage3,
            momentum,
            epsilon_r,
            theta,
            power,
            sigma_iters,
            sigma_tolerance,
            row_normalize,
            seed,
            intervals,
            interpolation_order,
            min_grid_dim,
            max_grid_dim,
        )?;
        let inner =
            TsNetSolver::<f32>::from_graph_pure_2hop(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `TsNetSolver` from a `KnnGraph` and optional per-node `perplexities`.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = fromKnn)]
    pub fn from_knn(
        knn: &JsKnnGraph,
        perplexities: Option<Vec<f64>>,
        uniform_perplexity: Option<f64>,
        backend_or_options: &JsValue,
        iterations_stage1: Option<f64>,
        iterations_stage2: Option<f64>,
        iterations_stage3: Option<f64>,
        learning_rate: Option<f64>,
        exaggeration: Option<f64>,
        lambda_c_stage1: Option<f64>,
        lambda_c_stage2: Option<f64>,
        lambda_c_stage3: Option<f64>,
        lambda_r_stage3: Option<f64>,
        momentum: Option<f64>,
        epsilon_r: Option<f64>,
        theta: Option<f64>,
        power: Option<f64>,
        sigma_iters: Option<f64>,
        sigma_tolerance: Option<f64>,
        row_normalize: Option<bool>,
        seed: Option<f64>,
        intervals: Option<f64>,
        interpolation_order: &JsValue,
        min_grid_dim: Option<f64>,
        max_grid_dim: Option<f64>,
    ) -> Result<JsTsNetSolver, JsError> {
        let cfg = parse_tsnet_config(
            backend_or_options,
            iterations_stage1,
            iterations_stage2,
            iterations_stage3,
            learning_rate,
            None,
            None,
            None,
            exaggeration,
            lambda_c_stage1,
            lambda_c_stage2,
            lambda_c_stage3,
            lambda_r_stage3,
            momentum,
            epsilon_r,
            theta,
            power,
            sigma_iters,
            sigma_tolerance,
            row_normalize,
            seed,
            intervals,
            interpolation_order,
            min_grid_dim,
            max_grid_dim,
        )?;
        let uni = uniform_perplexity.unwrap_or(DEFAULT_UNIFORM_PERPLEXITY);
        let inner = TsNetSolver::<f32>::from_knn(knn.inner(), perplexities.as_deref(), uni, cfg)
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `TsNetSolver` directly from precomputed symmetric `SparseProbabilities`.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = fromProbabilities)]
    pub fn from_probabilities(
        probs: &JsSparseProbabilities,
        backend_or_options: &JsValue,
        iterations_stage1: Option<f64>,
        iterations_stage2: Option<f64>,
        iterations_stage3: Option<f64>,
        learning_rate: Option<f64>,
        exaggeration: Option<f64>,
        lambda_c_stage1: Option<f64>,
        lambda_c_stage2: Option<f64>,
        lambda_c_stage3: Option<f64>,
        lambda_r_stage3: Option<f64>,
        momentum: Option<f64>,
        epsilon_r: Option<f64>,
        theta: Option<f64>,
        seed: Option<f64>,
        intervals: Option<f64>,
        interpolation_order: &JsValue,
        min_grid_dim: Option<f64>,
        max_grid_dim: Option<f64>,
    ) -> Result<JsTsNetSolver, JsError> {
        let cfg = parse_tsnet_config(
            backend_or_options,
            iterations_stage1,
            iterations_stage2,
            iterations_stage3,
            learning_rate,
            None,
            None,
            None,
            exaggeration,
            lambda_c_stage1,
            lambda_c_stage2,
            lambda_c_stage3,
            lambda_r_stage3,
            momentum,
            epsilon_r,
            theta,
            None,
            None,
            None,
            None,
            seed,
            intervals,
            interpolation_order,
            min_grid_dim,
            max_grid_dim,
        )?;
        let inner = TsNetSolver::<f32>::from_probabilities(probs.inner().clone(), cfg)
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `TsNetSolver` from a `CsrGraph` using Partial BFS $k$-NN and a uniform perplexity.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = fromGraphPartialBfs)]
    pub fn from_graph_partial_bfs(
        graph: &JsCsrGraph,
        k: Option<f64>,
        uniform_perplexity: Option<f64>,
        backend_or_options: &JsValue,
        iterations_stage1: Option<f64>,
        iterations_stage2: Option<f64>,
        iterations_stage3: Option<f64>,
        learning_rate: Option<f64>,
        exaggeration: Option<f64>,
        lambda_c_stage1: Option<f64>,
        lambda_c_stage2: Option<f64>,
        lambda_c_stage3: Option<f64>,
        lambda_r_stage3: Option<f64>,
        momentum: Option<f64>,
        epsilon_r: Option<f64>,
        theta: Option<f64>,
        power: Option<f64>,
        sigma_iters: Option<f64>,
        sigma_tolerance: Option<f64>,
        row_normalize: Option<bool>,
        seed: Option<f64>,
        intervals: Option<f64>,
        interpolation_order: &JsValue,
        min_grid_dim: Option<f64>,
        max_grid_dim: Option<f64>,
    ) -> Result<JsTsNetSolver, JsError> {
        let k_val = parse_optional_usize(k, 30, "k")?;
        let uni = uniform_perplexity.unwrap_or(DEFAULT_UNIFORM_PERPLEXITY);
        let cfg = parse_tsnet_config(
            backend_or_options,
            iterations_stage1,
            iterations_stage2,
            iterations_stage3,
            learning_rate,
            None,
            None,
            None,
            exaggeration,
            lambda_c_stage1,
            lambda_c_stage2,
            lambda_c_stage3,
            lambda_r_stage3,
            momentum,
            epsilon_r,
            theta,
            power,
            sigma_iters,
            sigma_tolerance,
            row_normalize,
            seed,
            intervals,
            interpolation_order,
            min_grid_dim,
            max_grid_dim,
        )?;
        let inner = TsNetSolver::<f32>::from_graph_partial_bfs(graph.inner(), k_val, uni, cfg)
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `TsNetSolver` from a `CsrGraph` using direct row-normalized `A1 + w2 * A2` affinity.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = fromGraphRowNormalized)]
    pub fn from_graph_row_normalized(
        graph: &JsCsrGraph,
        w2_or_options: &JsValue,
        max_2hop_per_node: Option<f64>,
        backend_or_options: &JsValue,
        iterations_stage1: Option<f64>,
        iterations_stage2: Option<f64>,
        iterations_stage3: Option<f64>,
        learning_rate: Option<f64>,
        exaggeration: Option<f64>,
        lambda_c_stage1: Option<f64>,
        lambda_c_stage2: Option<f64>,
        lambda_c_stage3: Option<f64>,
        lambda_r_stage3: Option<f64>,
        momentum: Option<f64>,
        epsilon_r: Option<f64>,
        theta: Option<f64>,
        seed: Option<f64>,
        intervals: Option<f64>,
        interpolation_order: &JsValue,
        min_grid_dim: Option<f64>,
        max_grid_dim: Option<f64>,
        degree_beta: Option<f64>,
    ) -> Result<JsTsNetSolver, JsError> {
        let (w2, max_2hop, beta_val, cfg) = parse_row_normalized_tsnet_args(
            w2_or_options,
            max_2hop_per_node,
            backend_or_options,
            iterations_stage1,
            iterations_stage2,
            iterations_stage3,
            learning_rate,
            exaggeration,
            lambda_c_stage1,
            lambda_c_stage2,
            lambda_c_stage3,
            lambda_r_stage3,
            momentum,
            epsilon_r,
            theta,
            seed,
            intervals,
            interpolation_order,
            min_grid_dim,
            max_grid_dim,
            degree_beta,
        )?;
        let inner = TsNetSolver::<f32>::from_graph_row_normalized_with_degree_decay(
            graph.inner(),
            w2,
            max_2hop,
            beta_val,
            cfg,
        )
        .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Returns telemetry counters from the most recent FFT field evaluation, or `undefined`.
    #[wasm_bindgen(js_name = lastFftStats)]
    pub fn last_fft_stats(&self) -> Option<JsFitFieldStats> {
        self.inner
            .last_fft_stats()
            .map(|inner| JsFitFieldStats { inner })
    }

    /// Executes one optimization iteration on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` iterations on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "TsNetSolver::run_steps",
        )
    }

    /// Runs optimization to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "TsNetSolver::run",
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

    /// Returns the total scheduled iteration count across all stages.
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

    /// Returns the active (or final) 0-based stage index.
    #[wasm_bindgen(getter, js_name = stageIndex)]
    pub fn stage_index(&self) -> usize {
        self.inner.stage_index()
    }

    /// Returns the cumulative number of `fft_2d` calls executed across all steps.
    #[wasm_bindgen(getter, js_name = totalFft2dCalls)]
    pub fn total_fft_2d_calls(&self) -> f64 {
        self.inner.total_fft_2d_calls() as f64
    }

    /// Returns the stability-capped learning rate of the current stage.
    #[wasm_bindgen(getter, js_name = currentStageLr)]
    pub fn current_stage_lr(&self) -> f64 {
        self.inner.current_stage_lr()
    }
}
