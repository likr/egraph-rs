//! WebAssembly bindings for GEM and Structural Entropy Layout (`SEL`) baseline solvers (`f32`).

use egraph::layout::force::BarnesHut as ForceBarnesHut;
use egraph::layout::neighbor::{BarnesHut as NeighborBarnesHut, Exact as NeighborExact, IbFft};
use egraph::layout::{
    FrBackend, FruchtermanReingoldConfig, GemConfig, GemSolver, SelConfig, SelLayoutBackend,
    SelLayoutSolver, SelWeightMapping, TFdpBackend2d, TFdpConfig,
};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::force_classic::DEFAULT_BARNES_HUT_THETA;
use crate::force_fr::parse_fr_config;
use crate::js_convert::{core_err_to_js, invalid_input_err, unsupported_err};
use crate::js_options::{
    JsTargetKind, check_options_arg, get_prop, normalize_enum_token, opt_f64_prop,
    opt_maybe_bool_prop, opt_maybe_f64_prop, opt_maybe_usize_prop, opt_seed_prop, opt_string_prop,
    opt_usize_prop, parse_target_kind, solver_status_str,
};
use crate::layout_buffers::JsLayout2d;
use crate::solver_runner::{run_layout_in_place, run_steps_layout_in_place, step_layout_in_place};
use crate::t_fdp::{parse_ibfft_from_options, parse_t_fdp_config_with_ctx};

/// Cooperative Frick et al. (1995) GEM adaptive local-temperature layout solver (`f32`).
#[wasm_bindgen(js_name = GemSolver)]
#[derive(Debug, Clone)]
pub struct JsGemSolver {
    inner: GemSolver<f32>,
}

#[wasm_bindgen(js_class = GemSolver)]
impl JsGemSolver {
    /// Constructs a `GemSolver` from `graph` and optional `GemOptions`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsGemSolver, JsValue> {
        let mut cfg = GemConfig::default();
        if check_options_arg(options, "GemSolver")? {
            cfg.edge_length = opt_f64_prop(options, "edgeLength", cfg.edge_length)?;
            cfg.gravity = opt_f64_prop(options, "gravity", cfg.gravity)?;
            cfg.t_init = opt_f64_prop(options, "tInit", cfg.t_init)?;
            cfg.t_max = opt_f64_prop(options, "tMax", cfg.t_max)?;
            cfg.t_min = opt_f64_prop(options, "tMin", cfg.t_min)?;
            cfg.max_iterations = opt_maybe_usize_prop(options, "maxIterations")?;
            cfg.oscillation_angle =
                opt_f64_prop(options, "oscillationAngle", cfg.oscillation_angle)?;
            cfg.rotation_angle = opt_f64_prop(options, "rotationAngle", cfg.rotation_angle)?;
            cfg.oscillation_sensitivity = opt_f64_prop(
                options,
                "oscillationSensitivity",
                cfg.oscillation_sensitivity,
            )?;
            cfg.rotation_sensitivity = opt_maybe_f64_prop(options, "rotationSensitivity")?;
            cfg.perturbation = opt_f64_prop(options, "perturbation", cfg.perturbation)?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner = GemSolver::<f32>::new(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one GEM round on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` GEM rounds on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsValue> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "GemSolver::run_steps",
        )
    }

    /// Runs GEM layout optimization to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "GemSolver::run",
        )
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    /// Returns the number of rounds executed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the maximum scheduled iterations (`maxIterations`).
    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> Option<usize> {
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

fn parse_sel_weight_mapping(raw: &str) -> Result<SelWeightMapping, JsValue> {
    match normalize_enum_token(raw)?.as_str() {
        "sigmoid" => Ok(SelWeightMapping::Sigmoid),
        "linear" => Ok(SelWeightMapping::Linear),
        "tanh" => Ok(SelWeightMapping::Tanh),
        "customlogistic" => Ok(SelWeightMapping::CustomLogistic),
        "binary" => Ok(SelWeightMapping::Binary),
        _ => Err(invalid_input_err(format!(
            "unknown SelWeightMapping {raw:?}; expected 'sigmoid', 'linear', 'tanh', 'customLogistic', or 'binary'"
        ))),
    }
}

fn parse_sel_layout_backend(raw: &str) -> Result<SelLayoutBackend, JsValue> {
    match normalize_enum_token(raw)?.as_str() {
        "fr" | "fruchtermanreingold" => Ok(SelLayoutBackend::Fr),
        "tfdp" => Ok(SelLayoutBackend::TFdp),
        _ => Err(invalid_input_err(format!(
            "unknown SelLayoutBackend {raw:?}; expected 'fr' or 'tFdp'"
        ))),
    }
}

fn ensure_sel_subconfig_target_2d(val: &JsValue, ctx: &str) -> Result<(), JsValue> {
    if let Some(target_str) = opt_string_prop(val, "target")?
        && parse_target_kind(&target_str)? != JsTargetKind::Layout2d
    {
        return Err(unsupported_err(format!(
            "{ctx} only supports target 'layout2d', got {target_str:?}"
        )));
    }
    Ok(())
}

fn parse_sel_fr_config(
    fr_val: &JsValue,
) -> Result<(FruchtermanReingoldConfig, FrBackend), JsValue> {
    let cfg = parse_fr_config(fr_val, "SelLayoutSolver.frConfig")?;
    ensure_sel_subconfig_target_2d(fr_val, "SelLayoutSolver.frConfig")?;
    let theta = opt_f64_prop(fr_val, "theta", DEFAULT_BARNES_HUT_THETA)?;
    ForceBarnesHut::new(theta)
        .validate()
        .map_err(core_err_to_js)?;
    let backend_opt = opt_string_prop(fr_val, "backend")?;
    let use_bh_opt = opt_maybe_bool_prop(fr_val, "useBarnesHut")?;
    let fr_backend = match backend_opt {
        Some(ref b) => match normalize_enum_token(b)?.as_str() {
            "exact" => {
                if use_bh_opt == Some(true) {
                    return Err(invalid_input_err(
                        "conflicting SelLayoutSolver.frConfig options: backend='exact' cannot be combined with useBarnesHut=true",
                    ));
                }
                FrBackend::exact()
            }
            "barneshut" | "bh" => {
                if use_bh_opt == Some(false) {
                    return Err(invalid_input_err(
                        "conflicting SelLayoutSolver.frConfig options: backend='barnesHut' cannot be combined with useBarnesHut=false",
                    ));
                }
                FrBackend::barnes_hut(theta)
            }
            "grid" => {
                if use_bh_opt == Some(true) {
                    return Err(invalid_input_err(
                        "conflicting SelLayoutSolver.frConfig options: backend='grid' cannot be combined with useBarnesHut=true",
                    ));
                }
                FrBackend::grid()
            }
            _ => {
                return Err(invalid_input_err(format!(
                    "unknown FrBackend {b:?}; expected 'exact', 'barnesHut', or 'grid'"
                )));
            }
        },
        None => {
            if use_bh_opt == Some(true) {
                FrBackend::barnes_hut(theta)
            } else {
                FrBackend::exact()
            }
        }
    };
    cfg.validate().map_err(core_err_to_js)?;
    fr_backend.validate().map_err(core_err_to_js)?;
    Ok((cfg, fr_backend))
}

fn parse_sel_t_fdp_config(tfdp_val: &JsValue) -> Result<(TFdpConfig, TFdpBackend2d), JsValue> {
    let cfg = parse_t_fdp_config_with_ctx(tfdp_val, "SelLayoutSolver.tFdpConfig")?;
    ensure_sel_subconfig_target_2d(tfdp_val, "SelLayoutSolver.tFdpConfig")?;
    let theta = opt_f64_prop(tfdp_val, "theta", NeighborBarnesHut::default().theta)?;
    NeighborBarnesHut::new(theta)
        .validate()
        .map_err(core_err_to_js)?;
    let ibfft_cfg = parse_ibfft_from_options(tfdp_val)?;
    ibfft_cfg.validate().map_err(core_err_to_js)?;
    let backend_opt = opt_string_prop(tfdp_val, "backend")?;
    let use_bh_opt = opt_maybe_bool_prop(tfdp_val, "useBarnesHut")?;
    let t_fdp_backend = match backend_opt {
        Some(ref b) => match normalize_enum_token(b)?.as_str() {
            "ibfft" | "fft" => {
                if use_bh_opt == Some(true) {
                    return Err(invalid_input_err(
                        "conflicting SelLayoutSolver.tFdpConfig options: backend='ibFft' cannot be combined with useBarnesHut=true",
                    ));
                }
                TFdpBackend2d::IbFft(IbFft::new().with_ibfft(ibfft_cfg))
            }
            "barneshut" | "bh" => {
                if use_bh_opt == Some(false) {
                    return Err(invalid_input_err(
                        "conflicting SelLayoutSolver.tFdpConfig options: backend='barnesHut' cannot be combined with useBarnesHut=false",
                    ));
                }
                TFdpBackend2d::BarnesHut(NeighborBarnesHut::new(theta))
            }
            "exact" => {
                if use_bh_opt == Some(true) {
                    return Err(invalid_input_err(
                        "conflicting SelLayoutSolver.tFdpConfig options: backend='exact' cannot be combined with useBarnesHut=true",
                    ));
                }
                TFdpBackend2d::Exact(NeighborExact::new())
            }
            _ => {
                return Err(invalid_input_err(format!(
                    "unknown TFdpSolver backend {b:?}; expected 'ibFft', 'barnesHut', or 'exact'"
                )));
            }
        },
        None => {
            if use_bh_opt == Some(true) {
                TFdpBackend2d::BarnesHut(NeighborBarnesHut::new(theta))
            } else {
                TFdpBackend2d::IbFft(IbFft::new().with_ibfft(ibfft_cfg))
            }
        }
    };
    cfg.validate().map_err(core_err_to_js)?;
    t_fdp_backend.validate().map_err(core_err_to_js)?;
    Ok((cfg, t_fdp_backend))
}

/// Cooperative Xue et al. (2025) Structural Entropy Layout (`SEL`) solver (`f32`).
#[wasm_bindgen(js_name = SelLayoutSolver)]
#[derive(Debug, Clone)]
pub struct JsSelLayoutSolver {
    inner: SelLayoutSolver<f32>,
}

#[wasm_bindgen(js_class = SelLayoutSolver)]
impl JsSelLayoutSolver {
    /// Constructs a `SelLayoutSolver` from `graph` and optional `SelOptions`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsSelLayoutSolver, JsValue> {
        let mut cfg = SelConfig::default();
        if check_options_arg(options, "SelLayoutSolver")? {
            cfg.tree_depth = opt_usize_prop(options, "treeDepth", cfg.tree_depth)?;
            cfg.pq_max_pops = opt_usize_prop(options, "pqMaxPops", cfg.pq_max_pops)?;
            if let Some(m_str) = opt_string_prop(options, "mapping")? {
                cfg.mapping = parse_sel_weight_mapping(&m_str)?;
            }
            if let Some(b_str) = opt_string_prop(options, "layoutBackend")? {
                cfg.layout_backend = parse_sel_layout_backend(&b_str)?;
            }
            cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
            cfg.seed = opt_seed_prop(options)?;
            let fr_val = get_prop(options, "frConfig")?;
            if !fr_val.is_undefined() && !fr_val.is_null() {
                let (fr_cfg, fr_backend) = parse_sel_fr_config(&fr_val)?;
                cfg = cfg.with_fr_config(Some(fr_cfg)).with_fr_backend(fr_backend);
            }
            let tfdp_val = get_prop(options, "tFdpConfig")?;
            if !tfdp_val.is_undefined() && !tfdp_val.is_null() {
                let (tfdp_cfg, tfdp_backend) = parse_sel_t_fdp_config(&tfdp_val)?;
                cfg = cfg
                    .with_t_fdp_config(Some(tfdp_cfg))
                    .with_t_fdp_backend(tfdp_backend);
            }
        }
        let inner = SelLayoutSolver::<f32>::new(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one SEL layout iteration on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` SEL layout iterations on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsValue> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "SelLayoutSolver::run_steps",
        )
    }

    /// Runs SEL layout optimization to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "SelLayoutSolver::run",
        )
    }

    /// Returns the SEL-modulated edge weights aligned with canonical undirected edges `(u < v)` (`Float64Array`).
    #[wasm_bindgen(js_name = modulatedWeights)]
    pub fn modulated_weights(&self) -> Vec<f64> {
        self.inner.modulated_weights().to_vec()
    }

    /// Returns the elbow-selected structural entropy threshold $\mathrm{SE}_{\mathrm{target}}$.
    #[wasm_bindgen(getter, js_name = seTarget)]
    pub fn se_target(&self) -> f64 {
        self.inner.se_target()
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
    pub fn total_iterations(&self) -> Option<usize> {
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
