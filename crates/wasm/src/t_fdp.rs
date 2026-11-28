//! Cooperative bounded Student-$t$ force-directed placement (`t-FDP`) WebAssembly bindings (`f32`)
//! supporting Policy 1A + Option 1 `(target, backend)` dispatch across 2D, 3D, and N-D.

use egraph::core::{CoreError, Layout2d, Layout3d, LayoutNd};
use egraph::layout::{
    TFdpConfig, TFdpSolver,
    neighbor::{
        BarnesHut as NeighborBarnesHut, Exact as NeighborExact, IbFft, TFdpBackend2d, TFdpBackend3d,
    },
};
use egraph::neighborhood::KnnGraph;
use egraph::spatial::IbFftConfig;
use wasm_bindgen::{JsCast, prelude::*};

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    core_err_to_js, invalid_input_err, parse_f64_sequence, parse_non_negative_usize,
    parse_u32_sequence, unsupported_err,
};
use crate::js_options::{
    JsTargetKind, check_options_arg, get_prop, is_plain_js_object, normalize_enum_token,
    opt_f64_prop, opt_seed_prop, opt_string_prop, opt_target_prop, opt_usize_prop,
    parse_ibfft_order, solver_status_str,
};
use crate::layout_buffers::{JsLayout2d, JsLayoutNd};
use crate::layout3d::JsLayout3d;
use crate::solver_runner::{
    run_layout_in_place, run_layout3d_in_place, run_nd_in_place, run_steps_layout_in_place,
    run_steps_layout3d_in_place, run_steps_nd_in_place, step_layout_in_place,
    step_layout3d_in_place, step_nd_in_place,
};

pub(crate) fn parse_ibfft_from_options(options: &JsValue) -> Result<IbFftConfig, JsValue> {
    let mut cfg = IbFftConfig::default();
    let ord_val = get_prop(options, "interpolationOrder")?;
    if !ord_val.is_undefined() && !ord_val.is_null() {
        cfg.order = parse_ibfft_order(&ord_val)?;
    }
    cfg.intervals = opt_usize_prop(options, "intervals", cfg.intervals)?;
    cfg.min_grid_dim = opt_usize_prop(options, "minGridDim", cfg.min_grid_dim)?;
    cfg.max_grid_dim = opt_usize_prop(options, "maxGridDim", cfg.max_grid_dim)?;
    Ok(cfg)
}

pub(crate) fn parse_t_fdp_config_with_ctx(
    options: &JsValue,
    ctx: &str,
) -> Result<TFdpConfig, JsValue> {
    let mut cfg = TFdpConfig::default();
    if check_options_arg(options, ctx)? {
        cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
        cfg.learning_rate = opt_f64_prop(options, "learningRate", cfg.learning_rate)?;
        cfg.momentum = opt_f64_prop(options, "momentum", cfg.momentum)?;
        cfg.alpha = opt_f64_prop(options, "alpha", cfg.alpha)?;
        cfg.gamma = opt_f64_prop(options, "gamma", cfg.gamma)?;
        cfg.rho.start_weight = opt_f64_prop(options, "rhoStart", cfg.rho.start_weight)?;
        cfg.rho.end_weight = opt_f64_prop(options, "rhoEnd", cfg.rho.end_weight)?;
        cfg.rho.power = opt_f64_prop(options, "rhoPower", cfg.rho.power)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    Ok(cfg)
}

#[derive(Debug, Clone)]
enum TFdpInner {
    Dim2(TFdpSolver<Layout2d<f32>, TFdpBackend2d>),
    Dim3(TFdpSolver<Layout3d<f32>, TFdpBackend3d>),
    DimN(TFdpSolver<LayoutNd<f32>, NeighborExact>),
}

enum TFdpBackendChoice {
    Dim2(TFdpBackend2d),
    Dim3(TFdpBackend3d),
    DimN(NeighborExact),
}

fn parse_t_fdp_target_backend(
    options: &JsValue,
) -> Result<(JsTargetKind, TFdpBackendChoice), JsValue> {
    let has_opts = check_options_arg(options, "TFdpSolver")?;
    let target = if has_opts {
        opt_target_prop(options)?
    } else {
        JsTargetKind::Layout2d
    };
    let backend_opt = if has_opts {
        opt_string_prop(options, "backend")?
    } else {
        None
    };
    let theta = if has_opts {
        opt_f64_prop(options, "theta", NeighborBarnesHut::default().theta)?
    } else {
        NeighborBarnesHut::default().theta
    };
    let ibfft_cfg = if has_opts {
        parse_ibfft_from_options(options)?
    } else {
        IbFftConfig::default()
    };

    let choice = match (target, backend_opt) {
        (JsTargetKind::Manifold2d, _) => {
            return Err(unsupported_err(
                "TFdpSolver does not support target 'manifold2d'",
            ));
        }
        (JsTargetKind::Layout2d, None) => {
            TFdpBackendChoice::Dim2(TFdpBackend2d::IbFft(IbFft::new().with_ibfft(ibfft_cfg)))
        }
        (JsTargetKind::Layout2d, Some(ref b)) => match normalize_enum_token(b)?.as_str() {
            "ibfft" | "fft" => {
                TFdpBackendChoice::Dim2(TFdpBackend2d::IbFft(IbFft::new().with_ibfft(ibfft_cfg)))
            }
            "barneshut" | "bh" => {
                TFdpBackendChoice::Dim2(TFdpBackend2d::BarnesHut(NeighborBarnesHut::new(theta)))
            }
            "exact" => TFdpBackendChoice::Dim2(TFdpBackend2d::Exact(NeighborExact::new())),
            _ => {
                return Err(invalid_input_err(format!(
                    "unknown TFdpSolver backend {b:?}; expected 'ibFft', 'barnesHut', or 'exact'"
                )));
            }
        },
        (JsTargetKind::Layout3d, None) => {
            TFdpBackendChoice::Dim3(TFdpBackend3d::BarnesHut(NeighborBarnesHut::new(theta)))
        }
        (JsTargetKind::Layout3d, Some(ref b)) => match normalize_enum_token(b)?.as_str() {
            "barneshut" | "bh" => {
                TFdpBackendChoice::Dim3(TFdpBackend3d::BarnesHut(NeighborBarnesHut::new(theta)))
            }
            "exact" => TFdpBackendChoice::Dim3(TFdpBackend3d::Exact(NeighborExact::new())),
            "ibfft" | "fft" => {
                return Err(unsupported_err(
                    "TFdpSolver target 'layout3d' does not support 2D FFT backend 'ibFft'",
                ));
            }
            _ => {
                return Err(invalid_input_err(format!(
                    "unknown TFdpSolver backend {b:?}; expected 'barnesHut' or 'exact'"
                )));
            }
        },
        (JsTargetKind::LayoutNd, None) => TFdpBackendChoice::DimN(NeighborExact::new()),
        (JsTargetKind::LayoutNd, Some(ref b)) => match normalize_enum_token(b)?.as_str() {
            "exact" => TFdpBackendChoice::DimN(NeighborExact::new()),
            "ibfft" | "fft" | "barneshut" | "bh" => {
                return Err(unsupported_err(format!(
                    "TFdpSolver target 'layoutNd' only supports backend 'exact', got {b:?}"
                )));
            }
            _ => {
                return Err(invalid_input_err(format!(
                    "unknown TFdpSolver backend {b:?}; expected 'exact'"
                )));
            }
        },
    };
    Ok((target, choice))
}

fn try_extract_knn_graph(val: &JsValue) -> Result<Option<KnnGraph>, JsValue> {
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
        invalid_input_err(format!("TFdpSolver.fromKnn: knn.indptr() failed: {err:?}"))
    })?;
    let raw_indices = indices_fn.call0(val).map_err(|err| {
        invalid_input_err(format!("TFdpSolver.fromKnn: knn.indices() failed: {err:?}"))
    })?;
    let raw_distances = distances_fn.call0(val).map_err(|err| {
        invalid_input_err(format!(
            "TFdpSolver.fromKnn: knn.distances() failed: {err:?}"
        ))
    })?;
    let indptr = parse_u32_sequence(&raw_indptr, "knn.indptr")?;
    let indices = parse_u32_sequence(&raw_indices, "knn.indices")?;
    let distances = parse_f64_sequence(&raw_distances, "knn.distances")?;
    let knn = KnnGraph::new_validated(n, indptr, indices, distances).map_err(core_err_to_js)?;
    Ok(Some(knn))
}

/// Cooperative bounded Student-$t$ force-directed placement (`t-FDP`) solver (`f32`)
/// supporting `ibFft`, `barnesHut`, and `exact` repulsion backends across 2D, 3D, and N-D.
#[wasm_bindgen(js_name = TFdpSolver)]
#[derive(Debug, Clone)]
pub struct JsTFdpSolver {
    inner: TFdpInner,
}

#[wasm_bindgen(js_class = TFdpSolver)]
impl JsTFdpSolver {
    /// Constructs a `TFdpSolver` directly from a `CsrGraph`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsTFdpSolver, JsValue> {
        let cfg = parse_t_fdp_config_with_ctx(options, "TFdpSolver")?;
        let (_, backend) = parse_t_fdp_target_backend(options)?;
        let inner = match backend {
            TFdpBackendChoice::Dim2(b) => TFdpInner::Dim2(
                TFdpSolver::<Layout2d<f32>, TFdpBackend2d>::new_with_backend(graph.inner(), cfg, b)
                    .map_err(core_err_to_js)?,
            ),
            TFdpBackendChoice::Dim3(b) => TFdpInner::Dim3(
                TFdpSolver::<Layout3d<f32>, TFdpBackend3d>::new_with_backend(graph.inner(), cfg, b)
                    .map_err(core_err_to_js)?,
            ),
            TFdpBackendChoice::DimN(b) => TFdpInner::DimN(
                TFdpSolver::<LayoutNd<f32>, NeighborExact>::new_with_backend(graph.inner(), cfg, b)
                    .map_err(core_err_to_js)?,
            ),
        };
        Ok(Self { inner })
    }

    /// Constructs a `TFdpSolver` from a `KnnGraph` (`fromKnn(knn, options?)`) or CSR slices
    /// (`fromKnn(indptr, indices, weights?, options?)`).
    #[wasm_bindgen(js_name = fromKnn)]
    pub fn from_knn(
        knn_or_indptr: &JsValue,
        indices_or_options: &JsValue,
        weights_or_options: &JsValue,
        options: &JsValue,
    ) -> Result<JsTFdpSolver, JsValue> {
        let (knn, opts_val) = if let Some(knn) = try_extract_knn_graph(knn_or_indptr)? {
            if (!weights_or_options.is_undefined() && !weights_or_options.is_null())
                || (!options.is_undefined() && !options.is_null())
            {
                return Err(invalid_input_err(
                    "TFdpSolver.fromKnn(knn, options?): unexpected extra positional arguments",
                ));
            }
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
                if !options.is_undefined() && !options.is_null() {
                    return Err(invalid_input_err(
                        "TFdpSolver.fromKnn: unexpected trailing argument after options object",
                    ));
                }
                (vec![0.0_f64; indices.len()], weights_or_options)
            } else if weights_or_options.is_undefined() || weights_or_options.is_null() {
                (vec![0.0_f64; indices.len()], options)
            } else {
                let ws = parse_f64_sequence(weights_or_options, "weights")?;
                (ws, options)
            };
            let knn =
                KnnGraph::new_validated(n, indptr, indices, weights_vec).map_err(core_err_to_js)?;
            (knn, opts_val)
        };

        let cfg = parse_t_fdp_config_with_ctx(opts_val, "TFdpSolver.fromKnn")?;
        let (_, backend) = parse_t_fdp_target_backend(opts_val)?;
        let inner = match backend {
            TFdpBackendChoice::Dim2(b) => TFdpInner::Dim2(
                TFdpSolver::<Layout2d<f32>, TFdpBackend2d>::from_knn_with_backend(&knn, cfg, b)
                    .map_err(core_err_to_js)?,
            ),
            TFdpBackendChoice::Dim3(b) => TFdpInner::Dim3(
                TFdpSolver::<Layout3d<f32>, TFdpBackend3d>::from_knn_with_backend(&knn, cfg, b)
                    .map_err(core_err_to_js)?,
            ),
            TFdpBackendChoice::DimN(b) => TFdpInner::DimN(
                TFdpSolver::<LayoutNd<f32>, NeighborExact>::from_knn_with_backend(&knn, cfg, b)
                    .map_err(core_err_to_js)?,
            ),
        };
        Ok(Self { inner })
    }

    #[wasm_bindgen(js_name = _step2d)]
    pub fn step_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            TFdpInner::Dim2(s) => step_layout_in_place(s, layout),
            _ => Err(unsupported_err(
                "TFdpSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps2d)]
    pub fn run_steps_2d(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            TFdpInner::Dim2(s) => {
                run_steps_layout_in_place(s.num_nodes(), s, layout, steps, "TFdpSolver::run_steps")
            }
            _ => Err(unsupported_err(
                "TFdpSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run2d)]
    pub fn run_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            TFdpInner::Dim2(s) => run_layout_in_place(s.num_nodes(), s, layout, "TFdpSolver::run"),
            _ => Err(unsupported_err(
                "TFdpSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _step3d)]
    pub fn step_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match &mut self.inner {
            TFdpInner::Dim3(s) => step_layout3d_in_place(s, layout),
            _ => Err(unsupported_err(
                "TFdpSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps3d)]
    pub fn run_steps_3d(&mut self, layout: &mut JsLayout3d, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            TFdpInner::Dim3(s) => run_steps_layout3d_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "TFdpSolver::run_steps_3d",
            ),
            _ => Err(unsupported_err(
                "TFdpSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run3d)]
    pub fn run_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match &mut self.inner {
            TFdpInner::Dim3(s) => {
                run_layout3d_in_place(s.num_nodes(), s, layout, "TFdpSolver::run_3d")
            }
            _ => Err(unsupported_err(
                "TFdpSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _stepNd)]
    pub fn step_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match &mut self.inner {
            TFdpInner::DimN(s) => step_nd_in_place(s, layout),
            _ => Err(unsupported_err(
                "TFdpSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runStepsNd)]
    pub fn run_steps_nd(&mut self, layout: &mut JsLayoutNd, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            TFdpInner::DimN(s) => {
                run_steps_nd_in_place(s.num_nodes(), s, layout, steps, "TFdpSolver::run_steps_nd")
            }
            _ => Err(unsupported_err(
                "TFdpSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runNd)]
    pub fn run_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match &mut self.inner {
            TFdpInner::DimN(s) => run_nd_in_place(s.num_nodes(), s, layout, "TFdpSolver::run_nd"),
            _ => Err(unsupported_err(
                "TFdpSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn target(&self) -> String {
        match &self.inner {
            TFdpInner::Dim2(_) => "layout2d",
            TFdpInner::Dim3(_) => "layout3d",
            TFdpInner::DimN(_) => "layoutNd",
        }
        .to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn backend(&self) -> String {
        match &self.inner {
            TFdpInner::Dim2(s) => match s.backend() {
                TFdpBackend2d::IbFft(_) => "ibFft",
                TFdpBackend2d::BarnesHut(_) => "barnesHut",
                TFdpBackend2d::Exact(_) => "exact",
                _ => "ibFft",
            },
            TFdpInner::Dim3(s) => match s.backend() {
                TFdpBackend3d::BarnesHut(_) => "barnesHut",
                TFdpBackend3d::Exact(_) => "exact",
                _ => "barnesHut",
            },
            TFdpInner::DimN(_) => "exact",
        }
        .to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        let st = match &self.inner {
            TFdpInner::Dim2(s) => s.status(),
            TFdpInner::Dim3(s) => s.status(),
            TFdpInner::DimN(s) => s.status(),
        };
        solver_status_str(st).to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        match &self.inner {
            TFdpInner::Dim2(s) => s.iteration(),
            TFdpInner::Dim3(s) => s.iteration(),
            TFdpInner::DimN(s) => s.iteration(),
        }
    }

    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> Option<usize> {
        match &self.inner {
            TFdpInner::Dim2(s) => s.total_iterations(),
            TFdpInner::Dim3(s) => s.total_iterations(),
            TFdpInner::DimN(s) => s.total_iterations(),
        }
    }

    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        let st = match &self.inner {
            TFdpInner::Dim2(s) => s.status(),
            TFdpInner::Dim3(s) => s.status(),
            TFdpInner::DimN(s) => s.status(),
        };
        st.is_finished()
    }

    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        match &self.inner {
            TFdpInner::Dim2(s) => s.num_nodes(),
            TFdpInner::Dim3(s) => s.num_nodes(),
            TFdpInner::DimN(s) => s.num_nodes(),
        }
    }

    #[wasm_bindgen(getter, js_name = currentRho)]
    pub fn current_rho(&self) -> f64 {
        match &self.inner {
            TFdpInner::Dim2(s) => s.current_rho(),
            TFdpInner::Dim3(s) => s.current_rho(),
            TFdpInner::DimN(s) => s.current_rho(),
        }
    }

    #[wasm_bindgen(getter, js_name = numEdges)]
    pub fn num_edges(&self) -> usize {
        match &self.inner {
            TFdpInner::Dim2(s) => s.num_edges(),
            TFdpInner::Dim3(s) => s.num_edges(),
            TFdpInner::DimN(s) => s.num_edges(),
        }
    }

    #[wasm_bindgen(getter, js_name = totalFft2dCalls)]
    pub fn total_fft_2d_calls(&self) -> f64 {
        match &self.inner {
            TFdpInner::Dim2(s) => s.total_fft_2d_calls() as f64,
            TFdpInner::Dim3(s) => s.total_fft_2d_calls() as f64,
            TFdpInner::DimN(s) => s.total_fft_2d_calls() as f64,
        }
    }
}
