//! Cooperative Force-Directed Edge Bundling (`FDEB`) WebAssembly bindings (`f32`)
//! operating in place on `EdgePolylines` across `Layout2d`, `Layout3d`, and `LayoutNd`.

use egraph::core::{IterativeSolver, Layout2d, Layout3d, LayoutNd};
use egraph::edge_routing::{FdebConfig, FdebSolver};
use wasm_bindgen::prelude::*;

use crate::edge_polylines::{EdgePolylinesInner, JsEdgePolylines};
use crate::js_convert::{core_err_to_js, ffi_err_to_js, parse_non_negative_usize, unsupported_err};
use crate::js_options::{check_options_arg, opt_f64_prop, opt_usize_prop, solver_status_str};

#[derive(Debug, Clone)]
enum FdebInner {
    Dim2(FdebSolver<Layout2d<f32>>),
    Dim3(FdebSolver<Layout3d<f32>>),
    DimN(FdebSolver<LayoutNd<f32>>),
}

fn parse_fdeb_config(options: &JsValue, ctx: &str) -> Result<FdebConfig, JsValue> {
    let mut cfg = FdebConfig::default();
    if check_options_arg(options, ctx)? {
        cfg.cycles = opt_usize_prop(options, "cycles", cfg.cycles)?;
        cfg.s0 = opt_f64_prop(options, "s0", cfg.s0)?;
        cfg.steps_per_cycle = opt_usize_prop(options, "stepsPerCycle", cfg.steps_per_cycle)?;
        cfg.s_step = opt_f64_prop(options, "sStep", cfg.s_step)?;
        cfg.i_step = opt_f64_prop(options, "iStep", cfg.i_step)?;
        cfg.spring_k = opt_f64_prop(options, "springK", cfg.spring_k)?;
        cfg.minimum_edge_compatibility = opt_f64_prop(
            options,
            "minimumEdgeCompatibility",
            cfg.minimum_edge_compatibility,
        )?;
    }
    Ok(cfg)
}

/// Cooperative Force-Directed Edge Bundling (`FDEB`) solver (`f32`) mutating `EdgePolylines`
/// in place across `"layout2d"`, `"layout3d"`, and `"layoutNd"`.
#[wasm_bindgen(js_name = FdebSolver)]
#[derive(Debug, Clone)]
pub struct JsFdebSolver {
    inner: FdebInner,
}

#[wasm_bindgen(js_class = FdebSolver)]
impl JsFdebSolver {
    /// Constructs an `FdebSolver` bound to the geometry/target of `polylines`.
    #[wasm_bindgen(constructor)]
    pub fn new(polylines: &JsEdgePolylines, options: &JsValue) -> Result<JsFdebSolver, JsValue> {
        let cfg = parse_fdeb_config(options, "FdebSolver")?;
        let inner = match polylines.inner() {
            EdgePolylinesInner::Dim2(p2) => {
                let core = p2.to_core().map_err(ffi_err_to_js)?;
                FdebInner::Dim2(
                    FdebSolver::<Layout2d<f32>>::new(&core, cfg).map_err(core_err_to_js)?,
                )
            }
            EdgePolylinesInner::Dim3(p3) => {
                let core = p3.to_core().map_err(ffi_err_to_js)?;
                FdebInner::Dim3(
                    FdebSolver::<Layout3d<f32>>::new(&core, cfg).map_err(core_err_to_js)?,
                )
            }
            EdgePolylinesInner::DimN(pn) => {
                FdebInner::DimN(FdebSolver::<LayoutNd<f32>>::new(pn, cfg).map_err(core_err_to_js)?)
            }
        };
        Ok(Self { inner })
    }

    /// Executes one FDEB bundling iteration in place on `polylines`.
    pub fn step(&mut self, polylines: &mut JsEdgePolylines) -> Result<String, JsValue> {
        let status = match (&mut self.inner, polylines.inner_mut()) {
            (FdebInner::Dim2(solver), EdgePolylinesInner::Dim2(p2)) => {
                let mut core = p2.to_core().map_err(ffi_err_to_js)?;
                let st = solver.step(&mut core).map_err(core_err_to_js)?;
                p2.replace_from_core(&core);
                st
            }
            (FdebInner::Dim3(solver), EdgePolylinesInner::Dim3(p3)) => {
                let mut core = p3.to_core().map_err(ffi_err_to_js)?;
                let st = solver.step(&mut core).map_err(core_err_to_js)?;
                p3.replace_from_core(&core);
                st
            }
            (FdebInner::DimN(solver), EdgePolylinesInner::DimN(pn)) => {
                solver.step(pn).map_err(core_err_to_js)?
            }
            _ => {
                return Err(unsupported_err(
                    "FdebSolver target mismatch with EdgePolylines buffer",
                ));
            }
        };
        Ok(solver_status_str(status).to_owned())
    }

    /// Executes up to `steps` FDEB bundling iterations in place on `polylines`.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(
        &mut self,
        polylines: &mut JsEdgePolylines,
        steps: f64,
    ) -> Result<String, JsValue> {
        let n_steps = parse_non_negative_usize(steps, "steps")?;
        let status = match (&mut self.inner, polylines.inner_mut()) {
            (FdebInner::Dim2(solver), EdgePolylinesInner::Dim2(p2)) => {
                let mut core = p2.to_core().map_err(ffi_err_to_js)?;
                let st = solver
                    .run_steps(&mut core, n_steps)
                    .map_err(core_err_to_js)?;
                p2.replace_from_core(&core);
                st
            }
            (FdebInner::Dim3(solver), EdgePolylinesInner::Dim3(p3)) => {
                let mut core = p3.to_core().map_err(ffi_err_to_js)?;
                let st = solver
                    .run_steps(&mut core, n_steps)
                    .map_err(core_err_to_js)?;
                p3.replace_from_core(&core);
                st
            }
            (FdebInner::DimN(solver), EdgePolylinesInner::DimN(pn)) => {
                solver.run_steps(pn, n_steps).map_err(core_err_to_js)?
            }
            _ => {
                return Err(unsupported_err(
                    "FdebSolver target mismatch with EdgePolylines buffer",
                ));
            }
        };
        Ok(solver_status_str(status).to_owned())
    }

    /// Runs FDEB bundling to completion in place on `polylines`.
    pub fn run(&mut self, polylines: &mut JsEdgePolylines) -> Result<String, JsValue> {
        let status = match (&mut self.inner, polylines.inner_mut()) {
            (FdebInner::Dim2(solver), EdgePolylinesInner::Dim2(p2)) => {
                let mut core = p2.to_core().map_err(ffi_err_to_js)?;
                let st = solver.run(&mut core).map_err(core_err_to_js)?;
                p2.replace_from_core(&core);
                st
            }
            (FdebInner::Dim3(solver), EdgePolylinesInner::Dim3(p3)) => {
                let mut core = p3.to_core().map_err(ffi_err_to_js)?;
                let st = solver.run(&mut core).map_err(core_err_to_js)?;
                p3.replace_from_core(&core);
                st
            }
            (FdebInner::DimN(solver), EdgePolylinesInner::DimN(pn)) => {
                solver.run(pn).map_err(core_err_to_js)?
            }
            _ => {
                return Err(unsupported_err(
                    "FdebSolver target mismatch with EdgePolylines buffer",
                ));
            }
        };
        Ok(solver_status_str(status).to_owned())
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        let st = match &self.inner {
            FdebInner::Dim2(s) => s.status(),
            FdebInner::Dim3(s) => s.status(),
            FdebInner::DimN(s) => s.status(),
        };
        solver_status_str(st).to_owned()
    }

    /// Returns the number of iterations executed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        match &self.inner {
            FdebInner::Dim2(s) => s.iteration(),
            FdebInner::Dim3(s) => s.iteration(),
            FdebInner::DimN(s) => s.iteration(),
        }
    }

    /// Returns the total scheduled iterations across all subdivision cycles.
    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> Option<usize> {
        match &self.inner {
            FdebInner::Dim2(s) => s.total_iterations(),
            FdebInner::Dim3(s) => s.total_iterations(),
            FdebInner::DimN(s) => s.total_iterations(),
        }
    }

    /// Returns `true` when edge bundling has converged or completed.
    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        match &self.inner {
            FdebInner::Dim2(s) => s.is_finished(),
            FdebInner::Dim3(s) => s.is_finished(),
            FdebInner::DimN(s) => s.is_finished(),
        }
    }

    /// Returns the number of edges being bundled.
    #[wasm_bindgen(getter, js_name = numEdges)]
    pub fn num_edges(&self) -> usize {
        match &self.inner {
            FdebInner::Dim2(s) => s.num_edges(),
            FdebInner::Dim3(s) => s.num_edges(),
            FdebInner::DimN(s) => s.num_edges(),
        }
    }

    /// Returns the number of compatible edge pairs interacting during bundling.
    #[wasm_bindgen(getter, js_name = numCompatiblePairs)]
    pub fn num_compatible_pairs(&self) -> usize {
        match &self.inner {
            FdebInner::Dim2(s) => s.num_compatible_pairs(),
            FdebInner::Dim3(s) => s.num_compatible_pairs(),
            FdebInner::DimN(s) => s.num_compatible_pairs(),
        }
    }

    /// Returns the current number of control points per edge.
    #[wasm_bindgen(getter, js_name = ptsPerEdge)]
    pub fn pts_per_edge(&self) -> usize {
        match &self.inner {
            FdebInner::Dim2(s) => s.pts_per_edge(),
            FdebInner::Dim3(s) => s.pts_per_edge(),
            FdebInner::DimN(s) => s.pts_per_edge(),
        }
    }

    /// Returns the current subdivision cycle index.
    #[wasm_bindgen(getter, js_name = currentCycle)]
    pub fn current_cycle(&self) -> usize {
        match &self.inner {
            FdebInner::Dim2(s) => s.current_cycle(),
            FdebInner::Dim3(s) => s.current_cycle(),
            FdebInner::DimN(s) => s.current_cycle(),
        }
    }

    /// Returns the coordinate dimensionality (`2`, `3`, or `LayoutNd.dims`).
    #[wasm_bindgen(getter)]
    pub fn dims(&self) -> usize {
        match &self.inner {
            FdebInner::Dim2(s) => s.dims(),
            FdebInner::Dim3(s) => s.dims(),
            FdebInner::DimN(s) => s.dims(),
        }
    }

    /// Returns the bound target storage kind (`"layout2d"`, `"layout3d"`, or `"layoutNd"`).
    #[wasm_bindgen(getter)]
    pub fn target(&self) -> String {
        match &self.inner {
            FdebInner::Dim2(_) => "layout2d".to_owned(),
            FdebInner::Dim3(_) => "layout3d".to_owned(),
            FdebInner::DimN(_) => "layoutNd".to_owned(),
        }
    }
}
