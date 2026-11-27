//! Packed symmetric `SgdSolver` constructors and schedule-window stepping (`fromPackedSymmetric`,
//! `resetSchedule`, `runScheduleWindow`).

use egraph::core::CoreError;
use egraph::layout::{SgdSolver, sgd::PackedSymPair};
use wasm_bindgen::prelude::*;

use crate::js_convert::{core_err_to_js, ffi_err_to_js, parse_non_negative_usize};
use crate::js_options::solver_status_str;
use crate::layout_buffers::JsLayout2d;
use crate::manifold::JsManifoldLayout2d;
use crate::sgd::{JsSgdSolver, parse_sgd_schedule_and_seed};

#[wasm_bindgen(js_class = SgdSolver)]
impl JsSgdSolver {
    /// Constructs an `SgdSolver` directly from packed symmetric pair arrays
    /// `(src, tgt, dists, weights?, isHinge?, options?)`.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = fromPackedSymmetric)]
    pub fn from_packed_symmetric(
        num_nodes: f64,
        src: &[u32],
        tgt: &[u32],
        dists: &[f32],
        weights: Option<Vec<f32>>,
        is_hinge: Option<Vec<u8>>,
        iterations_or_options: &JsValue,
        eps: Option<f64>,
        kind: Option<String>,
        step_scale: Option<f64>,
        seed: Option<f64>,
    ) -> Result<JsSgdSolver, JsError> {
        let n = parse_non_negative_usize(num_nodes, "numNodes")?;
        if src.len() != tgt.len() || src.len() != dists.len() {
            return Err(core_err_to_js(CoreError::dimension_mismatch(
                src.len(),
                tgt.len().min(dists.len()),
                "SgdSolver.fromPackedSymmetric",
            )));
        }
        if let Some(ref ws) = weights
            && ws.len() != src.len()
        {
            return Err(core_err_to_js(CoreError::dimension_mismatch(
                src.len(),
                ws.len(),
                "SgdSolver.fromPackedSymmetric::weights",
            )));
        }
        if let Some(ref flags) = is_hinge
            && flags.len() != src.len()
        {
            return Err(core_err_to_js(CoreError::dimension_mismatch(
                src.len(),
                flags.len(),
                "SgdSolver.fromPackedSymmetric::isHinge",
            )));
        }
        let (sched, seed_u64) =
            parse_sgd_schedule_and_seed(iterations_or_options, eps, kind, step_scale, seed)?;
        let mut pairs = Vec::with_capacity(src.len());
        let mut num_springs = 0usize;
        let mut num_hinges = 0usize;
        for idx in 0..src.len() {
            let d = dists[idx];
            let df = f64::from(d);
            if !df.is_finite() || df <= 0.0 {
                return Err(core_err_to_js(CoreError::invalid_input(
                    "SgdSolver.fromPackedSymmetric: dists must be finite and > 0",
                )));
            }
            let w = match weights.as_ref() {
                Some(ws) => ws[idx],
                None => (1.0 / (df * df)) as f32,
            };
            let hinge = is_hinge.as_ref().is_some_and(|flags| flags[idx] != 0);
            if hinge {
                num_hinges += 1;
            } else {
                num_springs += 1;
            }
            pairs.push(PackedSymPair::new(src[idx], tgt[idx], hinge, d, w));
        }
        let inner = SgdSolver::<f32>::from_packed_symmetric(
            n,
            pairs,
            num_springs,
            num_hinges,
            sched,
            seed_u64,
        )
        .map_err(core_err_to_js)?;
        Ok(Self::from_schedule_and_inner(sched.max_mu, inner))
    }

    /// Resets the SGD learning-rate schedule and RNG without reallocating stored pair buffers.
    #[wasm_bindgen(js_name = resetSchedule)]
    pub fn reset_schedule(
        &mut self,
        iterations_or_options: &JsValue,
        eps: Option<f64>,
        kind: Option<String>,
        step_scale: Option<f64>,
        seed: Option<f64>,
    ) -> Result<(), JsError> {
        let (sched, seed_u64) =
            parse_sgd_schedule_and_seed(iterations_or_options, eps, kind, step_scale, seed)?;
        self.set_max_mu(sched.max_mu);
        self.inner_mut()
            .reset_schedule(sched, seed_u64)
            .map_err(core_err_to_js)
    }

    /// Executes `count` Euclidean SGD iterations over schedule indices `startIter..startIter + count`.
    #[wasm_bindgen(js_name = _runScheduleWindow2d)]
    pub fn run_schedule_window_2d(
        &mut self,
        layout: &mut JsLayout2d,
        start_iter: f64,
        count: f64,
    ) -> Result<String, JsError> {
        let start = parse_non_negative_usize(start_iter, "startIter")?;
        let cnt = parse_non_negative_usize(count, "count")?;
        let mut soa = layout.inner().to_layout2d();
        let status = self
            .inner_mut()
            .run_schedule_window(&mut soa, start, cnt)
            .map_err(core_err_to_js)?;
        layout
            .inner_mut()
            .copy_from_layout2d(&soa)
            .map_err(ffi_err_to_js)?;
        Ok(solver_status_str(status).to_owned())
    }

    /// Executes `count` Manifold SGD iterations over schedule indices `startIter..startIter + count`.
    #[wasm_bindgen(js_name = _runScheduleWindowManifold)]
    pub fn run_schedule_window_manifold(
        &mut self,
        layout: &mut JsManifoldLayout2d,
        start_iter: f64,
        count: f64,
    ) -> Result<String, JsError> {
        let start = parse_non_negative_usize(start_iter, "startIter")?;
        let cnt = parse_non_negative_usize(count, "count")?;
        let status = self
            .inner_mut()
            .run_schedule_window_manifold(layout.inner_mut(), start, cnt)
            .map_err(core_err_to_js)?;
        layout.sync_shadow()?;
        Ok(solver_status_str(status).to_owned())
    }
}
