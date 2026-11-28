//! Explicit-pair & packed symmetric `SgdSolver` constructors and schedule-window stepping
//! (`fromPairs`, `fromPairsWithHinges`, `fromPackedSymmetric`, `resetSchedule`, `runScheduleWindow`).

use egraph::core::{CoreError, Layout2d, Layout3d, LayoutNd, ManifoldLayout2d, RiemannianLayout};
use egraph::layout::{SgdPairBuffer, SgdScheduleConfig, SgdSolver};
use wasm_bindgen::{JsCast, prelude::*};

use crate::js_convert::{
    core_err_to_js, ffi_err_to_js, invalid_input_err, parse_non_negative_usize, unsupported_err,
};
use crate::js_options::{JsTargetKind, check_options_arg, get_prop, solver_status_str};
use crate::layout_buffers::{JsLayout2d, JsLayoutNd};
use crate::layout3d::JsLayout3d;
use crate::manifold::JsManifoldLayout2d;
use crate::sgd::{JsSgdSolver, SgdInner, parse_sgd_schedule_options};
use crate::solver_runner::check_and_lock_manifold;

fn default_inv_sq_weights_f32(dists: &[f32]) -> Vec<f32> {
    dists
        .iter()
        .map(|&d| {
            let df = f64::from(d);
            if df.is_finite() && df > 0.0 {
                (1.0 / (df * df)) as f32
            } else {
                0.0
            }
        })
        .collect()
}

fn opt_f32_array_prop(obj: &JsValue, key: &str) -> Result<Option<Vec<f32>>, JsValue> {
    let val = get_prop(obj, key)?;
    if val.is_undefined() || val.is_null() {
        return Ok(None);
    }
    if let Some(arr) = val.dyn_ref::<js_sys::Float32Array>() {
        return Ok(Some(arr.to_vec()));
    }
    if let Some(arr) = val.dyn_ref::<js_sys::Float64Array>() {
        return Ok(Some(arr.to_vec().into_iter().map(|v| v as f32).collect()));
    }
    if !val.is_string() && val.is_object() {
        let iter = js_sys::try_iter(&val)
            .map_err(|err| invalid_input_err(format!("failed to iterate options.{key}: {err:?}")))?
            .ok_or_else(|| {
                invalid_input_err(format!(
                    "options.{key} must be a Float32Array or number array"
                ))
            })?;
        let mut out = Vec::new();
        for item_res in iter {
            let item = item_res.map_err(|err| {
                invalid_input_err(format!("failed to read options.{key} entry: {err:?}"))
            })?;
            let num = item.as_f64().ok_or_else(|| {
                invalid_input_err(format!("options.{key} entries must be numbers"))
            })?;
            out.push(num as f32);
        }
        return Ok(Some(out));
    }
    Err(invalid_input_err(format!(
        "options.{key} must be a Float32Array or number array"
    )))
}

fn opt_u8_flag_array_prop(obj: &JsValue, key: &str) -> Result<Option<Vec<u8>>, JsValue> {
    let val = get_prop(obj, key)?;
    if val.is_undefined() || val.is_null() {
        return Ok(None);
    }
    if let Some(arr) = val.dyn_ref::<js_sys::Uint8Array>() {
        return Ok(Some(arr.to_vec()));
    }
    if !val.is_string() && val.is_object() {
        let iter = js_sys::try_iter(&val)
            .map_err(|err| invalid_input_err(format!("failed to iterate options.{key}: {err:?}")))?
            .ok_or_else(|| {
                invalid_input_err(format!("options.{key} must be a Uint8Array or array"))
            })?;
        let mut out = Vec::new();
        for item_res in iter {
            let item = item_res.map_err(|err| {
                invalid_input_err(format!("failed to read options.{key} entry: {err:?}"))
            })?;
            if let Some(b) = item.as_bool() {
                out.push(u8::from(b));
            } else if let Some(num) = item.as_f64() {
                out.push(u8::from(num != 0.0));
            } else {
                return Err(invalid_input_err(format!(
                    "options.{key} entries must be numbers or booleans"
                )));
            }
        }
        return Ok(Some(out));
    }
    Err(invalid_input_err(format!(
        "options.{key} must be a Uint8Array or array"
    )))
}

fn build_packed_buffer(
    src: &[u32],
    tgt: &[u32],
    dists: &[f32],
    weights: Option<&[f32]>,
    is_hinge: Option<&[u8]>,
) -> Result<SgdPairBuffer<f32>, JsValue> {
    if src.len() != tgt.len() || src.len() != dists.len() {
        return Err(core_err_to_js(CoreError::dimension_mismatch(
            src.len(),
            tgt.len().min(dists.len()),
            "SgdSolver.fromPackedSymmetric",
        )));
    }
    if let Some(ws) = weights
        && ws.len() != src.len()
    {
        return Err(core_err_to_js(CoreError::dimension_mismatch(
            src.len(),
            ws.len(),
            "SgdSolver.fromPackedSymmetric::weights",
        )));
    }
    if let Some(flags) = is_hinge
        && flags.len() != src.len()
    {
        return Err(core_err_to_js(CoreError::dimension_mismatch(
            src.len(),
            flags.len(),
            "SgdSolver.fromPackedSymmetric::isHinge",
        )));
    }

    let mut buf = SgdPairBuffer::<f32>::with_capacity(src.len());
    for idx in 0..src.len() {
        let d = dists[idx];
        let df = f64::from(d);
        if !df.is_finite() || df <= 0.0 {
            return Err(core_err_to_js(CoreError::invalid_input(
                "SgdSolver.fromPackedSymmetric: dists must be finite and > 0",
            )));
        }
        let w = match weights {
            Some(ws) => ws[idx],
            None => (1.0 / (df * df)) as f32,
        };
        let hinge = is_hinge.is_some_and(|flags| flags[idx] != 0);
        if hinge {
            buf.push_hinge(src[idx], tgt[idx], d, w)
                .map_err(core_err_to_js)?;
        } else {
            buf.push_spring(src[idx], tgt[idx], d, w)
                .map_err(core_err_to_js)?;
        }
    }
    Ok(buf)
}

#[allow(clippy::too_many_arguments)]
fn build_from_pairs_for_target<L: RiemannianLayout<Scalar = f32>>(
    n: usize,
    src: &[u32],
    tgt: &[u32],
    dij: &[f32],
    dji: &[f32],
    wij: &[f32],
    wji: &[f32],
    sched: SgdScheduleConfig,
    seed: u64,
) -> Result<SgdSolver<L>, JsValue>
where
    SgdSolver<L>: Sized,
{
    SgdSolver::<L>::from_pairs(n, src, tgt, dij, dji, wij, wji, sched, seed).map_err(core_err_to_js)
}

#[wasm_bindgen(js_class = SgdSolver)]
impl JsSgdSolver {
    /// Constructs an `SgdSolver` from explicit flat SoA two-sided spring arrays.
    ///
    /// Optional `options`: `{ dji?, wij?, wji?, iterations?, eps?, kind?, stepScale?, maxMu?, seed?, target? }`.
    #[wasm_bindgen(js_name = fromPairs)]
    pub fn from_pairs(
        num_nodes: f64,
        src: &[u32],
        tgt: &[u32],
        dij: &[f32],
        options: &JsValue,
    ) -> Result<JsSgdSolver, JsValue> {
        let n = parse_non_negative_usize(num_nodes, "numNodes")?;
        let (sched, seed_u64, target) = parse_sgd_schedule_options(options, "SgdSolver.fromPairs")?;
        let max_mu = sched.max_mu;
        let has_opts = check_options_arg(options, "SgdSolver.fromPairs")?;
        let dji = if has_opts {
            opt_f32_array_prop(options, "dji")?
        } else {
            None
        };
        let wij = if has_opts {
            opt_f32_array_prop(options, "wij")?
        } else {
            None
        };
        let wji = if has_opts {
            opt_f32_array_prop(options, "wji")?
        } else {
            None
        };
        let dji_vec = dji.unwrap_or_else(|| dij.to_vec());
        let wij_was_none = wij.is_none();
        let wij_vec = wij.unwrap_or_else(|| default_inv_sq_weights_f32(dij));
        let wji_vec = wji.unwrap_or_else(|| {
            if wij_was_none {
                default_inv_sq_weights_f32(&dji_vec)
            } else {
                wij_vec.clone()
            }
        });
        let inner = match target {
            JsTargetKind::Layout2d => SgdInner::Dim2(build_from_pairs_for_target::<Layout2d<f32>>(
                n, src, tgt, dij, &dji_vec, &wij_vec, &wji_vec, sched, seed_u64,
            )?),
            JsTargetKind::Layout3d => SgdInner::Dim3(build_from_pairs_for_target::<Layout3d<f32>>(
                n, src, tgt, dij, &dji_vec, &wij_vec, &wji_vec, sched, seed_u64,
            )?),
            JsTargetKind::LayoutNd => SgdInner::DimN(build_from_pairs_for_target::<LayoutNd<f32>>(
                n, src, tgt, dij, &dji_vec, &wij_vec, &wji_vec, sched, seed_u64,
            )?),
            JsTargetKind::Manifold2d => {
                SgdInner::Manifold(build_from_pairs_for_target::<ManifoldLayout2d<f32>>(
                    n, src, tgt, dij, &dji_vec, &wij_vec, &wji_vec, sched, seed_u64,
                )?)
            }
        };
        Ok(Self::from_schedule_and_inner(max_mu, inner))
    }

    /// Constructs an `SgdSolver` combining two-sided spring pairs and one-sided hinge pairs.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = fromPairsWithHinges)]
    pub fn from_pairs_with_hinges(
        num_nodes: f64,
        spring_src: &[u32],
        spring_tgt: &[u32],
        spring_dij: &[f32],
        hinge_src: &[u32],
        hinge_tgt: &[u32],
        hinge_dij: &[f32],
        options: &JsValue,
    ) -> Result<JsSgdSolver, JsValue> {
        let n = parse_non_negative_usize(num_nodes, "numNodes")?;
        let (sched, seed_u64, target) =
            parse_sgd_schedule_options(options, "SgdSolver.fromPairsWithHinges")?;
        let max_mu = sched.max_mu;
        let has_opts = check_options_arg(options, "SgdSolver.fromPairsWithHinges")?;
        let spring_dji = if has_opts {
            opt_f32_array_prop(options, "springDji")?
        } else {
            None
        };
        let spring_wij = if has_opts {
            opt_f32_array_prop(options, "springWij")?
        } else {
            None
        };
        let spring_wji = if has_opts {
            opt_f32_array_prop(options, "springWji")?
        } else {
            None
        };
        let hinge_dji = if has_opts {
            opt_f32_array_prop(options, "hingeDji")?
        } else {
            None
        };
        let hinge_wij = if has_opts {
            opt_f32_array_prop(options, "hingeWij")?
        } else {
            None
        };
        let hinge_wji = if has_opts {
            opt_f32_array_prop(options, "hingeWji")?
        } else {
            None
        };

        let s_dji = spring_dji.unwrap_or_else(|| spring_dij.to_vec());
        let s_wij_none = spring_wij.is_none();
        let s_wij = spring_wij.unwrap_or_else(|| default_inv_sq_weights_f32(spring_dij));
        let s_wji = spring_wji.unwrap_or_else(|| {
            if s_wij_none {
                default_inv_sq_weights_f32(&s_dji)
            } else {
                s_wij.clone()
            }
        });
        let h_dji = hinge_dji.unwrap_or_else(|| hinge_dij.to_vec());
        let h_wij_none = hinge_wij.is_none();
        let h_wij = hinge_wij.unwrap_or_else(|| default_inv_sq_weights_f32(hinge_dij));
        let h_wji = hinge_wji.unwrap_or_else(|| {
            if h_wij_none {
                default_inv_sq_weights_f32(&h_dji)
            } else {
                h_wij.clone()
            }
        });

        let inner = match target {
            JsTargetKind::Layout2d => SgdInner::Dim2(
                SgdSolver::<Layout2d<f32>>::from_pairs_with_hinges(
                    n, spring_src, spring_tgt, spring_dij, &s_dji, &s_wij, &s_wji, hinge_src,
                    hinge_tgt, hinge_dij, &h_dji, &h_wij, &h_wji, sched, seed_u64,
                )
                .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Layout3d => SgdInner::Dim3(
                SgdSolver::<Layout3d<f32>>::from_pairs_with_hinges(
                    n, spring_src, spring_tgt, spring_dij, &s_dji, &s_wij, &s_wji, hinge_src,
                    hinge_tgt, hinge_dij, &h_dji, &h_wij, &h_wji, sched, seed_u64,
                )
                .map_err(core_err_to_js)?,
            ),
            JsTargetKind::LayoutNd => SgdInner::DimN(
                SgdSolver::<LayoutNd<f32>>::from_pairs_with_hinges(
                    n, spring_src, spring_tgt, spring_dij, &s_dji, &s_wij, &s_wji, hinge_src,
                    hinge_tgt, hinge_dij, &h_dji, &h_wij, &h_wji, sched, seed_u64,
                )
                .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Manifold2d => SgdInner::Manifold(
                SgdSolver::<ManifoldLayout2d<f32>>::from_pairs_with_hinges(
                    n, spring_src, spring_tgt, spring_dij, &s_dji, &s_wij, &s_wji, hinge_src,
                    hinge_tgt, hinge_dij, &h_dji, &h_wij, &h_wji, sched, seed_u64,
                )
                .map_err(core_err_to_js)?,
            ),
        };
        Ok(Self::from_schedule_and_inner(max_mu, inner))
    }

    /// Constructs an `SgdSolver` directly from packed symmetric pair arrays
    /// `(numNodes, src, tgt, dists, options?)`.
    #[wasm_bindgen(js_name = fromPackedSymmetric)]
    pub fn from_packed_symmetric(
        num_nodes: f64,
        src: &[u32],
        tgt: &[u32],
        dists: &[f32],
        options: &JsValue,
    ) -> Result<JsSgdSolver, JsValue> {
        let n = parse_non_negative_usize(num_nodes, "numNodes")?;
        let (sched, seed_u64, target) =
            parse_sgd_schedule_options(options, "SgdSolver.fromPackedSymmetric")?;
        let max_mu = sched.max_mu;
        let has_opts = check_options_arg(options, "SgdSolver.fromPackedSymmetric")?;
        let weights = if has_opts {
            opt_f32_array_prop(options, "weights")?
        } else {
            None
        };
        let is_hinge = if has_opts {
            opt_u8_flag_array_prop(options, "isHinge")?
        } else {
            None
        };
        let buf = build_packed_buffer(src, tgt, dists, weights.as_deref(), is_hinge.as_deref())?;
        let inner = match target {
            JsTargetKind::Layout2d => SgdInner::Dim2(
                SgdSolver::<Layout2d<f32>>::from_pair_buffer(n, buf, sched, seed_u64)
                    .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Layout3d => SgdInner::Dim3(
                SgdSolver::<Layout3d<f32>>::from_pair_buffer(n, buf, sched, seed_u64)
                    .map_err(core_err_to_js)?,
            ),
            JsTargetKind::LayoutNd => SgdInner::DimN(
                SgdSolver::<LayoutNd<f32>>::from_pair_buffer(n, buf, sched, seed_u64)
                    .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Manifold2d => SgdInner::Manifold(
                SgdSolver::<ManifoldLayout2d<f32>>::from_pair_buffer(n, buf, sched, seed_u64)
                    .map_err(core_err_to_js)?,
            ),
        };
        Ok(Self::from_schedule_and_inner(max_mu, inner))
    }

    /// Resets the SGD learning-rate schedule and RNG without reallocating stored pair buffers.
    #[wasm_bindgen(js_name = resetSchedule)]
    pub fn reset_schedule(&mut self, options: &JsValue) -> Result<(), JsValue> {
        let (sched, seed_u64, _) = parse_sgd_schedule_options(options, "SgdSolver.resetSchedule")?;
        self.set_max_mu(sched.max_mu);
        self.inner_mut()
            .reset_schedule(sched, seed_u64)
            .map_err(core_err_to_js)
    }

    /// Executes `count` 2D Euclidean SGD iterations over schedule indices `startIter..startIter + count`.
    #[wasm_bindgen(js_name = _runScheduleWindow2d)]
    pub fn run_schedule_window_2d(
        &mut self,
        layout: &mut JsLayout2d,
        start_iter: f64,
        count: f64,
    ) -> Result<String, JsValue> {
        let start = parse_non_negative_usize(start_iter, "startIter")?;
        let cnt = parse_non_negative_usize(count, "count")?;
        match self.inner_mut() {
            SgdInner::Dim2(s) => {
                let mut soa = layout.inner().to_layout2d();
                let status = s
                    .run_schedule_window(&mut soa, start, cnt)
                    .map_err(core_err_to_js)?;
                layout
                    .inner_mut()
                    .copy_from_layout2d(&soa)
                    .map_err(ffi_err_to_js)?;
                Ok(solver_status_str(status).to_owned())
            }
            _ => Err(unsupported_err(
                "SgdSolver was not configured for target 'layout2d'",
            )),
        }
    }

    /// Executes `count` 3D Euclidean SGD iterations over schedule indices `startIter..startIter + count`.
    #[wasm_bindgen(js_name = _runScheduleWindow3d)]
    pub fn run_schedule_window_3d(
        &mut self,
        layout: &mut JsLayout3d,
        start_iter: f64,
        count: f64,
    ) -> Result<String, JsValue> {
        let start = parse_non_negative_usize(start_iter, "startIter")?;
        let cnt = parse_non_negative_usize(count, "count")?;
        match self.inner_mut() {
            SgdInner::Dim3(s) => {
                let mut soa = layout.inner().to_layout3d();
                let status = s
                    .run_schedule_window(&mut soa, start, cnt)
                    .map_err(core_err_to_js)?;
                layout
                    .inner_mut()
                    .copy_from_layout3d(&soa)
                    .map_err(ffi_err_to_js)?;
                Ok(solver_status_str(status).to_owned())
            }
            _ => Err(unsupported_err(
                "SgdSolver was not configured for target 'layout3d'",
            )),
        }
    }

    /// Executes `count` N-D Euclidean SGD iterations over schedule indices `startIter..startIter + count`.
    #[wasm_bindgen(js_name = _runScheduleWindowNd)]
    pub fn run_schedule_window_nd(
        &mut self,
        layout: &mut JsLayoutNd,
        start_iter: f64,
        count: f64,
    ) -> Result<String, JsValue> {
        let start = parse_non_negative_usize(start_iter, "startIter")?;
        let cnt = parse_non_negative_usize(count, "count")?;
        match self.inner_mut() {
            SgdInner::DimN(s) => {
                let status = s
                    .run_schedule_window(layout.inner_mut(), start, cnt)
                    .map_err(core_err_to_js)?;
                Ok(solver_status_str(status).to_owned())
            }
            _ => Err(unsupported_err(
                "SgdSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    /// Executes `count` Manifold SGD iterations over schedule indices `startIter..startIter + count`.
    #[wasm_bindgen(js_name = _runScheduleWindowManifold)]
    pub fn run_schedule_window_manifold(
        &mut self,
        layout: &mut JsManifoldLayout2d,
        start_iter: f64,
        count: f64,
    ) -> Result<String, JsValue> {
        let start = parse_non_negative_usize(start_iter, "startIter")?;
        let cnt = parse_non_negative_usize(count, "count")?;
        let expected_nodes = self.inner_mut().num_nodes();
        if !matches!(self.inner_mut(), SgdInner::Manifold(_)) {
            return Err(unsupported_err(
                "SgdSolver was not configured for target 'manifold2d'",
            ));
        }
        check_and_lock_manifold(
            self.locked_manifold_mut(),
            layout,
            expected_nodes,
            "SgdSolver::run_schedule_window_manifold",
        )?;
        match self.inner_mut() {
            SgdInner::Manifold(s) => {
                let status = s
                    .run_schedule_window(layout.inner_mut(), start, cnt)
                    .map_err(core_err_to_js)?;
                layout.sync_shadow()?;
                Ok(solver_status_str(status).to_owned())
            }
            _ => unreachable!(),
        }
    }
}
