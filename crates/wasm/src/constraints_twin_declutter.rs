//! WebAssembly bindings for 2D twin/near-twin decluttering (`TwinDeclutterer`,
//! `declutterTwins2d`).

use egraph::core::LayoutTransform;
use egraph::layout::{TwinDeclutterConfig, TwinDeclutterer, declutter_twins_2d};
use wasm_bindgen::prelude::*;

use crate::js_convert::{core_err_to_js, ffi_err_to_js};
use crate::js_options::{check_options_arg, opt_f64_prop, opt_usize_prop};
use crate::layout_buffers::JsLayout2d;

fn parse_twin_declutter_config(
    options: &JsValue,
    ctx: &str,
) -> Result<TwinDeclutterConfig, JsValue> {
    let mut cfg = TwinDeclutterConfig::default();
    if check_options_arg(options, ctx)? {
        cfg.eps_rel = opt_f64_prop(options, "epsRel", cfg.eps_rel)?;
        cfg.r_frac = opt_f64_prop(options, "rFrac", cfg.r_frac)?;
        cfg.r_cap_rel = opt_f64_prop(options, "rCapRel", cfg.r_cap_rel)?;
        cfg.min_pair_rel = opt_f64_prop(options, "minPairRel", cfg.min_pair_rel)?;
        cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
    }
    cfg.validate().map_err(core_err_to_js)?;
    Ok(cfg)
}

/// 2D layout transform resolving exact duplicate and near-twin node overlaps (`f32`).
#[wasm_bindgen(js_name = TwinDeclutterer)]
#[derive(Debug, Clone)]
pub struct JsTwinDeclutterer {
    inner: TwinDeclutterer<f32>,
}

#[wasm_bindgen(js_class = TwinDeclutterer)]
impl JsTwinDeclutterer {
    /// Constructs a `TwinDeclutterer` validated against `options`.
    #[wasm_bindgen(constructor)]
    pub fn new(options: &JsValue) -> Result<JsTwinDeclutterer, JsValue> {
        let cfg = parse_twin_declutter_config(options, "TwinDeclutterer")?;
        let inner = TwinDeclutterer::<f32>::new(cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Applies twin-decluttering in place on `layout`.
    pub fn apply(&mut self, layout: &mut JsLayout2d) -> Result<(), JsValue> {
        let mut soa = layout.inner().to_layout2d();
        self.inner.apply(&mut soa).map_err(core_err_to_js)?;
        layout
            .inner_mut()
            .copy_from_layout2d(&soa)
            .map_err(ffi_err_to_js)?;
        Ok(())
    }

    #[wasm_bindgen(getter, js_name = lastPassCount)]
    pub fn last_pass_count(&self) -> usize {
        self.inner.last_pass_count()
    }

    #[wasm_bindgen(getter, js_name = lastClusterCount)]
    pub fn last_cluster_count(&self) -> usize {
        self.inner.last_cluster_count()
    }

    #[wasm_bindgen(getter, js_name = lastMovedCount)]
    pub fn last_moved_count(&self) -> usize {
        self.inner.last_moved_count()
    }
}

/// Runs twin/near-twin decluttering in place on `layout`, returning the number of passes executed.
#[wasm_bindgen(js_name = declutterTwins2d)]
pub fn js_declutter_twins_2d(layout: &mut JsLayout2d, options: &JsValue) -> Result<usize, JsValue> {
    let cfg = parse_twin_declutter_config(options, "declutterTwins2d")?;
    let mut soa = layout.inner().to_layout2d();
    let passes = declutter_twins_2d(&mut soa, &cfg).map_err(core_err_to_js)?;
    layout
        .inner_mut()
        .copy_from_layout2d(&soa)
        .map_err(ffi_err_to_js)?;
    Ok(passes)
}
