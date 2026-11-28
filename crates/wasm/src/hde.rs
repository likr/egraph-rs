//! WebAssembly bindings for Harel-Koren High-Dimensional Embedding (`HDE`, `f32`).

use egraph::core::{Layout2d, LayoutGenerator, LayoutNd};
use egraph::linalg::{Hde, HdeConfig};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, parse_u32_sequence};
use crate::js_options::{
    check_options_arg, get_prop, opt_bool_prop, opt_f64_prop, opt_maybe_usize_prop, opt_seed_prop,
    opt_usize_prop,
};
use crate::layout_buffers::{JsLayout2d, JsLayoutNd};

pub(crate) fn parse_hde_config(
    options: &JsValue,
    is_nd: bool,
    ctx: &str,
) -> Result<(HdeConfig, Option<Vec<u32>>), JsValue> {
    let mut cfg = HdeConfig::default();
    let mut pivots = None;
    if check_options_arg(options, ctx)? {
        cfg.dimensions = opt_usize_prop(options, "dimensions", cfg.dimensions)?;
        if is_nd {
            cfg.output_dims = opt_maybe_usize_prop(options, "outputDims")?;
        }
        cfg.seed = opt_seed_prop(options)?;
        cfg.tol = opt_f64_prop(options, "tol", cfg.tol)?;
        cfg.max_sweeps = opt_usize_prop(options, "maxSweeps", cfg.max_sweeps)?;
        cfg.normalize = opt_bool_prop(options, "normalize", cfg.normalize)?;
        let piv_val = get_prop(options, "pivots")?;
        if !piv_val.is_undefined() && !piv_val.is_null() {
            pivots = Some(parse_u32_sequence(&piv_val, "pivots")?);
        }
    }
    Ok((cfg, pivots))
}

/// Computes a 2D Harel-Koren High-Dimensional Embedding (`HDE`) `Layout2d` (`f32`).
#[wasm_bindgen(js_name = hde)]
pub fn js_hde(graph: &JsCsrGraph, options: &JsValue) -> Result<JsLayout2d, JsValue> {
    let (cfg, pivots) = parse_hde_config(options, false, "hde")?;
    let mut generator = Hde::<Layout2d<f32>, f32>::from_config(graph.inner(), cfg);
    if let Some(ref pivs) = pivots {
        generator = generator.with_pivots(pivs);
    }
    let layout = generator.generate().map_err(core_err_to_js)?;
    Ok(JsLayout2d::from_layout2d(&layout))
}

/// Computes an $N$-dimensional Harel-Koren High-Dimensional Embedding (`HDE`) `LayoutNd` (`f32`).
#[wasm_bindgen(js_name = hdeNd)]
pub fn js_hde_nd(graph: &JsCsrGraph, options: &JsValue) -> Result<JsLayoutNd, JsValue> {
    let (cfg, pivots) = parse_hde_config(options, true, "hdeNd")?;
    let mut generator = Hde::<LayoutNd<f32>, f32>::from_config(graph.inner(), cfg);
    if let Some(ref pivs) = pivots {
        generator = generator.with_pivots(pivs);
    }
    let layout = generator.generate().map_err(core_err_to_js)?;
    Ok(JsLayoutNd::from_inner(layout))
}
