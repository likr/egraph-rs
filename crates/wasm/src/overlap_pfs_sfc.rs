//! WebAssembly bindings for `PFS'` node overlap removal (`LayoutTransform`) and canonical
//! community-ordered Space-Filling Curve (`SFC`) layout generator (`LayoutGenerator`, `f32`).

use egraph::core::{CsrGraph, LayoutGenerator, LayoutTransform, NodeExtents2d};
use egraph::layout::{PfsPrimeConfig, PfsPrimeOverlapRemoval, SfcConfig, SfcCurveKind, SfcLayout};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, ffi_err_to_js, invalid_input_err, parse_non_negative_u32};
use crate::js_options::{
    check_options_arg, get_prop, normalize_enum_token, opt_f64_prop, opt_seed_prop, opt_string_prop,
};
use crate::layout_buffers::JsLayout2d;

fn opt_maybe_u32_prop(obj: &JsValue, key: &str) -> Result<Option<u32>, JsValue> {
    let val = get_prop(obj, key)?;
    if val.is_undefined() || val.is_null() {
        return Ok(None);
    }
    let num = val
        .as_f64()
        .ok_or_else(|| invalid_input_err(format!("options.{key} must be an integer number")))?;
    parse_non_negative_u32(num, key).map(Some)
}

fn parse_pfs_prime_config(options: &JsValue, ctx: &str) -> Result<PfsPrimeConfig, JsValue> {
    let mut cfg = PfsPrimeConfig::default();
    if check_options_arg(options, ctx)? {
        cfg.padding = opt_f64_prop(options, "padding", cfg.padding)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    Ok(cfg)
}

/// Hayashi et al. (1998) `PFS'` two-pass scan-line node overlap removal transform (`f32`).
#[wasm_bindgen(js_name = PfsPrimeOverlapRemoval)]
#[derive(Debug, Clone)]
pub struct JsPfsPrimeOverlapRemoval {
    inner: PfsPrimeOverlapRemoval<f32>,
}

#[wasm_bindgen(js_class = PfsPrimeOverlapRemoval)]
impl JsPfsPrimeOverlapRemoval {
    /// Constructs a `PfsPrimeOverlapRemoval` transform from per-node `halfWidths` and `halfHeights`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        half_widths: &[f32],
        half_heights: &[f32],
        options: &JsValue,
    ) -> Result<JsPfsPrimeOverlapRemoval, JsValue> {
        let cfg = parse_pfs_prime_config(options, "PfsPrimeOverlapRemoval")?;
        let extents = NodeExtents2d::new(half_widths, half_heights).map_err(core_err_to_js)?;
        let inner = PfsPrimeOverlapRemoval::<f32>::new(extents, cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `PfsPrimeOverlapRemoval` transform with uniform square `radii`.
    #[wasm_bindgen(js_name = fromRadii)]
    pub fn from_radii(
        radii: &[f32],
        options: &JsValue,
    ) -> Result<JsPfsPrimeOverlapRemoval, JsValue> {
        let cfg = parse_pfs_prime_config(options, "PfsPrimeOverlapRemoval.fromRadii")?;
        let inner =
            PfsPrimeOverlapRemoval::<f32>::from_radii(radii, cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Applies the single two-pass `PFS'` overlap removal transform on `layout` in place.
    pub fn apply(&mut self, layout: &mut JsLayout2d) -> Result<(), JsValue> {
        let mut soa = layout.inner().to_layout2d();
        self.inner.apply(&mut soa).map_err(core_err_to_js)?;
        layout
            .inner_mut()
            .copy_from_layout2d(&soa)
            .map_err(ffi_err_to_js)?;
        Ok(())
    }

    /// Returns the number of overlapping rectangle pairs remaining after `apply`.
    #[wasm_bindgen(getter, js_name = lastOverlapCount)]
    pub fn last_overlap_count(&self) -> usize {
        self.inner.last_overlap_count()
    }

    /// Returns the expected vertex count.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }
}

fn parse_sfc_curve_kind(raw: &str) -> Result<SfcCurveKind, JsValue> {
    match normalize_enum_token(raw)?.as_str() {
        "hilbert" => Ok(SfcCurveKind::Hilbert),
        "morton" | "zorder" => Ok(SfcCurveKind::Morton),
        _ => Err(invalid_input_err(format!(
            "unknown SfcCurveKind {raw:?}; expected 'hilbert' or 'morton'"
        ))),
    }
}

/// Canonical community-ordered Space-Filling Curve (`SFC`) layout generator (`f32`).
#[wasm_bindgen(js_name = SfcLayout)]
#[derive(Debug, Clone)]
pub struct JsSfcLayout {
    graph: CsrGraph<f32>,
    config: SfcConfig,
    vertex_order: Vec<u32>,
}

#[wasm_bindgen(js_class = SfcLayout)]
impl JsSfcLayout {
    /// Constructs an `SfcLayout` generator from `graph` and optional `SfcOptions`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsSfcLayout, JsValue> {
        let mut cfg = SfcConfig::default();
        if check_options_arg(options, "SfcLayout")? {
            if let Some(c_str) = opt_string_prop(options, "curve")? {
                cfg.curve = parse_sfc_curve_kind(&c_str)?;
            }
            cfg.scale = opt_f64_prop(options, "scale", cfg.scale)?;
            cfg.resolution_order = opt_maybe_u32_prop(options, "resolutionOrder")?;
            cfg.seed = opt_seed_prop(options)?;
        }
        cfg.validate().map_err(core_err_to_js)?;
        Ok(Self {
            graph: graph.inner().clone(),
            config: cfg,
            vertex_order: Vec::new(),
        })
    }

    /// Generates a community-ordered Hilbert/Morton `Layout2d` (`f32`).
    pub fn generate(&mut self) -> Result<JsLayout2d, JsValue> {
        let mut generator = SfcLayout::<f32, f32>::from_config(&self.graph, self.config.clone());
        let layout = generator.generate().map_err(core_err_to_js)?;
        self.vertex_order = generator.vertex_order().to_vec();
        Ok(JsLayout2d::from_layout2d(&layout))
    }

    /// Returns the deterministic community-ordered vertex permutation (`Uint32Array`).
    #[wasm_bindgen(js_name = vertexOrder)]
    pub fn vertex_order(&self) -> Vec<u32> {
        self.vertex_order.clone()
    }

    /// Returns the graph vertex count.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.graph.num_nodes()
    }
}
