//! WebAssembly bindings for Leiden community detection (`Modularity` and `CPM`).

use egraph::clustering::{LeidenConfig, LeidenObjective, LeidenResult, leiden};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, invalid_input_err, set_f64_prop, set_js_prop};
use crate::js_options::{
    check_options_arg, normalize_enum_token, opt_f64_prop, opt_seed_prop, opt_string_prop,
    opt_usize_prop,
};

fn parse_leiden_objective(raw: &str) -> Result<LeidenObjective, JsValue> {
    match normalize_enum_token(raw)?.as_str() {
        "modularity" | "mod" => Ok(LeidenObjective::Modularity),
        "cpm" | "constantpottsmodel" => Ok(LeidenObjective::Cpm),
        _ => Err(invalid_input_err(format!(
            "unknown LeidenObjective {raw:?}; expected 'modularity' or 'cpm'"
        ))),
    }
}

/// Community detection output returned by `leidenCommunities`.
#[wasm_bindgen(js_name = LeidenResult)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsLeidenResult {
    inner: LeidenResult,
}

#[wasm_bindgen(js_class = LeidenResult)]
impl JsLeidenResult {
    /// Returns the community ID per vertex (`Uint32Array`).
    pub fn labels(&self) -> Vec<u32> {
        self.inner.labels.clone()
    }

    /// Returns the linear-memory byte offset of `labels` for zero-copy `labelsView()`.
    #[wasm_bindgen(js_name = labelsPtr)]
    pub fn labels_ptr(&self) -> usize {
        self.inner.labels.as_ptr() as usize
    }

    /// Returns the element count of `labels`.
    #[wasm_bindgen(js_name = labelsLen)]
    pub fn labels_len(&self) -> usize {
        self.inner.labels.len()
    }

    /// Returns the number of distinct communities detected.
    #[wasm_bindgen(getter, js_name = numCommunities)]
    pub fn num_communities(&self) -> usize {
        self.inner.num_communities
    }

    /// Returns the objective quality value achieved by Leiden.
    #[wasm_bindgen(getter)]
    pub fn quality(&self) -> f64 {
        self.inner.quality
    }

    /// Returns the standard normalized modularity $Q$ evaluated at $\gamma = 1.0$.
    #[wasm_bindgen(getter)]
    pub fn modularity(&self) -> f64 {
        self.inner.modularity
    }

    /// Returns the number of multi-level aggregation passes committed.
    #[wasm_bindgen(getter)]
    pub fn passes(&self) -> usize {
        self.inner.passes
    }

    /// Converts this result into a plain JavaScript object without requiring `serde`.
    #[wasm_bindgen(js_name = toRecord)]
    pub fn to_record(&self) -> Result<js_sys::Object, JsValue> {
        let obj = js_sys::Object::new();
        set_js_prop(
            &obj,
            "labels",
            &js_sys::Uint32Array::from(self.inner.labels.as_slice()).into(),
        )?;
        set_f64_prop(&obj, "numCommunities", self.inner.num_communities as f64)?;
        set_f64_prop(&obj, "quality", self.inner.quality)?;
        set_f64_prop(&obj, "modularity", self.inner.modularity)?;
        set_f64_prop(&obj, "passes", self.inner.passes as f64)?;
        Ok(obj)
    }
}

/// Detects communities on an undirected `CsrGraph` via the Leiden algorithm (`Modularity` or `CPM`).
#[wasm_bindgen(js_name = leidenCommunities)]
pub fn js_leiden_communities(
    graph: &JsCsrGraph,
    options: &JsValue,
) -> Result<JsLeidenResult, JsValue> {
    let mut cfg = LeidenConfig::default();
    if check_options_arg(options, "leidenCommunities")? {
        if let Some(obj_str) = opt_string_prop(options, "objective")? {
            cfg.objective = parse_leiden_objective(&obj_str)?;
        }
        cfg.resolution = opt_f64_prop(options, "resolution", cfg.resolution)?;
        cfg.randomness = opt_f64_prop(options, "randomness", cfg.randomness)?;
        cfg.max_passes = opt_usize_prop(options, "maxPasses", cfg.max_passes)?;
        cfg.min_quality_gain = opt_f64_prop(options, "minQualityGain", cfg.min_quality_gain)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    let inner = leiden(graph.inner(), &cfg).map_err(core_err_to_js)?;
    Ok(JsLeidenResult { inner })
}
