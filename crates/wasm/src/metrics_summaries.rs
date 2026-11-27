//! Auxiliary metric summary wrapper types (`PrismDistortion`, `MinimumAngleSummary`, `AdjustedPatchNigDetail`).

use egraph::metrics::{AdjustedPatchNigDetail, MinimumAngleSummary};
use wasm_bindgen::prelude::*;

use crate::js_convert::{set_f64_prop, set_js_prop};

/// PRISM layout-preservation distortion metrics comparing an original and adjusted layout.
#[wasm_bindgen(js_name = PrismDistortion)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsPrismDistortion {
    sigma_dist: f64,
    sigma_disp: f64,
}

impl JsPrismDistortion {
    pub(crate) fn new(sigma_dist: f64, sigma_disp: f64) -> Self {
        Self {
            sigma_dist,
            sigma_disp,
        }
    }
}

#[wasm_bindgen(js_class = PrismDistortion)]
impl JsPrismDistortion {
    /// Returns the proximity edge-length ratio coefficient of variation (`sigmaDist`).
    #[wasm_bindgen(getter, js_name = sigmaDist)]
    pub fn sigma_dist(&self) -> f64 {
        self.sigma_dist
    }

    /// Returns the scale-, translation-, and rotation-invariant Procrustes disparity (`sigmaDisp`)
    /// lying in `[0.0, 1.0]`.
    #[wasm_bindgen(getter, js_name = sigmaDisp)]
    pub fn sigma_disp(&self) -> f64 {
        self.sigma_disp
    }

    /// Converts fields into a plain JavaScript object without requiring `serde`.
    #[wasm_bindgen(js_name = toRecord)]
    pub fn to_record(&self) -> Result<js_sys::Object, JsError> {
        let obj = js_sys::Object::new();
        set_f64_prop(&obj, "sigmaDist", self.sigma_dist)?;
        set_f64_prop(&obj, "sigmaDisp", self.sigma_disp)?;
        Ok(obj)
    }
}

/// DRGraph Eq. (19) minimum-angle resolution summary (`a1` and `a2`).
#[wasm_bindgen(js_name = MinimumAngleSummary)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsMinimumAngleSummary {
    inner: MinimumAngleSummary,
}

impl JsMinimumAngleSummary {
    pub(crate) fn from_inner(inner: MinimumAngleSummary) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen(js_class = MinimumAngleSummary)]
impl JsMinimumAngleSummary {
    #[wasm_bindgen(getter)]
    pub fn a1(&self) -> f64 {
        self.inner.a1
    }

    #[wasm_bindgen(getter)]
    pub fn a2(&self) -> f64 {
        self.inner.a2
    }

    #[wasm_bindgen(js_name = toRecord)]
    pub fn to_record(&self) -> Result<js_sys::Object, JsError> {
        let obj = js_sys::Object::new();
        set_f64_prop(&obj, "a1", self.inner.a1)?;
        set_f64_prop(&obj, "a2", self.inner.a2)?;
        Ok(obj)
    }
}

/// Detailed patch-occupancy diagnostics returned by `adjustedPatchNigDetail`.
#[wasm_bindgen(js_name = AdjustedPatchNigDetail)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsAdjustedPatchNigDetail {
    inner: AdjustedPatchNigDetail,
}

impl JsAdjustedPatchNigDetail {
    pub(crate) fn from_inner(inner: AdjustedPatchNigDetail) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen(js_class = AdjustedPatchNigDetail)]
impl JsAdjustedPatchNigDetail {
    #[wasm_bindgen(getter)]
    pub fn nig(&self) -> f64 {
        self.inner.nig
    }

    #[wasm_bindgen(getter, js_name = hObs)]
    pub fn h_obs(&self) -> f64 {
        self.inner.h_obs
    }

    #[wasm_bindgen(getter, js_name = hNull)]
    pub fn h_null(&self) -> f64 {
        self.inner.h_null
    }

    #[wasm_bindgen(getter, js_name = singletonPatchRatio)]
    pub fn singleton_patch_ratio(&self) -> f64 {
        self.inner.singleton_patch_ratio
    }

    #[wasm_bindgen(getter, js_name = numNonemptyPatches)]
    pub fn num_nonempty_patches(&self) -> usize {
        self.inner.num_nonempty_patches
    }

    #[wasm_bindgen(getter, js_name = recommendedPatchRadius)]
    pub fn recommended_patch_radius(&self) -> usize {
        self.inner.recommended_patch_radius
    }

    #[wasm_bindgen(getter, js_name = isDegenerate)]
    pub fn is_degenerate(&self) -> bool {
        self.inner.is_degenerate
    }

    #[wasm_bindgen(js_name = toRecord)]
    pub fn to_record(&self) -> Result<js_sys::Object, JsError> {
        let obj = js_sys::Object::new();
        set_f64_prop(&obj, "nig", self.inner.nig)?;
        set_f64_prop(&obj, "hObs", self.inner.h_obs)?;
        set_f64_prop(&obj, "hNull", self.inner.h_null)?;
        set_f64_prop(
            &obj,
            "singletonPatchRatio",
            self.inner.singleton_patch_ratio,
        )?;
        set_f64_prop(
            &obj,
            "numNonemptyPatches",
            self.inner.num_nonempty_patches as f64,
        )?;
        set_f64_prop(
            &obj,
            "recommendedPatchRadius",
            self.inner.recommended_patch_radius as f64,
        )?;
        set_js_prop(
            &obj,
            "isDegenerate",
            &JsValue::from_bool(self.inner.is_degenerate),
        )?;
        Ok(obj)
    }
}
