//! WebAssembly linear-memory `Layout3d` coordinate buffer bindings.

use crate::js_convert::{
    DEFAULT_SEED, core_err_to_js, ffi_err_to_js, non_finite_err, parse_non_negative_usize,
};
use crate::js_options::{check_options_arg, opt_f64_prop, opt_seed_prop};
use egraph::core::{CoreError, Layout3d};
use egraph_ffi::InterleavedLayout3d;
use wasm_bindgen::prelude::*;

/// Contiguous interleaved `[x0, y0, z0, x1, y1, z1, ...]` 3D node coordinate buffer (`f32`)
/// suitable for zero-copy `Float32Array` linear-memory views via `coordsView()`.
#[wasm_bindgen(js_name = Layout3d)]
#[derive(Debug, Clone)]
pub struct JsLayout3d {
    inner: InterleavedLayout3d<f32>,
}

impl JsLayout3d {
    /// Creates a `JsLayout3d` by copying coordinates from a core SoA `Layout3d<f32>`.
    pub(crate) fn from_layout3d(layout: &Layout3d<f32>) -> Self {
        Self {
            inner: InterleavedLayout3d::from_layout3d(layout),
        }
    }

    /// Returns a shared reference to the underlying `InterleavedLayout3d<f32>`.
    pub fn inner(&self) -> &InterleavedLayout3d<f32> {
        &self.inner
    }

    /// Returns a mutable reference to the underlying `InterleavedLayout3d<f32>`.
    pub fn inner_mut(&mut self) -> &mut InterleavedLayout3d<f32> {
        &mut self.inner
    }
}

#[wasm_bindgen(js_class = Layout3d)]
impl JsLayout3d {
    /// Creates a zero-initialized `Layout3d` for `n` vertices.
    pub fn zeros(n: f64) -> Result<JsLayout3d, JsValue> {
        let count = parse_non_negative_usize(n, "n")?;
        let inner = InterleavedLayout3d::<f32>::try_zeros(count).map_err(ffi_err_to_js)?;
        Ok(JsLayout3d { inner })
    }

    /// Creates a deterministically initialized `Layout3d` for `n` vertices using `{ seed?, scale? }`.
    pub fn init(n: f64, options: &JsValue) -> Result<JsLayout3d, JsValue> {
        let count = parse_non_negative_usize(n, "n")?;
        let (seed_u64, scale_val) = if check_options_arg(options, "Layout3d.init")? {
            (
                opt_seed_prop(options)?,
                opt_f64_prop(options, "scale", 1.0)?,
            )
        } else {
            (DEFAULT_SEED, 1.0)
        };
        let soa = Layout3d::<f32>::init(count, seed_u64, scale_val).map_err(core_err_to_js)?;
        Ok(Self::from_layout3d(&soa))
    }

    /// Returns a deep copy of this `Layout3d`.
    pub fn copy(&self) -> JsLayout3d {
        self.clone()
    }

    /// Creates a `Layout3d` from an interleaved `[x0, y0, z0, ...]` array of length divisible by 3.
    #[wasm_bindgen(js_name = fromCoords)]
    pub fn from_coords(coords: &[f32]) -> Result<JsLayout3d, JsValue> {
        let inner = InterleavedLayout3d::<f32>::from_interleaved(coords).map_err(ffi_err_to_js)?;
        Ok(JsLayout3d { inner })
    }

    /// Creates a `Layout3d` from separate `x`, `y`, and `z` coordinate arrays of equal length.
    #[wasm_bindgen(js_name = fromXyz)]
    pub fn from_xyz(x: &[f32], y: &[f32], z: &[f32]) -> Result<JsLayout3d, JsValue> {
        let soa = Layout3d::<f32>::from_xyz_slices(x, y, z).map_err(core_err_to_js)?;
        soa.validate_finite().map_err(core_err_to_js)?;
        Ok(Self::from_layout3d(&soa))
    }

    /// Returns the number of vertices in the layout.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns the number of vertices in the layout.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the layout contains zero vertices.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the byte offset of the interleaved `f32` buffer in WebAssembly linear memory.
    #[wasm_bindgen(js_name = coordsPtr)]
    pub fn coords_ptr(&self) -> usize {
        self.inner.coords_ptr() as usize
    }

    /// Returns the total scalar length of the interleaved buffer (`3 * len()`).
    #[wasm_bindgen(js_name = coordsLen)]
    pub fn coords_len(&self) -> usize {
        self.inner.coords_len()
    }

    /// Returns a freshly allocated `Float32Array` copy of `[x0, y0, z0, ...]` coordinates.
    #[wasm_bindgen(js_name = copyCoords)]
    pub fn copy_coords(&self) -> Vec<f32> {
        self.inner.as_slice().to_vec()
    }

    /// Returns a freshly allocated `Float32Array` copy of interleaved `[x0, y0, z0, ...]` coordinates.
    #[wasm_bindgen(js_name = toInterleaved)]
    pub fn to_interleaved(&self) -> Vec<f32> {
        self.inner.as_slice().to_vec()
    }

    /// Returns `[x, y, z]` coordinates for vertex `node`.
    pub fn get(&self, node: f64) -> Result<Vec<f32>, JsValue> {
        let idx = parse_non_negative_usize(node, "node")?;
        let x = self.inner.x(idx).map_err(ffi_err_to_js)?;
        let y = self.inner.y(idx).map_err(ffi_err_to_js)?;
        let z = self.inner.z(idx).map_err(ffi_err_to_js)?;
        Ok(vec![x, y, z])
    }

    /// Sets `[x, y, z]` coordinates for vertex `node`.
    pub fn set(&mut self, node: f64, x: f32, y: f32, z: f32) -> Result<(), JsValue> {
        let idx = parse_non_negative_usize(node, "node")?;
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            return Err(non_finite_err("Layout3d.set"));
        }
        self.inner.set_x(idx, x).map_err(ffi_err_to_js)?;
        self.inner.set_y(idx, y).map_err(ffi_err_to_js)?;
        self.inner.set_z(idx, z).map_err(ffi_err_to_js)
    }

    /// Returns the `x` coordinate of vertex `index`.
    pub fn x(&self, index: f64) -> Result<f32, JsValue> {
        let idx = parse_non_negative_usize(index, "index")?;
        self.inner.x(idx).map_err(ffi_err_to_js)
    }

    /// Returns the `y` coordinate of vertex `index`.
    pub fn y(&self, index: f64) -> Result<f32, JsValue> {
        let idx = parse_non_negative_usize(index, "index")?;
        self.inner.y(idx).map_err(ffi_err_to_js)
    }

    /// Returns the `z` coordinate of vertex `index`.
    pub fn z(&self, index: f64) -> Result<f32, JsValue> {
        let idx = parse_non_negative_usize(index, "index")?;
        self.inner.z(idx).map_err(ffi_err_to_js)
    }

    /// Sets the `x` coordinate of vertex `index`.
    #[wasm_bindgen(js_name = setX)]
    pub fn set_x(&mut self, index: f64, value: f32) -> Result<(), JsValue> {
        let idx = parse_non_negative_usize(index, "index")?;
        if !value.is_finite() {
            return Err(non_finite_err("Layout3d.setX value must be finite"));
        }
        self.inner.set_x(idx, value).map_err(ffi_err_to_js)
    }

    /// Sets the `y` coordinate of vertex `index`.
    #[wasm_bindgen(js_name = setY)]
    pub fn set_y(&mut self, index: f64, value: f32) -> Result<(), JsValue> {
        let idx = parse_non_negative_usize(index, "index")?;
        if !value.is_finite() {
            return Err(non_finite_err("Layout3d.setY value must be finite"));
        }
        self.inner.set_y(idx, value).map_err(ffi_err_to_js)
    }

    /// Sets the `z` coordinate of vertex `index`.
    #[wasm_bindgen(js_name = setZ)]
    pub fn set_z(&mut self, index: f64, value: f32) -> Result<(), JsValue> {
        let idx = parse_non_negative_usize(index, "index")?;
        if !value.is_finite() {
            return Err(non_finite_err("Layout3d.setZ value must be finite"));
        }
        self.inner.set_z(idx, value).map_err(ffi_err_to_js)
    }

    /// Centers coordinates at zero mean and scales isotropically to `targetStd` (default `1.0`).
    #[wasm_bindgen(js_name = normalizeStd)]
    pub fn normalize_std(&mut self, target_std: Option<f64>) -> Result<(), JsValue> {
        let t_std = target_std.unwrap_or(1.0);
        if !t_std.is_finite() || t_std <= 0.0 {
            return Err(core_err_to_js(CoreError::invalid_input(
                "targetStd must be finite and strictly positive",
            )));
        }
        let mut soa = self.inner.to_layout3d();
        soa.center_and_normalize_to_std(t_std)
            .map_err(core_err_to_js)?;
        self.inner.copy_from_layout3d(&soa).map_err(ffi_err_to_js)?;
        Ok(())
    }

    /// Returns `true` if every coordinate in the buffer is finite (`!NaN && !Inf`).
    #[wasm_bindgen(js_name = isAllFinite)]
    pub fn is_all_finite(&self) -> bool {
        self.inner.is_all_finite()
    }
}
