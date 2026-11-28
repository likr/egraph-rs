//! WebAssembly linear-memory `Layout2d` and `LayoutNd` coordinate buffer bindings.

use crate::js_convert::{
    core_err_to_js, ffi_err_to_js, invalid_input_err, non_finite_err, parse_non_negative_usize,
    parse_seed_u64,
};
use egraph::core::{CoreError, Layout2d, LayoutNd};
use egraph_ffi::InterleavedLayout2d;
use wasm_bindgen::{JsCast, prelude::*};

use crate::js_options::{check_options_arg, is_plain_js_object, opt_f64_prop, opt_seed_prop};

/// Returns the active WebAssembly linear `WebAssembly.Memory` instance.
///
/// The JavaScript wrapper uses `wasmMemory().buffer` to detect `memory.grow` buffer detachments
/// in $O(1)$ time before returning a cached `Float32Array` view from `Layout*.coordsView()`.
#[wasm_bindgen(js_name = wasmMemory)]
pub fn wasm_memory() -> js_sys::WebAssembly::Memory {
    wasm_bindgen::memory().unchecked_into()
}

/// Contiguous interleaved `[x0, y0, x1, y1, ...]` 2D node coordinate buffer (`f32`)
/// suitable for zero-copy `Float32Array` linear-memory views via `coordsView()`.
#[wasm_bindgen(js_name = Layout2d)]
#[derive(Debug, Clone)]
pub struct JsLayout2d {
    inner: InterleavedLayout2d<f32>,
}

impl JsLayout2d {
    /// Creates a `JsLayout2d` by copying coordinates from a core SoA `Layout2d<f32>`.
    pub(crate) fn from_layout2d(layout: &Layout2d<f32>) -> Self {
        Self {
            inner: InterleavedLayout2d::from_layout2d(layout),
        }
    }

    /// Returns a shared reference to the underlying `InterleavedLayout2d<f32>`.
    pub fn inner(&self) -> &InterleavedLayout2d<f32> {
        &self.inner
    }

    /// Returns a mutable reference to the underlying `InterleavedLayout2d<f32>`.
    pub fn inner_mut(&mut self) -> &mut InterleavedLayout2d<f32> {
        &mut self.inner
    }
}

#[wasm_bindgen(js_class = Layout2d)]
impl JsLayout2d {
    /// Creates a zero-initialized `Layout2d` for `n` vertices.
    pub fn zeros(n: f64) -> Result<JsLayout2d, JsValue> {
        let count = parse_non_negative_usize(n, "n")?;
        let inner = InterleavedLayout2d::<f32>::try_zeros(count).map_err(ffi_err_to_js)?;
        Ok(JsLayout2d { inner })
    }

    /// Creates a deterministically initialized `Layout2d` for `n` vertices using `seed` (default `42`)
    /// and isotropic uniform range `[-0.5 * scale, 0.5 * scale)^2` (`scale` default `1.0`).
    pub fn init(
        n: f64,
        seed_or_options: &JsValue,
        scale: Option<f64>,
    ) -> Result<JsLayout2d, JsValue> {
        let count = parse_non_negative_usize(n, "n")?;
        let (seed_u64, scale_val) = if is_plain_js_object(seed_or_options) {
            if scale.is_some() {
                return Err(invalid_input_err(
                    "Layout2d.init: unexpected trailing scale argument when passing an options object",
                ));
            }
            (
                opt_seed_prop(seed_or_options)?,
                opt_f64_prop(seed_or_options, "scale", 1.0)?,
            )
        } else {
            (parse_seed_u64(seed_or_options)?, scale.unwrap_or(1.0))
        };
        let soa = Layout2d::<f32>::init(count, seed_u64, scale_val).map_err(core_err_to_js)?;
        Ok(Self::from_layout2d(&soa))
    }

    /// Returns a deep copy of this `Layout2d`.
    pub fn copy(&self) -> JsLayout2d {
        self.clone()
    }

    /// Creates a `Layout2d` from an even-length interleaved `[x0, y0, ...]` array.
    #[wasm_bindgen(js_name = fromCoords)]
    pub fn from_coords(coords: &[f32]) -> Result<JsLayout2d, JsValue> {
        let inner = InterleavedLayout2d::<f32>::from_interleaved(coords).map_err(ffi_err_to_js)?;
        Ok(JsLayout2d { inner })
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

    /// Returns the total scalar length of the interleaved buffer (`2 * len()`).
    #[wasm_bindgen(js_name = coordsLen)]
    pub fn coords_len(&self) -> usize {
        self.inner.coords_len()
    }

    /// Returns a freshly allocated `Float32Array` copy of `[x0, y0, ...]` coordinates.
    #[wasm_bindgen(js_name = copyCoords)]
    pub fn copy_coords(&self) -> Vec<f32> {
        self.inner.as_slice().to_vec()
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

    /// Sets the `x` coordinate of vertex `index`.
    #[wasm_bindgen(js_name = setX)]
    pub fn set_x(&mut self, index: f64, value: f32) -> Result<(), JsValue> {
        let idx = parse_non_negative_usize(index, "index")?;
        if !value.is_finite() {
            return Err(non_finite_err("Layout2d.setX value must be finite"));
        }
        self.inner.set_x(idx, value).map_err(ffi_err_to_js)
    }

    /// Sets the `y` coordinate of vertex `index`.
    #[wasm_bindgen(js_name = setY)]
    pub fn set_y(&mut self, index: f64, value: f32) -> Result<(), JsValue> {
        let idx = parse_non_negative_usize(index, "index")?;
        if !value.is_finite() {
            return Err(non_finite_err("Layout2d.setY value must be finite"));
        }
        self.inner.set_y(idx, value).map_err(ffi_err_to_js)
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
        let mut soa = self.inner.to_layout2d();
        soa.center_and_normalize_to_std(t_std)
            .map_err(core_err_to_js)?;
        self.inner.copy_from_layout2d(&soa).map_err(ffi_err_to_js)?;
        Ok(())
    }

    /// Returns `true` if every coordinate in the buffer is finite (`!NaN && !Inf`).
    #[wasm_bindgen(js_name = isAllFinite)]
    pub fn is_all_finite(&self) -> bool {
        self.inner.is_all_finite()
    }
}

/// Contiguous row-major `(numNodes * dim)` $d$-dimensional node coordinate buffer (`f32`)
/// suitable for zero-copy `Float32Array` linear-memory views via `coordsView()`.
#[wasm_bindgen(js_name = LayoutNd)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsLayoutNd {
    inner: LayoutNd<f32>,
}

impl JsLayoutNd {
    /// Creates a `JsLayoutNd` from a core `LayoutNd<f32>`.
    pub(crate) fn from_inner(inner: LayoutNd<f32>) -> Self {
        Self { inner }
    }

    /// Returns a shared reference to the underlying `LayoutNd<f32>`.
    pub fn inner(&self) -> &LayoutNd<f32> {
        &self.inner
    }

    /// Returns a mutable reference to the underlying `LayoutNd<f32>`.
    pub fn inner_mut(&mut self) -> &mut LayoutNd<f32> {
        &mut self.inner
    }
}

#[wasm_bindgen(js_class = LayoutNd)]
impl JsLayoutNd {
    /// Creates a zero-initialized `dims`-dimensional layout for `n` vertices.
    pub fn zeros(n: f64, dims: f64) -> Result<JsLayoutNd, JsValue> {
        let num_nodes = parse_non_negative_usize(n, "n")?;
        let d = parse_non_negative_usize(dims, "dims")?;
        let inner = LayoutNd::<f32>::zeros(num_nodes, d).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Creates a deterministically seeded uniform random `dims`-dimensional layout.
    pub fn init(n: f64, dims: f64, options: &JsValue) -> Result<JsLayoutNd, JsValue> {
        let num_nodes = parse_non_negative_usize(n, "n")?;
        let d = parse_non_negative_usize(dims, "dims")?;
        let (seed_u64, scale_val) = if check_options_arg(options, "LayoutNd.init")? {
            (
                opt_seed_prop(options)?,
                opt_f64_prop(options, "scale", 1.0)?,
            )
        } else {
            (crate::js_convert::DEFAULT_SEED, 1.0)
        };
        let inner =
            LayoutNd::<f32>::init(num_nodes, d, seed_u64, scale_val).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Creates a `LayoutNd` from flat row-major coordinates of length `n * dims`.
    #[wasm_bindgen(js_name = fromCoords)]
    pub fn from_coords(n: f64, dims: f64, coords: &[f32]) -> Result<JsLayoutNd, JsValue> {
        let num_nodes = parse_non_negative_usize(n, "n")?;
        let d = parse_non_negative_usize(dims, "dims")?;
        if coords.iter().any(|v| !v.is_finite()) {
            return Err(core_err_to_js(CoreError::non_finite(
                "LayoutNd::fromCoords",
            )));
        }
        let inner = LayoutNd::<f32>::from_flat(num_nodes, d, coords).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Returns a deep copy of this `LayoutNd`.
    pub fn copy(&self) -> JsLayoutNd {
        self.clone()
    }

    /// Returns the number of vertices in the layout.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Returns the dimensionality of the embedding space.
    #[wasm_bindgen(getter)]
    pub fn dims(&self) -> usize {
        self.inner.dims()
    }

    /// Returns the number of vertices in the layout.
    pub fn len(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Returns `true` if the layout contains zero vertices.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the scalar coordinate at `(node, dim)`.
    pub fn get(&self, node: f64, dim: f64) -> Result<f32, JsValue> {
        let i = parse_non_negative_usize(node, "node")?;
        let d = parse_non_negative_usize(dim, "dim")?;
        self.inner.get(i, d).map_err(core_err_to_js)
    }

    /// Sets the scalar coordinate at `(node, dim)`.
    pub fn set(&mut self, node: f64, dim: f64, value: f32) -> Result<(), JsValue> {
        let i = parse_non_negative_usize(node, "node")?;
        let d = parse_non_negative_usize(dim, "dim")?;
        self.inner.set(i, d, value).map_err(core_err_to_js)
    }

    /// Centers coordinates at zero mean per axis and scales isotropically to `targetStd` (default `1.0`).
    #[wasm_bindgen(js_name = normalizeStd)]
    pub fn normalize_std(&mut self, target_std: Option<f64>) -> Result<(), JsValue> {
        let t_std = target_std.unwrap_or(1.0);
        if !t_std.is_finite() || t_std <= 0.0 {
            return Err(core_err_to_js(CoreError::invalid_input(
                "targetStd must be finite and strictly positive",
            )));
        }
        self.inner
            .center_and_normalize_to_std(t_std)
            .map_err(core_err_to_js)
    }

    /// Returns the byte offset of the row-major `f32` buffer in WebAssembly linear memory.
    #[wasm_bindgen(js_name = coordsPtr)]
    pub fn coords_ptr(&self) -> usize {
        self.inner.coords().as_ptr() as usize
    }

    /// Returns the total scalar length of the row-major buffer (`numNodes * dims`).
    #[wasm_bindgen(js_name = coordsLen)]
    pub fn coords_len(&self) -> usize {
        self.inner.coords().len()
    }

    /// Returns a freshly allocated `Float32Array` copy of the row-major coordinate buffer.
    #[wasm_bindgen(js_name = copyCoords)]
    pub fn copy_coords(&self) -> Vec<f32> {
        self.inner.coords().to_vec()
    }

    /// Returns `true` if every coordinate in the buffer is finite (`!NaN && !Inf`).
    #[wasm_bindgen(js_name = isAllFinite)]
    pub fn is_all_finite(&self) -> bool {
        self.inner.is_all_finite()
    }
}
