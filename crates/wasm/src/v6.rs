//! v6 zero-copy `CsrGraph` and WebAssembly linear-memory `Layout2d` bindings.

use egraph_core::CsrGraph;
use egraph_ffi::{FfiError, InterleavedLayout2d, build_csr_f32, build_edges_f32};
use wasm_bindgen::{JsCast, prelude::*};

fn ffi_err_to_js(err: FfiError) -> JsError {
    JsError::new(&err.to_string())
}

fn parse_non_negative_usize(value: f64, name: &str) -> Result<usize, JsError> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > (u32::MAX as f64) {
        return Err(JsError::new(&format!(
            "{name} must be a non-negative integer <= u32::MAX, got {value}"
        )));
    }
    Ok(value as usize)
}

fn parse_csr_options(
    directed_or_options: &JsValue,
    sanitize: Option<bool>,
) -> Result<(bool, bool), JsError> {
    if directed_or_options.is_undefined() || directed_or_options.is_null() {
        return Ok((false, sanitize.unwrap_or(true)));
    }
    if let Some(directed) = directed_or_options.as_bool() {
        return Ok((directed, sanitize.unwrap_or(true)));
    }
    if directed_or_options.is_object()
        && !js_sys::Array::is_array(directed_or_options)
        && !js_sys::ArrayBuffer::is_view(directed_or_options)
        && !directed_or_options.is_function()
    {
        if sanitize.is_some() {
            return Err(JsError::new(
                "unexpected sanitize argument when passing an options object",
            ));
        }
        let directed_val =
            js_sys::Reflect::get(directed_or_options, &JsValue::from_str("directed")).map_err(
                |err| JsError::new(&format!("failed to read options.directed: {err:?}")),
            )?;
        let directed = if directed_val.is_undefined() {
            false
        } else {
            directed_val
                .as_bool()
                .ok_or_else(|| JsError::new("options.directed must be a boolean"))?
        };

        let sanitize_val =
            js_sys::Reflect::get(directed_or_options, &JsValue::from_str("sanitize")).map_err(
                |err| JsError::new(&format!("failed to read options.sanitize: {err:?}")),
            )?;
        let sanitize_flag = if sanitize_val.is_undefined() {
            true
        } else {
            sanitize_val
                .as_bool()
                .ok_or_else(|| JsError::new("options.sanitize must be a boolean"))?
        };

        return Ok((directed, sanitize_flag));
    }
    Err(JsError::new(
        "expected boolean `directed` or options object `{ directed?: boolean, sanitize?: boolean }`",
    ))
}

/// Returns the active WebAssembly linear `WebAssembly.Memory` instance.
///
/// The JavaScript wrapper uses `wasmMemory().buffer` to detect `memory.grow` buffer detachments
/// in $O(1)$ time before returning a cached `Float32Array` view from `Layout2d.coordsView()`.
#[wasm_bindgen(js_name = wasmMemory)]
pub fn wasm_memory() -> js_sys::WebAssembly::Memory {
    wasm_bindgen::memory().unchecked_into()
}

/// Compact `u32`-indexed Compressed Sparse Row (CSR) graph storing `f32` edge weights.
#[wasm_bindgen(js_name = CsrGraph)]
#[derive(Debug, Clone)]
pub struct JsCsrGraph {
    inner: CsrGraph<f32>,
}

impl JsCsrGraph {
    /// Returns a reference to the underlying `egraph_core::CsrGraph<f32>`.
    pub fn inner(&self) -> &CsrGraph<f32> {
        &self.inner
    }
}

#[wasm_bindgen(js_class = CsrGraph)]
impl JsCsrGraph {
    /// Constructs a `CsrGraph` from CSR `indptr`, `indices`, and optional `weights`.
    ///
    /// The optional trailing parameters accept either `(directed, sanitize)` booleans or an
    /// options object `{ directed?: boolean, sanitize?: boolean }`. Defaults are
    /// `directed = false` and `sanitize = true`.
    #[wasm_bindgen(js_name = fromCsr)]
    pub fn from_csr(
        indptr: &[u32],
        indices: &[u32],
        weights: Option<Vec<f32>>,
        directed_or_options: &JsValue,
        sanitize: Option<bool>,
    ) -> Result<JsCsrGraph, JsError> {
        let (directed, sanitize_flag) = parse_csr_options(directed_or_options, sanitize)?;
        let inner = build_csr_f32(indptr, indices, weights.as_deref(), directed, sanitize_flag)
            .map_err(ffi_err_to_js)?;
        Ok(JsCsrGraph { inner })
    }

    /// Constructs a `CsrGraph` from a flat interleaved edge list `[u0, v0, u1, v1, ...]`
    /// and optional `weights`.
    #[wasm_bindgen(js_name = fromEdgeList)]
    pub fn from_edge_list(
        num_nodes: f64,
        edges: &[u32],
        weights: Option<Vec<f32>>,
        directed_or_options: &JsValue,
        sanitize: Option<bool>,
    ) -> Result<JsCsrGraph, JsError> {
        let n = parse_non_negative_usize(num_nodes, "numNodes")?;
        let (directed, sanitize_flag) = parse_csr_options(directed_or_options, sanitize)?;
        let inner = build_edges_f32(n, edges, weights.as_deref(), directed, sanitize_flag)
            .map_err(ffi_err_to_js)?;
        Ok(JsCsrGraph { inner })
    }

    /// Returns the number of vertices in the graph.
    #[wasm_bindgen(js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Returns the number of logical edges in the graph.
    #[wasm_bindgen(js_name = numEdges)]
    pub fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    /// Returns `true` if the graph is directed.
    #[wasm_bindgen(js_name = isDirected)]
    pub fn is_directed(&self) -> bool {
        self.inner.is_directed()
    }

    /// Returns a copy of the CSR `indptr` row-offset array (`length === numNodes() + 1`).
    pub fn indptr(&self) -> Vec<u32> {
        self.inner.indptr().to_vec()
    }

    /// Returns a copy of the flat CSR `indices` neighbor array.
    pub fn indices(&self) -> Vec<u32> {
        self.inner.indices().to_vec()
    }

    /// Returns a copy of the flat CSR `weights` array, or `undefined` if unweighted.
    pub fn weights(&self) -> Option<Vec<f32>> {
        self.inner.weights().map(|w| w.to_vec())
    }
}

/// Contiguous interleaved `[x0, y0, x1, y1, ...]` 2D node coordinate buffer (`f32`)
/// suitable for zero-copy `Float32Array` linear-memory views via `coordsView()`.
#[wasm_bindgen(js_name = Layout2d)]
#[derive(Debug, Clone)]
pub struct JsLayout2d {
    inner: InterleavedLayout2d<f32>,
}

impl JsLayout2d {
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
    pub fn zeros(n: f64) -> Result<JsLayout2d, JsError> {
        let count = parse_non_negative_usize(n, "n")?;
        let inner = InterleavedLayout2d::<f32>::try_zeros(count).map_err(ffi_err_to_js)?;
        Ok(JsLayout2d { inner })
    }

    /// Creates a `Layout2d` from an even-length interleaved `[x0, y0, ...]` array.
    #[wasm_bindgen(js_name = fromCoords)]
    pub fn from_coords(coords: &[f32]) -> Result<JsLayout2d, JsError> {
        let inner = InterleavedLayout2d::<f32>::from_interleaved(coords).map_err(ffi_err_to_js)?;
        Ok(JsLayout2d { inner })
    }

    /// Returns the number of vertices in the layout.
    pub fn len(&self) -> usize {
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

    /// Returns the `x` coordinate of vertex `index`.
    pub fn x(&self, index: f64) -> Result<f32, JsError> {
        let idx = parse_non_negative_usize(index, "index")?;
        self.inner.x(idx).map_err(ffi_err_to_js)
    }

    /// Returns the `y` coordinate of vertex `index`.
    pub fn y(&self, index: f64) -> Result<f32, JsError> {
        let idx = parse_non_negative_usize(index, "index")?;
        self.inner.y(idx).map_err(ffi_err_to_js)
    }

    /// Sets the `x` coordinate of vertex `index`.
    #[wasm_bindgen(js_name = setX)]
    pub fn set_x(&mut self, index: f64, value: f32) -> Result<(), JsError> {
        let idx = parse_non_negative_usize(index, "index")?;
        self.inner.set_x(idx, value).map_err(ffi_err_to_js)
    }

    /// Sets the `y` coordinate of vertex `index`.
    #[wasm_bindgen(js_name = setY)]
    pub fn set_y(&mut self, index: f64, value: f32) -> Result<(), JsError> {
        let idx = parse_non_negative_usize(index, "index")?;
        self.inner.set_y(idx, value).map_err(ffi_err_to_js)
    }

    /// Returns `true` if every coordinate in the buffer is finite (`!NaN && !Inf`).
    #[wasm_bindgen(js_name = isAllFinite)]
    pub fn is_all_finite(&self) -> bool {
        self.inner.is_all_finite()
    }
}
