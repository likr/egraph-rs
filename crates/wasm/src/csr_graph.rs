//! WebAssembly bindings for zero-copy `CsrGraph` construction and graph queries.

use crate::js_convert::{core_err_to_js, ffi_err_to_js, parse_non_negative_usize};
use egraph::core::CsrGraph;
use egraph_ffi::{build_csr_f32, build_edges_f32};
use wasm_bindgen::prelude::*;

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

/// Compact `u32`-indexed Compressed Sparse Row (CSR) graph storing `f32` edge weights.
#[wasm_bindgen(js_name = CsrGraph)]
#[derive(Debug, Clone)]
pub struct JsCsrGraph {
    inner: CsrGraph<f32>,
}

impl JsCsrGraph {
    /// Wraps an existing `egraph_core::CsrGraph<f32>` in a `JsCsrGraph`.
    pub(crate) fn from_inner(inner: CsrGraph<f32>) -> Self {
        Self { inner }
    }

    /// Returns a reference to the underlying `egraph_core::CsrGraph<f32>`.
    pub fn inner(&self) -> &CsrGraph<f32> {
        &self.inner
    }
}

/// Largest connected component result returned by `CsrGraph.largestConnectedComponent()`.
#[wasm_bindgen(js_name = LargestConnectedComponentResult)]
#[derive(Debug, Clone)]
pub struct JsLargestConnectedComponent {
    graph: CsrGraph<f32>,
    original_node_ids: Vec<u32>,
}

#[wasm_bindgen(js_class = LargestConnectedComponentResult)]
impl JsLargestConnectedComponent {
    /// Returns a cloned handle to the induced `CsrGraph` of the largest connected component.
    pub fn graph(&self) -> JsCsrGraph {
        JsCsrGraph::from_inner(self.graph.clone())
    }

    /// Returns the sorted original vertex IDs (`Uint32Array`) corresponding to compact vertices `0..k`.
    #[wasm_bindgen(js_name = originalNodeIds)]
    pub fn original_node_ids(&self) -> Vec<u32> {
        self.original_node_ids.clone()
    }

    /// Alias for `originalNodeIds()`.
    #[wasm_bindgen(js_name = nodeIds)]
    pub fn node_ids(&self) -> Vec<u32> {
        self.original_node_ids.clone()
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

    /// Returns the number of stored CSR half-edge entries (`indices().length`).
    pub fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    /// Returns `true` if the graph is directed.
    #[wasm_bindgen(js_name = isDirected)]
    pub fn is_directed(&self) -> bool {
        self.inner.is_directed()
    }

    /// Returns `true` if explicit edge weights are stored.
    #[wasm_bindgen(js_name = isWeighted)]
    pub fn is_weighted(&self) -> bool {
        self.inner.is_weighted()
    }

    /// Returns the degree (outgoing degree for directed graphs) of vertex `u`.
    pub fn degree(&self, u: f64) -> Result<usize, JsError> {
        let idx = parse_non_negative_usize(u, "u")?;
        self.inner.try_degree(idx).map_err(core_err_to_js)
    }

    /// Returns a copy of the neighbor indices adjacent to vertex `u`.
    pub fn neighbors(&self, u: f64) -> Result<Vec<u32>, JsError> {
        let idx = parse_non_negative_usize(u, "u")?;
        self.inner
            .try_neighbors(idx)
            .map(|slice| slice.to_vec())
            .map_err(core_err_to_js)
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

    /// Extracts the induced subgraph on `nodes` with compact `0..nodes.length` vertex IDs.
    #[wasm_bindgen(js_name = inducedSubgraph)]
    pub fn induced_subgraph(&self, nodes: &[u32]) -> Result<JsCsrGraph, JsError> {
        let sub = self.inner.induced_subgraph(nodes).map_err(core_err_to_js)?;
        Ok(JsCsrGraph { inner: sub })
    }

    /// Extracts the largest (weakly) connected component as `{ graph(), originalNodeIds() }`.
    #[wasm_bindgen(js_name = largestConnectedComponent)]
    pub fn largest_connected_component(&self) -> JsLargestConnectedComponent {
        let (graph, original_node_ids) = self.inner.largest_connected_component();
        JsLargestConnectedComponent {
            graph,
            original_node_ids,
        }
    }
}
