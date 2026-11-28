//! WebAssembly bindings for zero-copy `CsrGraph` construction and graph queries.

use crate::js_convert::{
    core_err_to_js, ffi_err_to_js, invalid_input_err, parse_non_negative_usize,
};
use egraph::core::CsrGraph;
use egraph_ffi::{build_csr_f32, build_edges_f32};
use wasm_bindgen::prelude::*;

fn parse_csr_options(
    directed_or_options: &JsValue,
    sanitize: Option<bool>,
) -> Result<(bool, bool), JsValue> {
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
            return Err(invalid_input_err(
                "unexpected sanitize argument when passing an options object",
            ));
        }
        let directed_val =
            js_sys::Reflect::get(directed_or_options, &JsValue::from_str("directed")).map_err(
                |err| invalid_input_err(format!("failed to read options.directed: {err:?}")),
            )?;
        let directed = if directed_val.is_undefined() || directed_val.is_null() {
            false
        } else {
            directed_val
                .as_bool()
                .ok_or_else(|| invalid_input_err("options.directed must be a boolean"))?
        };

        let sanitize_val =
            js_sys::Reflect::get(directed_or_options, &JsValue::from_str("sanitize")).map_err(
                |err| invalid_input_err(format!("failed to read options.sanitize: {err:?}")),
            )?;
        let sanitize_flag = if sanitize_val.is_undefined() || sanitize_val.is_null() {
            true
        } else {
            sanitize_val
                .as_bool()
                .ok_or_else(|| invalid_input_err("options.sanitize must be a boolean"))?
        };

        let dtype_val =
            js_sys::Reflect::get(directed_or_options, &JsValue::from_str("weightDtype")).map_err(
                |err| invalid_input_err(format!("failed to read options.weightDtype: {err:?}")),
            )?;
        if !dtype_val.is_undefined() && !dtype_val.is_null() {
            let dtype_str = dtype_val
                .as_string()
                .ok_or_else(|| invalid_input_err("options.weightDtype must be a string"))?;
            if dtype_str != "float32" {
                return Err(invalid_input_err(format!(
                    "unsupported weightDtype {dtype_str:?}; WebAssembly CsrGraph only supports 'float32'"
                )));
            }
        }

        return Ok((directed, sanitize_flag));
    }
    Err(invalid_input_err(
        "expected boolean `directed` or options object `{ directed?: boolean, sanitize?: boolean, weightDtype?: 'float32' }`",
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
    /// options object `{ directed?: boolean, sanitize?: boolean, weightDtype?: "float32" }`.
    /// Defaults are `directed = false` and `sanitize = true`.
    #[wasm_bindgen(js_name = fromCsr)]
    pub fn from_csr(
        indptr: &[u32],
        indices: &[u32],
        weights: Option<Vec<f32>>,
        directed_or_options: &JsValue,
        sanitize: Option<bool>,
    ) -> Result<JsCsrGraph, JsValue> {
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
    ) -> Result<JsCsrGraph, JsValue> {
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

    /// Returns the number of vertices in the graph.
    pub fn len(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Returns `true` if the graph contains zero vertices.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.num_nodes() == 0
    }

    /// Returns the number of vertices in the graph.
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
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

    /// Returns the scalar weight dtype stored in WebAssembly (`"float32"`).
    #[wasm_bindgen(getter, js_name = weightDtype)]
    pub fn weight_dtype(&self) -> String {
        "float32".to_owned()
    }

    /// Returns `true` if the graph has <= 1 vertex or forms a single weakly connected component.
    #[wasm_bindgen(js_name = isConnected)]
    pub fn is_connected(&self) -> bool {
        self.inner.is_connected()
    }

    /// Returns a flat `[u0, v0, u1, v1, ...]` `Uint32Array` of canonical edges.
    pub fn edges(&self) -> Vec<u32> {
        let pairs = self.inner.edges();
        let mut flat = Vec::with_capacity(pairs.len().saturating_mul(2));
        for (u, v) in pairs {
            flat.push(u);
            flat.push(v);
        }
        flat
    }

    /// Returns `true` if an edge `(u, v)` exists in the graph.
    #[wasm_bindgen(js_name = hasEdge)]
    pub fn has_edge(&self, u: f64, v: f64) -> Result<bool, JsValue> {
        let ui = parse_non_negative_usize(u, "u")?;
        let vi = parse_non_negative_usize(v, "v")?;
        self.inner.try_has_edge(ui, vi).map_err(core_err_to_js)
    }

    /// Returns the stored weight of edge `(u, v)`, `1.0` if unweighted, or `undefined` if absent.
    #[wasm_bindgen(js_name = edgeWeight)]
    pub fn edge_weight(&self, u: f64, v: f64) -> Result<Option<f32>, JsValue> {
        let ui = parse_non_negative_usize(u, "u")?;
        let vi = parse_non_negative_usize(v, "v")?;
        self.inner.try_edge_weight(ui, vi).map_err(core_err_to_js)
    }

    /// Returns the degree (outgoing degree for directed graphs) of vertex `u`.
    pub fn degree(&self, u: f64) -> Result<usize, JsValue> {
        let idx = parse_non_negative_usize(u, "u")?;
        self.inner.try_degree(idx).map_err(core_err_to_js)
    }

    /// Returns a copy of the neighbor indices adjacent to vertex `u`.
    pub fn neighbors(&self, u: f64) -> Result<Vec<u32>, JsValue> {
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
    pub fn induced_subgraph(&self, nodes: &[u32]) -> Result<JsCsrGraph, JsValue> {
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
