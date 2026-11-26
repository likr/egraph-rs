//! WebAssembly bindings for 2D Delaunay triangulation, rectangle overlap sweeps, and CSV CSR parsing.

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::core_err_to_js;
use crate::layout_buffers::JsLayout2d;
#[cfg(feature = "delaunay")]
use egraph::core::{CoreError, CsrGraph};
#[cfg(feature = "delaunay")]
use egraph::spatial::Delaunay2d;
use egraph::spatial::RectangleOverlapSweep;
use egraph_dataset::parse_csr_csv;
use wasm_bindgen::prelude::*;

/// Deterministic 2D Delaunay triangulation result returned by `delaunayTriangulation`.
#[cfg(feature = "delaunay")]
#[wasm_bindgen(js_name = DelaunayResult)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsDelaunayResult {
    num_nodes: usize,
    edges: Vec<u32>,
    triangles: Vec<u32>,
}

#[cfg(feature = "delaunay")]
#[wasm_bindgen(js_class = DelaunayResult)]
impl JsDelaunayResult {
    /// Returns the number of vertices in the triangulated layout.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns a flat `[u0, v0, u1, v1, ...]` `Uint32Array` of canonical undirected edges (`u < v`).
    pub fn edges(&self) -> Vec<u32> {
        self.edges.clone()
    }

    /// Returns a flat `[a0, b0, c0, a1, b1, c1, ...]` `Uint32Array` of canonical Delaunay triangles.
    pub fn triangles(&self) -> Vec<u32> {
        self.triangles.clone()
    }
}

/// Computes the deterministic 2D Delaunay triangulation of `layout` (`delaunay` feature).
#[cfg(feature = "delaunay")]
#[wasm_bindgen(js_name = delaunayTriangulation)]
pub fn js_delaunay_triangulation(layout: &JsLayout2d) -> Result<JsDelaunayResult, JsError> {
    let soa = layout.inner().to_layout2d();
    let dt = Delaunay2d::triangulate(&soa).map_err(core_err_to_js)?;
    let mut flat_edges = Vec::with_capacity(dt.edges().len() * 2);
    for &(u, v) in dt.edges() {
        flat_edges.push(u);
        flat_edges.push(v);
    }
    let mut flat_triangles = Vec::with_capacity(dt.triangles().len() * 3);
    for &[a, b, c] in dt.triangles() {
        flat_triangles.push(a);
        flat_triangles.push(b);
        flat_triangles.push(c);
    }
    Ok(JsDelaunayResult {
        num_nodes: dt.num_nodes(),
        edges: flat_edges,
        triangles: flat_triangles,
    })
}

#[cfg(feature = "delaunay")]
fn csr_f64_to_f32_unweighted_or_checked(g64: CsrGraph<f64>) -> Result<CsrGraph<f32>, JsError> {
    let weights_f32: Option<Vec<f32>> = g64
        .weights()
        .map(|ws| {
            ws.iter()
                .map(|&w| {
                    let wf = w as f32;
                    if !wf.is_finite() {
                        return Err(core_err_to_js(CoreError::non_finite("CsrGraph weight")));
                    }
                    Ok(wf)
                })
                .collect::<Result<Vec<f32>, JsError>>()
        })
        .transpose()?;
    CsrGraph::<f32>::from_csr(
        g64.indptr(),
        g64.indices(),
        weights_f32.as_deref(),
        g64.is_directed(),
        false,
    )
    .map_err(core_err_to_js)
}

/// Converts the 2D Delaunay triangulation of `layout` directly into an unweighted `CsrGraph`.
#[wasm_bindgen(js_name = delaunayToCsr)]
pub fn js_delaunay_to_csr(layout: &JsLayout2d) -> Result<JsCsrGraph, JsError> {
    #[cfg(feature = "delaunay")]
    {
        let soa = layout.inner().to_layout2d();
        let dt = Delaunay2d::triangulate(&soa).map_err(core_err_to_js)?;
        let csr64 = dt.to_csr().map_err(core_err_to_js)?;
        let csr32 = csr_f64_to_f32_unweighted_or_checked(csr64)?;
        Ok(JsCsrGraph::from_inner(csr32))
    }
    #[cfg(not(feature = "delaunay"))]
    {
        let _ = layout;
        Err(JsError::new(
            "delaunayToCsr requires the 'delaunay' feature to be enabled",
        ))
    }
}

/// Finds all overlapping rectangle pairs `[i0, j0, i1, j1, ...]` (`i < j`) for `layout`
/// with per-node `halfWidths` and `halfHeights` in $O(N \log N + K)$ time.
#[wasm_bindgen(js_name = rectangleOverlaps)]
pub fn js_rectangle_overlaps(
    layout: &JsLayout2d,
    half_widths: &[f32],
    half_heights: &[f32],
) -> Result<Vec<u32>, JsError> {
    let soa = layout.inner().to_layout2d();
    let pairs = RectangleOverlapSweep::find_layout_overlaps(&soa, half_widths, half_heights)
        .map_err(core_err_to_js)?;
    let mut flat = Vec::with_capacity(pairs.len() * 2);
    for (i, j) in pairs {
        let iu = u32::try_from(i)
            .map_err(|_| JsError::new("rectangleOverlaps index i exceeds u32::MAX"))?;
        let ju = u32::try_from(j)
            .map_err(|_| JsError::new("rectangleOverlaps index j exceeds u32::MAX"))?;
        flat.push(iu);
        flat.push(ju);
    }
    Ok(flat)
}

/// Parses an edge-list CSV string (`num_nodes,num_edges` header + `u,v` or weighted `u,v,w` rows) into a `CsrGraph`.
#[wasm_bindgen(js_name = parseCsrCsv)]
pub fn js_parse_csr_csv(
    csv: &str,
    directed: Option<bool>,
    sanitize: Option<bool>,
) -> Result<JsCsrGraph, JsError> {
    let g32 = parse_csr_csv::<f32>(csv, directed.unwrap_or(false), sanitize.unwrap_or(false))
        .map_err(core_err_to_js)?;
    Ok(JsCsrGraph::from_inner(g32))
}
