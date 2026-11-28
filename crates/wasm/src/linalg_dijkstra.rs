//! WebAssembly bindings for weighted shortest-path queries (`DijkstraWorkspace`,
//! `dijkstraSingleSource`, `allPairsDijkstra`).

use egraph::linalg::{DijkstraWorkspace, all_pairs_dijkstra, dijkstra_single_source};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, parse_non_negative_usize};

/// Reusable epoch-stamped single-source Dijkstra workspace over `CsrGraph`.
#[wasm_bindgen(js_name = DijkstraWorkspace)]
#[derive(Debug, Clone)]
pub struct JsDijkstraWorkspace {
    inner: DijkstraWorkspace,
}

#[wasm_bindgen(js_class = DijkstraWorkspace)]
impl JsDijkstraWorkspace {
    /// Creates a `DijkstraWorkspace` pre-sized for `numNodes` vertices.
    #[wasm_bindgen(constructor)]
    pub fn new(num_nodes: Option<f64>) -> Result<JsDijkstraWorkspace, JsValue> {
        let n = num_nodes
            .map(|v| parse_non_negative_usize(v, "numNodes"))
            .transpose()?
            .unwrap_or(0);
        Ok(Self {
            inner: DijkstraWorkspace::new(n),
        })
    }

    /// Ensures internal scratch capacity covers at least `numNodes` vertices.
    #[wasm_bindgen(js_name = ensureCapacity)]
    pub fn ensure_capacity(&mut self, num_nodes: f64) -> Result<(), JsValue> {
        let n = parse_non_negative_usize(num_nodes, "numNodes")?;
        self.inner.ensure_capacity(n);
        Ok(())
    }

    /// Computes single-source shortest-path distances (`Float64Array`, `Infinity` for unreachable)
    /// from `source` on `graph`, reusing internal workspace allocations.
    #[wasm_bindgen(js_name = dijkstraFrom)]
    pub fn dijkstra_from(&mut self, graph: &JsCsrGraph, source: f64) -> Result<Vec<f64>, JsValue> {
        let src = parse_non_negative_usize(source, "source")?;
        let n = graph.inner().num_nodes();
        let mut out = vec![f64::INFINITY; n];
        self.inner
            .dijkstra_from(graph.inner(), src, &mut out)
            .map_err(core_err_to_js)?;
        Ok(out)
    }
}

/// Computes weighted single-source Dijkstra shortest-path distances from `source` (`Float64Array`).
#[wasm_bindgen(js_name = dijkstraSingleSource)]
pub fn js_dijkstra_single_source(graph: &JsCsrGraph, source: f64) -> Result<Vec<f64>, JsValue> {
    let src = parse_non_negative_usize(source, "source")?;
    dijkstra_single_source(graph.inner(), src).map_err(core_err_to_js)
}

/// Computes the flat row-major `(numNodes, numNodes)` all-pairs Dijkstra distance matrix (`Float64Array`).
#[wasm_bindgen(js_name = allPairsDijkstra)]
pub fn js_all_pairs_dijkstra(graph: &JsCsrGraph) -> Result<Vec<f64>, JsValue> {
    all_pairs_dijkstra(graph.inner()).map_err(core_err_to_js)
}
