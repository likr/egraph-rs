//! WebAssembly wrapper types for $k$-NN graphs, sparse probabilities, and 1+2-hop CSR results.

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, parse_non_negative_usize};
use egraph::core::CsrGraph;
use egraph::neighborhood::{KnnGraph, KnnStats, Sparse12HopPairs, SparseProbabilities};
use wasm_bindgen::prelude::*;

/// Variable-length Compressed Sparse Row (CSR) $k$-nearest-neighbor graph in `f64`.
#[wasm_bindgen(js_name = KnnGraph)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsKnnGraph {
    inner: KnnGraph,
}

impl JsKnnGraph {
    /// Returns a shared reference to the underlying `egraph_neighborhood::KnnGraph`.
    pub fn inner(&self) -> &KnnGraph {
        &self.inner
    }

    pub(crate) fn from_inner(inner: KnnGraph) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen(js_class = KnnGraph)]
impl JsKnnGraph {
    /// Constructs a validated `KnnGraph` from CSR `numNodes`, `indptr`, `indices`, and `distances`.
    #[wasm_bindgen(js_name = fromCsr)]
    pub fn from_csr(
        num_nodes: f64,
        indptr: &[u32],
        indices: &[u32],
        distances: &[f64],
    ) -> Result<JsKnnGraph, JsValue> {
        let n = parse_non_negative_usize(num_nodes, "numNodes")?;
        let inner =
            KnnGraph::new_validated(n, indptr.to_vec(), indices.to_vec(), distances.to_vec())
                .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Returns the number of vertices in the $k$-NN graph.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Returns the total number of stored neighbor entries.
    #[wasm_bindgen(getter)]
    pub fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    /// Returns a copy of the CSR `indptr` row-offset array (`length === numNodes + 1`).
    pub fn indptr(&self) -> Result<Vec<u32>, JsValue> {
        Ok(self.inner.indptr().to_vec())
    }

    /// Returns a copy of the flat `Uint32Array` neighbor indices (`length === nnz`).
    pub fn indices(&self) -> Vec<u32> {
        self.inner.indices().to_vec()
    }

    /// Returns a copy of the flat `Float64Array` distances (`length === nnz`).
    pub fn distances(&self) -> Vec<f64> {
        self.inner.distances().to_vec()
    }
}

/// Canonical sorted symmetric CSR sparse joint probability matrix $P$.
#[wasm_bindgen(js_name = SparseProbabilities)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsSparseProbabilities {
    inner: SparseProbabilities,
}

impl JsSparseProbabilities {
    /// Returns a shared reference to the underlying `egraph_neighborhood::SparseProbabilities`.
    pub fn inner(&self) -> &SparseProbabilities {
        &self.inner
    }

    pub(crate) fn from_inner(inner: SparseProbabilities) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen(js_class = SparseProbabilities)]
impl JsSparseProbabilities {
    /// Constructs a validated `SparseProbabilities` from CSR `numNodes`, `indptr`, `indices`, and `values`.
    #[wasm_bindgen(js_name = fromCsr)]
    pub fn from_csr(
        num_nodes: f64,
        indptr: &[u32],
        indices: &[u32],
        values: &[f64],
    ) -> Result<JsSparseProbabilities, JsValue> {
        let n = parse_non_negative_usize(num_nodes, "numNodes")?;
        let inner =
            SparseProbabilities::from_csr(n, indptr.to_vec(), indices.to_vec(), values.to_vec())
                .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Returns the number of vertices in the probability matrix.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Returns the total number of stored probability entries.
    #[wasm_bindgen(getter)]
    pub fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    /// Returns a copy of the CSR `indptr` row-offset array (`length === numNodes + 1`).
    pub fn indptr(&self) -> Result<Vec<u32>, JsValue> {
        Ok(self.inner.indptr().to_vec())
    }

    /// Returns a copy of the flat `Uint32Array` neighbor column indices (`length === nnz`).
    pub fn indices(&self) -> Vec<u32> {
        self.inner.indices().to_vec()
    }

    /// Returns a copy of the flat `Float64Array` symmetric probability values (`length === nnz`).
    pub fn values(&self) -> Vec<f64> {
        self.inner.values().to_vec()
    }
}

/// Extracted canonical undirected 1-hop and hub-safe capped 2-hop vertex pair arrays (`u < v`).
#[wasm_bindgen(js_name = Sparse12HopPairs)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsSparse12HopPairs {
    inner: Sparse12HopPairs,
}

impl JsSparse12HopPairs {
    pub(crate) fn from_inner(inner: Sparse12HopPairs) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen(js_class = Sparse12HopPairs)]
impl JsSparse12HopPairs {
    /// Returns the number of unique undirected 1-hop pairs (`u1 < v1`).
    #[wasm_bindgen(getter, js_name = num1hop)]
    pub fn num_1hop(&self) -> usize {
        self.inner.u1().len()
    }

    /// Returns the number of unique undirected 2-hop pairs (`u2 < v2`).
    #[wasm_bindgen(getter, js_name = num2hop)]
    pub fn num_2hop(&self) -> usize {
        self.inner.u2().len()
    }

    /// Returns a copy of 1-hop source vertex indices (`Uint32Array`).
    #[wasm_bindgen(getter)]
    pub fn u1(&self) -> Vec<u32> {
        self.inner.u1().to_vec()
    }

    /// Returns a copy of 1-hop target vertex indices (`Uint32Array`).
    #[wasm_bindgen(getter)]
    pub fn v1(&self) -> Vec<u32> {
        self.inner.v1().to_vec()
    }

    /// Returns a copy of 2-hop source vertex indices (`Uint32Array`).
    #[wasm_bindgen(getter)]
    pub fn u2(&self) -> Vec<u32> {
        self.inner.u2().to_vec()
    }

    /// Returns a copy of 2-hop target vertex indices (`Uint32Array`).
    #[wasm_bindgen(getter)]
    pub fn v2(&self) -> Vec<u32> {
        self.inner.v2().to_vec()
    }
}

/// Disjoint unweighted 1-hop and pure 2-hop CSR graphs returned by `extract1hop2hopCsr`.
#[wasm_bindgen(js_name = TwoHopCsr)]
#[derive(Debug, Clone)]
pub struct JsTwoHopCsr {
    hop1: CsrGraph<f32>,
    hop2: CsrGraph<f32>,
}

impl JsTwoHopCsr {
    pub(crate) fn from_parts(hop1: CsrGraph<f32>, hop2: CsrGraph<f32>) -> Self {
        Self { hop1, hop2 }
    }
}

#[wasm_bindgen(js_class = TwoHopCsr)]
impl JsTwoHopCsr {
    /// Returns a cloned handle to the canonical unweighted 1-hop `CsrGraph`.
    pub fn hop1(&self) -> JsCsrGraph {
        JsCsrGraph::from_inner(self.hop1.clone())
    }

    /// Returns a cloned handle to the disjoint pure 2-hop `CsrGraph`.
    pub fn hop2(&self) -> JsCsrGraph {
        JsCsrGraph::from_inner(self.hop2.clone())
    }
}

/// Pure 1+2-hop CSR `KnnGraph` and summary degree statistics returned by `buildPure2hopKnn`.
#[wasm_bindgen(js_name = TwoHopKnnResult)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsTwoHopKnnResult {
    knn: KnnGraph,
    k2_deg: Vec<u32>,
    stats: KnnStats,
}

impl JsTwoHopKnnResult {
    pub(crate) fn from_parts(knn: KnnGraph, k2_deg: Vec<u32>, stats: KnnStats) -> Self {
        Self { knn, k2_deg, stats }
    }
}

#[wasm_bindgen(js_class = TwoHopKnnResult)]
impl JsTwoHopKnnResult {
    /// Returns a cloned `KnnGraph` containing exact 1-hop (`1.0`) and 2-hop (`2.0`) neighbors.
    pub fn knn(&self) -> JsKnnGraph {
        JsKnnGraph {
            inner: self.knn.clone(),
        }
    }

    /// Returns per-node 2-hop neighborhood sizes $|B_2(u) \setminus \{u\}|$ (`Uint32Array`).
    #[wasm_bindgen(js_name = k2Deg)]
    pub fn k2_deg(&self) -> Vec<u32> {
        self.k2_deg.clone()
    }

    /// Returns the mean 1-hop degree across all vertices.
    #[wasm_bindgen(getter, js_name = avgDeg1)]
    pub fn avg_deg1(&self) -> f64 {
        self.stats.avg_deg1
    }

    /// Returns the mean 1+2-hop neighborhood size across all vertices.
    #[wasm_bindgen(getter, js_name = avgDeg2)]
    pub fn avg_deg2(&self) -> f64 {
        self.stats.avg_deg2
    }

    /// Returns the total number of stored neighbor entries.
    #[wasm_bindgen(getter)]
    pub fn nnz(&self) -> usize {
        self.stats.nnz
    }

    /// Returns the number of stored 1-hop directed neighbor entries.
    #[wasm_bindgen(getter, js_name = n1hop)]
    pub fn n_1hop(&self) -> usize {
        self.stats.n_1hop
    }

    /// Returns the number of stored 2-hop directed neighbor entries.
    #[wasm_bindgen(getter, js_name = n2hop)]
    pub fn n_2hop(&self) -> usize {
        self.stats.n_2hop
    }

    /// Returns the number of vertices whose 2-hop expansion was truncated by `max2hopPerNode`.
    #[wasm_bindgen(getter, js_name = nTruncated2hop)]
    pub fn n_truncated_2hop(&self) -> usize {
        self.stats.n_truncated_2hop
    }
}

/// Hybrid 1+2-hop + two-step reranked `KnnGraph` and diagnostics returned by `buildHybridAdaptiveKnn`.
#[wasm_bindgen(js_name = HybridKnnResult)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsHybridKnnResult {
    knn: KnnGraph,
    stats: KnnStats,
}

impl JsHybridKnnResult {
    pub(crate) fn from_parts(knn: KnnGraph, stats: KnnStats) -> Self {
        Self { knn, stats }
    }
}

#[wasm_bindgen(js_class = HybridKnnResult)]
impl JsHybridKnnResult {
    /// Returns a cloned hybrid adaptive `KnnGraph`.
    pub fn knn(&self) -> JsKnnGraph {
        JsKnnGraph {
            inner: self.knn.clone(),
        }
    }

    /// Returns the mean 1-hop degree across all vertices.
    #[wasm_bindgen(getter, js_name = avgDeg1)]
    pub fn avg_deg1(&self) -> f64 {
        self.stats.avg_deg1
    }

    /// Returns the mean 1+2-hop neighborhood size across all vertices.
    #[wasm_bindgen(getter, js_name = avgDeg2)]
    pub fn avg_deg2(&self) -> f64 {
        self.stats.avg_deg2
    }

    /// Returns the mean output row degree in the constructed `KnnGraph`.
    #[wasm_bindgen(getter, js_name = avgRowK)]
    pub fn avg_row_k(&self) -> f64 {
        self.stats.avg_row_k
    }

    /// Returns the total number of stored neighbor entries.
    #[wasm_bindgen(getter)]
    pub fn nnz(&self) -> usize {
        self.stats.nnz
    }

    /// Returns the number of rows augmented with non-2-hop residual candidates.
    #[wasm_bindgen(getter, js_name = nFilledResidual)]
    pub fn n_filled_residual(&self) -> usize {
        self.stats.n_filled_residual
    }

    /// Returns the number of rows composed solely of 1-hop and 2-hop neighbors.
    #[wasm_bindgen(getter, js_name = nPure2hopRows)]
    pub fn n_pure_2hop_rows(&self) -> usize {
        self.stats.n_pure_2hop_rows
    }
}
