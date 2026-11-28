"""Cross-target Python <-> WASM parity checks for 1+2-hop CSR, kNN, perplexities, and probabilities."""

from __future__ import annotations

import pathlib
import time
from typing import Any

from parity.common import (
    check_array_close,
    make_grid_star_graph_spec,
    resolve_wasm_entries,
    run_node_json,
)


def run_cross_target_neighborhood_knn_parity(
    eg: Any, wasm_pkg: pathlib.Path
) -> list[dict[str, Any]]:
    """Verify Python <-> WASM parity across 1+2-hop CSR, kNN, perplexities, and fuzzy sets."""
    if not hasattr(eg, "extract_1hop_2hop_csr"):
        return []
    if resolve_wasm_entries(wasm_pkg) is None:
        return []

    t0 = time.perf_counter()
    n, edges = make_grid_star_graph_spec()
    g = eg.CsrGraph.from_edges(n, edges, directed=False, sanitize=True)

    jl_emb = eg.chebyshev_dog_jl_embedding(
        g, degree=25, jl_dim=12, seed=42, degree_normalize=True, eps_psd=1e-4
    )
    hop1, hop2 = eg.extract_1hop_2hop_csr(g)
    sparse_pairs = eg.extract_sparse_1hop_2hop_pairs(g, max_2hop_per_node=8, seed=42)
    sparse_pairs_zero = eg.extract_sparse_1hop_2hop_pairs(g, max_2hop_per_node=0, seed=42)
    pure_res = eg.build_pure_2hop_knn(g)
    pure_knn = pure_res.knn
    k2_deg = pure_res.k2_deg
    pure_stats = pure_res.stats
    assert k2_deg is not None
    sizes_2hop = eg.compute_2hop_neighborhood_sizes(g)
    node_perps = eg.compute_nodewise_2hop_perplexities(g)
    med_perp = float(eg.compute_median_2hop_perplexity(g))
    hybrid_res = eg.build_hybrid_adaptive_knn(g, jl_emb, node_perps)
    hybrid_knn = hybrid_res.knn
    hybrid_stats = hybrid_res.stats
    bfs_knn = eg.partial_bfs_knn(g, k=6, seed=42)
    probs = eg.compute_knn_probabilities(hybrid_knn, node_perps)
    probs_rn = eg.compute_knn_probabilities(hybrid_knn, node_perps, row_normalize=True)
    fuzzy = eg.compute_fuzzy_simplicial_set(hybrid_knn)
    ab = eg.find_ab_params(0.1, 1.0)
    ht_gamma = float(eg.compute_horvitz_thompson_gamma(1.0, 1600, 16))

    spec_payload = {
        "n": n,
        "edges": [int(v) for edge in edges for v in edge],
    }

    node_body = """
if (typeof mod.extract1hop2hopCsr !== 'function') {
  console.log(JSON.stringify({ neighborhood_available: false }));
  process.exit(0);
}

const g = mod.CsrGraph.fromEdgeList(spec.n, new Uint32Array(spec.edges));
const jlEmb = mod.chebyshevDogJlEmbedding(g, { jlDim: 12, degree: 25, seed: 42, degreeNormalize: true, epsPsd: 1e-4 });

const twoHop = mod.extract1hop2hopCsr(g);
const h1 = twoHop.hop1();
const h2 = twoHop.hop2();
const sparsePairs = mod.extractSparse1hop2hopPairs(g, 8, 42);
const sparsePairsZero = mod.extractSparse1hop2hopPairs(g, 0, 42);
const pureRes = mod.buildPure2hopKnn(g);
const pureKnn = pureRes.knn();
const pureRt = mod.KnnGraph.fromCsr(pureKnn.numNodes, pureKnn.indptr(), pureKnn.indices(), pureKnn.distances());
const sizes2hop = Array.from(mod.compute2hopNeighborhoodSizes(g));
const nodePerps = mod.computeNodewise2hopPerplexities(g);
const medPerp = mod.computeMedian2hopPerplexity(g);
const hybRes = mod.buildHybridAdaptiveKnn(g, jlEmb, 12, nodePerps);
const hybKnn = hybRes.knn();
const bfsKnn = mod.partialBfsKnn(g, 6, 42);
const probs = mod.computeKnnProbabilities(hybKnn, nodePerps);
const probsRn = mod.computeKnnProbabilities(hybKnn, nodePerps, { rowNormalize: true });
const fuzzy = mod.computeFuzzySimplicialSet(hybKnn);
const ab = Array.from(mod.findAbParams(0.1, 1.0));
const htGamma = mod.computeHorvitzThompsonGamma(1.0, 1600, 16);

const out = {
  neighborhood_available: true,
  hop1: { indptr: Array.from(h1.indptr()), indices: Array.from(h1.indices()) },
  hop2: { indptr: Array.from(h2.indptr()), indices: Array.from(h2.indices()) },
  pure: {
    indptr: Array.from(pureRt.indptr()),
    indices: Array.from(pureRt.indices()),
    distances: Array.from(pureRt.distances()),
    k2Deg: Array.from(pureRes.k2Deg()),
    avgDeg1: pureRes.avgDeg1,
    avgDeg2: pureRes.avgDeg2,
    nnz: pureRes.nnz,
  },
  sizes2hop,
  nodePerps: Array.from(nodePerps),
  medPerp,
  hybrid: {
    indptr: Array.from(hybKnn.indptr()),
    indices: Array.from(hybKnn.indices()),
    distances: Array.from(hybKnn.distances()),
    avgDeg1: hybRes.avgDeg1,
    avgDeg2: hybRes.avgDeg2,
    avgRowK: hybRes.avgRowK,
    nnz: hybRes.nnz,
    nFilledResidual: hybRes.nFilledResidual,
    nPure2hopRows: hybRes.nPure2hopRows,
  },
  bfsKnn: {
    indptr: Array.from(bfsKnn.indptr()),
    indices: Array.from(bfsKnn.indices()),
    distances: Array.from(bfsKnn.distances()),
  },
  probs: {
    indptr: Array.from(probs.indptr()),
    indices: Array.from(probs.indices()),
    values: Array.from(probs.values()),
  },
  probsRn: {
    indptr: Array.from(probsRn.indptr()),
    indices: Array.from(probsRn.indices()),
    values: Array.from(probsRn.values()),
  },
  sparsePairs: {
    u1: Array.from(sparsePairs.u1),
    v1: Array.from(sparsePairs.v1),
    u2: Array.from(sparsePairs.u2),
    v2: Array.from(sparsePairs.v2),
  },
  sparsePairsZero: {
    u1: Array.from(sparsePairsZero.u1),
    v1: Array.from(sparsePairsZero.v1),
    u2: Array.from(sparsePairsZero.u2),
    v2: Array.from(sparsePairsZero.v2),
  },
  fuzzy: {
    indptr: Array.from(fuzzy.indptr()),
    indices: Array.from(fuzzy.indices()),
    values: Array.from(fuzzy.values()),
  },
  ab,
  htGamma,
};

fuzzy.free?.(); probsRn.free?.(); probs.free?.(); sparsePairsZero.free?.(); sparsePairs.free?.();
bfsKnn.free?.(); hybKnn.free?.(); hybRes.free?.(); pureRt.free?.(); pureKnn.free?.(); pureRes.free?.();
h1.free?.(); h2.free?.(); twoHop.free?.(); g.free?.();

console.log(JSON.stringify(out));
"""
    ws = run_node_json(wasm_pkg, node_body, spec_payload)
    if not ws.get("neighborhood_available", False):
        return [{
            "case": "cross_target_neighborhood_knn_py_vs_wasm",
            "status": "skipped_no_wasm_neighborhood",
            "elapsed_sec": round(time.perf_counter() - t0, 4),
        }]

    max_f64_diff = 0.0

    def check_f64(label: str, a: Any, b: Any, tol: float = 1e-12) -> None:
        nonlocal max_f64_diff
        max_f64_diff = max(max_f64_diff, check_array_close(label, a, b, tol))

    assert list(hop1.indptr()) == ws["hop1"]["indptr"] and list(hop1.indices()) == ws["hop1"]["indices"]
    assert list(hop2.indptr()) == ws["hop2"]["indptr"] and list(hop2.indices()) == ws["hop2"]["indices"]
    assert list(pure_knn.indptr()) == ws["pure"]["indptr"] and list(pure_knn.indices()) == ws["pure"]["indices"]
    check_f64("build_pure_2hop_knn/distances", pure_knn.distances(), ws["pure"]["distances"])
    assert list(k2_deg) == ws["pure"]["k2Deg"] == ws["sizes2hop"] == list(sizes_2hop)
    assert pure_stats["nnz"] == ws["pure"]["nnz"]
    check_f64("build_pure_2hop_knn/avg_deg1", [pure_stats["avg_deg1"]], [ws["pure"]["avgDeg1"]])
    check_f64("build_pure_2hop_knn/avg_deg2", [pure_stats["avg_deg2"]], [ws["pure"]["avgDeg2"]])

    check_f64("compute_nodewise_2hop_perplexities", node_perps, ws["nodePerps"])
    check_f64("compute_median_2hop_perplexity", [med_perp], [ws["medPerp"]])

    hyb_ptr = list(hybrid_knn.indptr())
    assert hyb_ptr == ws["hybrid"]["indptr"]
    hyb_py_idx = list(hybrid_knn.indices())
    hyb_py_dst = list(hybrid_knn.distances())
    hyb_ws_idx = ws["hybrid"]["indices"]
    hyb_ws_dst = ws["hybrid"]["distances"]
    for row_i in range(n):
        s, e = hyb_ptr[row_i], hyb_ptr[row_i + 1]
        py_pairs = sorted(zip(hyb_py_idx[s:e], hyb_py_dst[s:e]))
        ws_pairs = sorted(zip(hyb_ws_idx[s:e], hyb_ws_dst[s:e]))
        assert [c for c, _ in py_pairs] == [c for c, _ in ws_pairs], (
            f"build_hybrid_adaptive_knn/row_{row_i}_indices mismatch"
        )
        check_f64(
            f"build_hybrid_adaptive_knn/row_{row_i}_distances",
            [d for _, d in py_pairs],
            [d for _, d in ws_pairs],
        )
    assert hybrid_stats["nnz"] == ws["hybrid"]["nnz"]
    assert hybrid_stats["n_filled_residual"] == ws["hybrid"]["nFilledResidual"]
    assert hybrid_stats["n_pure_2hop_rows"] == ws["hybrid"]["nPure2hopRows"]
    check_f64("build_hybrid_adaptive_knn/avg_row_k", [hybrid_stats["avg_row_k"]], [ws["hybrid"]["avgRowK"]])

    assert list(bfs_knn.indptr()) == ws["bfsKnn"]["indptr"] and list(bfs_knn.indices()) == ws["bfsKnn"]["indices"]
    check_f64("partial_bfs_knn/distances", bfs_knn.distances(), ws["bfsKnn"]["distances"])

    assert list(probs.indptr()) == ws["probs"]["indptr"] and list(probs.indices()) == ws["probs"]["indices"]
    check_f64("compute_knn_probabilities/values", probs.values(), ws["probs"]["values"])
    assert list(probs_rn.indptr()) == ws["probsRn"]["indptr"] and list(probs_rn.indices()) == ws["probsRn"]["indices"]
    check_f64("compute_knn_probabilities_rn/values", probs_rn.values(), ws["probsRn"]["values"])

    assert list(sparse_pairs.u1()) == ws["sparsePairs"]["u1"]
    assert list(sparse_pairs.v1()) == ws["sparsePairs"]["v1"]
    assert list(sparse_pairs.u2()) == ws["sparsePairs"]["u2"]
    assert list(sparse_pairs.v2()) == ws["sparsePairs"]["v2"]
    assert list(sparse_pairs_zero.u1()) == ws["sparsePairsZero"]["u1"]
    assert list(sparse_pairs_zero.v1()) == ws["sparsePairsZero"]["v1"]
    assert list(sparse_pairs_zero.u2()) == ws["sparsePairsZero"]["u2"] == []
    assert list(sparse_pairs_zero.v2()) == ws["sparsePairsZero"]["v2"] == []

    assert list(fuzzy.indptr()) == ws["fuzzy"]["indptr"] and list(fuzzy.indices()) == ws["fuzzy"]["indices"]
    check_f64("compute_fuzzy_simplicial_set/values", fuzzy.values(), ws["fuzzy"]["values"])

    check_f64("find_ab_params", ab, ws["ab"])
    check_f64("compute_horvitz_thompson_gamma", [ht_gamma], [ws["htGamma"]])

    return [{
        "case": "cross_target_neighborhood_knn_py_vs_wasm",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
        "max_f64_diff": round(max_f64_diff, 15),
    }]
