"""Cross-target Python <-> WASM parity checks for MDS/spectral embeddings and kNN probabilities."""

from __future__ import annotations

import pathlib
import time
from typing import Any

from parity.common import (
    check_array_close,
    make_grid_edges,
    resolve_wasm_entries,
    run_node_json,
)


def run_cross_target_embeddings_knn_probs_parity(
    eg: Any, wasm_pkg: pathlib.Path
) -> list[dict[str, Any]]:
    """Verify Python <-> WASM parity across CMDS, RD-MDS, resistance/diffusion embeddings, and kNN probs."""
    import numpy as np

    if not hasattr(eg, "classical_mds_2d"):
        return []
    if resolve_wasm_entries(wasm_pkg) is None:
        return []

    t0 = time.perf_counter()

    g16_edges = make_grid_edges(4, 4)
    g16 = eg.CsrGraph.from_edges(16, g16_edges, directed=False, sanitize=True)
    cmds_py = eg.classical_mds_2d(g16, dtype="float32")
    eg.normalize_layout_2d(cmds_py, target_std=1.0, seed=42)
    rob_py = eg.classical_mds_2d(g16, dtype="float32")
    eg.robust_normalize_layout_2d(rob_py, target_std=1.5, quantile=0.95, seed=42)
    rd_py = eg.rd_mds(g16, dims=3, seed=42, dtype="float32")
    res_coords_py, res_lams_py = eg.compute_resistance_embedding(g16, dim=3, seed=42)
    diff_coords_py, diff_lams_py, diff_t_py = eg.compute_diffusion_embedding(g16, dim=3, p=0.5, seed=42)
    rep_in = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]], dtype=np.float64)
    rep_py = eg.representative_mds(rep_in, dims=2, dtype="float32")
    rerank_py = eg.build_rerank_knn(diff_coords_py, k=4, coarse_dim=2, rerank_factor=2)
    capped_res_py = eg.build_capped_12hop_knn(g16, max_2hop_per_node=4, seed=42)
    capped_knn_py = capped_res_py.knn
    capped_k2_py = capped_res_py.k2_deg
    assert capped_k2_py is not None
    row_probs_py = eg.graph_row_normalized_probabilities(g16)
    row_probs12_py = eg.graph_12hop_row_normalized_probabilities(g16, w2=0.5, max_2hop_per_node=4, degree_beta=0.35, seed=42)
    sampled_sns_strat_py = float(eg.sampled_scale_normalized_stress(g16, cmds_py, num_pivots=4, seed=42))
    sym_probs_py = eg.symmetrize_row_normalized(3, [0, 1, 2, 3], [1, 2, 0], [1.0, 1.0, 1.0])
    prio_py = eg.build_gumap_12hop_priority_csr(3, [0], [1], [1.0], [0, 1], [2, 2], [2.0, 2.0], tier2_margin=1.1)
    f1_py, f2_py, f3_py, floor_py = eg.apply_relative_distance_floor([1e-6, 1.0], [2.0], [4.0], eps_dist=1e-3, min_ratio=0.1)

    spec_payload = {
        "g16_edges": [int(v) for edge in g16_edges for v in edge],
        "diff_coords": [float(v) for v in diff_coords_py.ravel()],
        "diff_dim": int(diff_coords_py.shape[1]),
    }

    node_body = """
const g16 = mod.CsrGraph.fromEdgeList(16, new Uint32Array(spec.g16_edges));
const cmds = mod.classicalMds2d(g16);
mod.normalizeLayout2d(cmds, 1.0, 42);
const cmdsCoords = Array.from(cmds.coordsView());
const rob = mod.classicalMds2d(g16);
mod.robustNormalizeLayout2d(rob, { targetStd: 1.5, quantile: 0.95, seed: 42 });
const robCoords = Array.from(rob.coordsView());
const rd = mod.rdMds(g16, { dims: 3, eigEps: 1e-10, tol: 1e-6, maxIterations: 1000, seed: 42 });
const rdCoords = Array.from(rd.copyCoords());
const resEmb = mod.computeResistanceEmbedding(g16, { dim: 3, tol: 1e-4, maxIterations: 300, seed: 42, eigEps: 1e-10 });
const diffEmb = mod.computeDiffusionEmbedding(g16, { dim: 3, p: 0.5, t: null, tol: 1e-4, maxIterations: 300, seed: 42 });
const repNd = mod.representativeMds(new Float64Array([0, 3, 1, 2, 2, 1, 3, 0]), 4, 2, 2);
const rerank = mod.buildRerankKnn(new Float64Array(spec.diff_coords), 16, spec.diff_dim, 4, { coarseDim: 2, rerankFactor: 2 });
const cappedKnn = mod.buildCapped12hopKnn(g16, 4, 42);
const rowProbs = mod.graphRowNormalizedProbabilities(g16);
const rowProbs12 = mod.graph12hopRowNormalizedProbabilities(g16, { w2: 0.5, max2hopPerNode: 4, degreeBeta: 0.35, seed: 42 });
const sampledSnsStrat = mod.sampledScaleNormalizedStress(g16, cmds, { numPivots: 4, seed: 42 });
const symProbs = mod.symmetrizeRowNormalized(3, new Uint32Array([0, 1, 2, 3]), new Uint32Array([1, 2, 0]), new Float64Array([1, 1, 1]));
const prio = mod.buildGumap12hopPriorityCsr(3, new Uint32Array([0]), new Uint32Array([1]), new Float64Array([1]), new Uint32Array([0, 1]), new Uint32Array([2, 2]), new Float64Array([2, 2]), { tier2Margin: 1.1 });
const d1 = new Float64Array([1e-6, 1.0]);
const d2 = new Float64Array([2.0]);
const d3 = new Float64Array([4.0]);
const floorVal = mod.applyRelativeDistanceFloor(d1, d2, d3, { epsDist: 1e-3, minRatio: 0.1 });

const out = {
  cmdsCoords, robCoords, rdCoords, sampledSnsStrat,
  res: { lams: Array.from(resEmb.lambdas()), coords: Array.from(resEmb.coords()) },
  diff: { tUsed: diffEmb.tUsed, lams: Array.from(diffEmb.lambdas()), coords: Array.from(diffEmb.coords()) },
  repCoords: Array.from(repNd.copyCoords()),
  rerank: { indptr: Array.from(rerank.indptr()), indices: Array.from(rerank.indices()), dists: Array.from(rerank.distances()) },
  cappedKnn: { k2Deg: Array.from(cappedKnn.k2Deg()), indptr: Array.from(cappedKnn.knn().indptr()), indices: Array.from(cappedKnn.knn().indices()), dists: Array.from(cappedKnn.knn().distances()) },
  rowProbs: { indptr: Array.from(rowProbs.indptr()), indices: Array.from(rowProbs.indices()), values: Array.from(rowProbs.values()) },
  rowProbs12: { indptr: Array.from(rowProbs12.indptr()), indices: Array.from(rowProbs12.indices()), values: Array.from(rowProbs12.values()) },
  symProbs: { indptr: Array.from(symProbs.indptr()), indices: Array.from(symProbs.indices()), values: Array.from(symProbs.values()) },
  prio: { indptr: Array.from(prio.indptr()), indices: Array.from(prio.indices()), dists: Array.from(prio.distances()) },
  floor: { d1: Array.from(d1), d2: Array.from(d2), d3: Array.from(d3), val: floorVal },
};

prio.free?.(); symProbs.free?.(); rowProbs12.free?.(); rowProbs.free?.();
cappedKnn.free?.(); rerank.free?.(); repNd.free?.(); diffEmb.free?.(); resEmb.free?.();
rd.free?.(); rob.free?.(); cmds.free?.(); g16.free?.();

console.log(JSON.stringify(out));
"""
    ws = run_node_json(wasm_pkg, node_body, spec_payload)

    max_f32_diff = 0.0
    max_f64_diff = 0.0

    def check_close(label: str, a: Any, b: Any, tol: float, is_f32: bool = True) -> None:
        nonlocal max_f32_diff, max_f64_diff
        diff = check_array_close(label, a, b, tol)
        if is_f32:
            max_f32_diff = max(max_f32_diff, diff)
        else:
            max_f64_diff = max(max_f64_diff, diff)

    check_close("sampled_scale_normalized_stress/degree_stratified_k4", [sampled_sns_strat_py], [ws["sampledSnsStrat"]], 1e-6)
    check_close("classical_mds_2d", cmds_py.coords(copy=True), ws["cmdsCoords"], 1e-5)
    check_close("robust_normalize_layout_2d", rob_py.coords(copy=True), ws["robCoords"], 1e-5)
    check_close("rd_mds", rd_py.coords(copy=True), ws["rdCoords"], 1e-4)
    check_close("compute_resistance_embedding/lambdas", res_lams_py, ws["res"]["lams"], 1e-10, is_f32=False)
    check_close("compute_resistance_embedding/coords", res_coords_py, ws["res"]["coords"], 1e-10, is_f32=False)
    check_close("compute_diffusion_embedding/t_used", [diff_t_py], [ws["diff"]["tUsed"]], 1e-12, is_f32=False)
    check_close("compute_diffusion_embedding/lambdas", diff_lams_py, ws["diff"]["lams"], 1e-10, is_f32=False)
    check_close("compute_diffusion_embedding/coords", diff_coords_py, ws["diff"]["coords"], 1e-10, is_f32=False)
    check_close("representative_mds", rep_py.coords(copy=True), ws["repCoords"], 1e-5)

    rr_ptr = list(rerank_py.indptr())
    assert rr_ptr == ws["rerank"]["indptr"]
    for row_i in range(16):
        s, e = rr_ptr[row_i], rr_ptr[row_i + 1]
        py_pairs = sorted(zip(list(rerank_py.indices())[s:e], list(rerank_py.distances())[s:e]))
        ws_pairs = sorted(zip(ws["rerank"]["indices"][s:e], ws["rerank"]["dists"][s:e]))
        assert [c for c, _ in py_pairs] == [c for c, _ in ws_pairs], f"build_rerank_knn/row_{row_i} mismatch"
        check_close(f"build_rerank_knn/row_{row_i}_dists", [d for _, d in py_pairs], [d for _, d in ws_pairs], 1e-12, is_f32=False)

    assert list(row_probs_py.indptr()) == ws["rowProbs"]["indptr"]
    assert list(row_probs_py.indices()) == ws["rowProbs"]["indices"]
    check_close("graph_row_normalized_probabilities", row_probs_py.values(), ws["rowProbs"]["values"], 1e-12, is_f32=False)
    assert list(capped_k2_py) == ws["cappedKnn"]["k2Deg"]
    assert list(capped_knn_py.indptr()) == ws["cappedKnn"]["indptr"]
    assert list(capped_knn_py.indices()) == ws["cappedKnn"]["indices"]
    check_close("build_capped_12hop_knn", capped_knn_py.distances(), ws["cappedKnn"]["dists"], 1e-12, is_f32=False)
    assert list(row_probs12_py.indptr()) == ws["rowProbs12"]["indptr"]
    assert list(row_probs12_py.indices()) == ws["rowProbs12"]["indices"]
    check_close("graph_12hop_row_normalized_probabilities", row_probs12_py.values(), ws["rowProbs12"]["values"], 1e-12, is_f32=False)
    assert list(sym_probs_py.indptr()) == ws["symProbs"]["indptr"]
    assert list(sym_probs_py.indices()) == ws["symProbs"]["indices"]
    check_close("symmetrize_row_normalized", sym_probs_py.values(), ws["symProbs"]["values"], 1e-12, is_f32=False)
    assert list(prio_py.indptr()) == ws["prio"]["indptr"]
    assert list(prio_py.indices()) == ws["prio"]["indices"]
    check_close("build_gumap_12hop_priority_csr", prio_py.distances(), ws["prio"]["dists"], 1e-12, is_f32=False)
    check_close("apply_relative_distance_floor/d1", f1_py, ws["floor"]["d1"], 1e-12, is_f32=False)
    check_close("apply_relative_distance_floor/d2", f2_py, ws["floor"]["d2"], 1e-12, is_f32=False)
    check_close("apply_relative_distance_floor/d3", f3_py, ws["floor"]["d3"], 1e-12, is_f32=False)
    check_close("apply_relative_distance_floor/val", [floor_py], [ws["floor"]["val"]], 1e-12, is_f32=False)

    return [{
        "case": "cross_target_embeddings_knn_probs_py_vs_wasm",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
        "max_f32_diff": round(max_f32_diff, 12),
        "max_f64_diff": round(max_f64_diff, 12),
    }]
