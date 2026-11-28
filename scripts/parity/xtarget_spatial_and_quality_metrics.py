"""Cross-target Python <-> WASM parity checks for spatial utilities and drawing quality metrics."""

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


def run_cross_target_spatial_quality_parity(
    eg: Any, wasm_pkg: pathlib.Path
) -> list[dict[str, Any]]:
    """Verify Python <-> WASM parity across spatial CSR/Gabriel/RNG, crossings, NIG, and RTA."""
    import numpy as np

    if not hasattr(eg, "crosslessness"):
        return []
    if resolve_wasm_entries(wasm_pkg) is None:
        return []

    t0 = time.perf_counter()

    g_grid = eg.CsrGraph.from_edge_list(9, make_grid_edges(3, 3), directed=False, sanitize=True)
    grid_coords_f32 = np.array(
        [[float(c) + 0.03 * float(r), float(r) - 0.02 * float(c)] for r in range(3) for c in range(3)],
        dtype=np.float32,
    )
    l_grid = eg.Layout2d.from_coords(grid_coords_f32)
    del_csr_py = eg.delaunay_to_csr(l_grid)
    rect_coords_f32 = np.array([[0.0, 0.0], [0.5, 0.0], [5.0, 0.0]], dtype=np.float32)
    rect_hw = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    rect_ov_py = eg.rectangle_overlaps(eg.Layout2d.from_coords(rect_coords_f32), rect_hw, rect_hw)

    g_x = eg.CsrGraph.from_edges(4, [(0, 3), (1, 2)], directed=False, sanitize=True)
    x_coords_f32 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    l_x = eg.Layout2d.from_coords(x_coords_f32)
    ce_py = eg.crossing_edges(g_x, l_x)
    cet_py = eg.crossing_edges_torus(g_x, l_x)
    cs_py = eg.crossing_summary(g_x, l_x)
    gg_py = eg.gabriel_graph_edges(l_grid)
    rng_py = eg.relative_neighborhood_edges(l_grid)
    apsp_py = eg.all_pairs_bfs(g_grid, unit_edge_length=1.0)
    sssp_py = eg.bfs_single_source(g_grid, 0, unit_edge_length=2.0)
    sns_dist_py = float(eg.scale_normalized_stress_from_distances(l_grid, apsp_py))
    iel_unit_py = float(eg.ideal_edge_lengths_with_unit(g_grid, l_grid, 1.5))
    eval_iel_py = eg.evaluate_metrics(
        g_grid, l_grid, ["sns", "ideal_edge_lengths"], num_pivots=4, num_samples=64, seed=42, ideal_edge_length=1.5
    )
    eval_default_py = eg.evaluate_metrics(g_grid, l_grid, ["sampled_sns", "sampled_crossing_number"])
    sampled_sns_default_py = float(eg.sampled_scale_normalized_stress(g_grid, l_grid))
    sampled_cr_default_py = float(eg.sampled_crossing_number(g_grid, l_grid))
    l_grid_adj = eg.Layout2d.from_coords(grid_coords_f32 + np.array([0.05, -0.03], dtype=np.float32))
    pd_edges_py = eg.prism_distortion_with_edges(l_grid, l_grid_adj, gg_py)

    quality_comm_labels = [0, 0, 0, 0, 0, 1, 1, 1, 1]
    cl_py = float(eg.crosslessness(g_grid, l_grid))
    scl_py = float(eg.sampled_crosslessness(g_grid, l_grid, num_samples=100, seed=42))
    ma_py = float(eg.minimum_angle(g_grid, l_grid))
    ma_sum_py = eg.minimum_angle_summary(g_grid, l_grid)
    deg_py = eg.layout_degeneracy(g_grid, l_grid)
    nig_py = float(eg.adjusted_patch_nig(l_grid, quality_comm_labels, patch_radius=16, canvas_size=2048, pad_frac=0.04, seed=42))
    nig_ms_py = eg.adjusted_patch_nig_multiscale(l_grid, quality_comm_labels, radii=[16, 32], canvas_size=2048, pad_frac=0.04, seed=42)
    rta_py = float(eg.landmark_stratified_rta(g_grid, l_grid, num_landmarks=4, num_triplets=50, min_degree=1, seed=42))
    rta_det_py = eg.landmark_stratified_rta_detail(g_grid, l_grid, num_landmarks=4, num_triplets=50, min_degree=1, seed=42)
    rta_lm_ids = eg.select_maxmin_sp_pivots(g_grid, num_pivots=3, seed=42, min_degree=1)
    rta_apsp = eg.all_pairs_bfs(g_grid, 1.0)
    rta_tbl_f64 = np.where(np.isfinite(rta_apsp[rta_lm_ids]), rta_apsp[rta_lm_ids], np.inf).astype(np.float64)
    rta_tbl_py = eg.landmark_stratified_rta_from_landmarks(l_grid, rta_lm_ids, rta_tbl_f64, num_triplets=1000, seed=42)
    quality_eval_py = eg.evaluate_metrics(
        g_grid,
        l_grid,
        [
            "crosslessness",
            "sampled_crosslessness",
            "minimum_angle",
            "minimum_angle_a1",
            "layout_degeneracy",
            "adjusted_patch_nig",
            "cluster_area_faithfulness",
            "landmark_stratified_rta",
        ],
        num_samples=100,
        seed=42,
        labels=quality_comm_labels,
        num_landmarks=4,
        num_triplets=50,
        min_degree=1,
        patch_radius=16,
        canvas_size=2048,
        pad_frac=0.04,
    )

    g_dir4_py = eg.CsrGraph.from_edges(4, [(0, 1), (0, 3), (1, 3), (3, 0), (3, 1), (3, 2)], directed=True, sanitize=True)
    l_dir4_py = eg.Layout2d.from_coords(np.array([[0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0]], dtype=np.float32))
    cl_dir4_py = float(eg.crosslessness(g_dir4_py, l_dir4_py))
    scl_dir4_py = float(eg.sampled_crosslessness(g_dir4_py, l_dir4_py, num_samples=100, seed=42))
    assert cl_dir4_py == 0.0 and scl_dir4_py == 0.0

    g_rstar_py = eg.CsrGraph.from_edges(4, [(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0)], directed=True, sanitize=True)
    l_rstar_py = eg.Layout2d.from_coords(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32))
    cl_rstar_py = float(eg.crosslessness(g_rstar_py, l_rstar_py))
    assert cl_rstar_py == 1.0

    spec_payload = {
        "grid_coords": [float(v) for v in grid_coords_f32.ravel()],
        "rect_coords": [float(v) for v in rect_coords_f32.ravel()],
        "rect_hw": [float(v) for v in rect_hw],
        "x_coords": [float(v) for v in x_coords_f32.ravel()],
    }

    node_body = """
const g9Edges = [];
for (let r = 0; r < 3; r++) {
  for (let c = 0; c < 3; c++) {
    const u = r * 3 + c;
    if (c + 1 < 3) g9Edges.push(u, u + 1);
    if (r + 1 < 3) g9Edges.push(u, u + 3);
  }
}
const gGrid = mod.CsrGraph.fromEdgeList(9, new Uint32Array(g9Edges));
const lGrid = mod.Layout2d.fromCoords(new Float32Array(spec.grid_coords));
const delCsr = mod.delaunayToCsr(lGrid);
const lRect = mod.Layout2d.fromCoords(new Float32Array(spec.rect_coords));
const rectOv = Array.from(mod.rectangleOverlaps(lRect, new Float32Array(spec.rect_hw), new Float32Array(spec.rect_hw)));

const gX = mod.CsrGraph.fromEdgeList(4, new Uint32Array([0, 3, 1, 2]));
const lX = mod.Layout2d.fromCoords(new Float32Array(spec.x_coords));
const ce = mod.crossingEdges(gX, lX);
const cet = mod.crossingEdgesTorus(gX, lX);
const cs = mod.crossingSummary(gX, lX);
const gg = Array.from(mod.gabrielGraphEdges(lGrid));
const rngE = Array.from(mod.relativeNeighborhoodEdges(lGrid));
const apsp = mod.allPairsBfs(gGrid, 1.0);
const sssp = Array.from(mod.bfsSingleSource(gGrid, 0, 2.0));
const snsDist = mod.scaleNormalizedStressFromDistances(lGrid, apsp);
const ielUnit = mod.idealEdgeLengthsWithUnit(gGrid, lGrid, 1.5);
const evalRep = mod.evaluateMetrics(gGrid, lGrid, ['sns', 'idealEdgeLengths'], {
  numPivots: 4, numSamples: 64, seed: 42, idealEdgeLength: 1.5,
});
const evalDefRep = mod.evaluateMetrics(gGrid, lGrid, ['sampledScaleNormalizedStress', 'sampledCrossingNumber']);
const sampledSnsDef = mod.sampledScaleNormalizedStress(gGrid, lGrid);
const sampledCrDef = mod.sampledCrossingNumber(gGrid, lGrid);
const lGridAdj = mod.Layout2d.fromCoords(new Float32Array(spec.grid_coords.map((v, i) => v + (i % 2 === 0 ? 0.05 : -0.03))));
const pdEdges = mod.prismDistortionWithEdges(lGrid, lGridAdj, new Uint32Array(gg));

const qComm = new Uint32Array([0, 0, 0, 0, 0, 1, 1, 1, 1]);
const qCl = mod.crosslessness(gGrid, lGrid);
const qScl = mod.sampledCrosslessness(gGrid, lGrid, { numSamples: 100, seed: 42 });
const qMa = mod.minimumAngle(gGrid, lGrid);
const qMaSum = mod.minimumAngleSummary(gGrid, lGrid);
const qDeg = mod.layoutDegeneracy(gGrid, lGrid);
const qNig = mod.adjustedPatchNig(lGrid, qComm, { patchRadius: 16, canvasSize: 2048, padFrac: 0.04, seed: 42 });
const qNigMs = Array.from(mod.adjustedPatchNigMultiscale(lGrid, qComm, { radii: [16, 32], canvasSize: 2048, padFrac: 0.04, seed: 42 }));
const qRta = mod.landmarkStratifiedRta(gGrid, lGrid, { numLandmarks: 4, numTriplets: 50, minDegree: 1, seed: 42 });
const qRtaDet = mod.landmarkStratifiedRtaDetail(gGrid, lGrid, { numLandmarks: 4, numTriplets: 50, minDegree: 1, seed: 42 });
const qLmIds = mod.selectMaxminSpPivots(gGrid, { numPivots: 3, seed: 42, minDegree: 1 });
const qTblFlat = [];
for (let r = 0; r < qLmIds.length; r++) {
  const rowOff = qLmIds[r] * 9;
  for (let c = 0; c < 9; c++) {
    const dv = apsp[rowOff + c];
    qTblFlat.push(Number.isFinite(dv) ? dv : Number.POSITIVE_INFINITY);
  }
}
const qRtaTbl = mod.landmarkStratifiedRtaFromLandmarks(lGrid, qLmIds, new Float64Array(qTblFlat), { numTriplets: 1000, seed: 42 });
const qEval = mod.evaluateMetrics(
  gGrid, lGrid,
  ['crosslessness', 'sampledCrosslessness', 'minimumAngle', 'minimumAngleA1', 'layoutDegeneracy', 'adjustedPatchNig', 'clusterAreaFaithfulness', 'landmarkStratifiedRta'],
  { numSamples: 100, seed: 42, labels: qComm, numLandmarks: 4, numTriplets: 50, minDegree: 1, patchRadius: 16, canvasSize: 2048, padFrac: 0.04 },
);

const gDir4 = mod.CsrGraph.fromEdgeList(4, new Uint32Array([0, 1, 0, 3, 1, 3, 3, 0, 3, 1, 3, 2]), undefined, true, true);
const lDir4 = mod.Layout2d.fromCoords(new Float32Array([0, 0, 2, 2, 0, 2, 2, 0]));
const clDir4 = mod.crosslessness(gDir4, lDir4);
const sclDir4 = mod.sampledCrosslessness(gDir4, lDir4, { numSamples: 100, seed: 42 });
const gRstar = mod.CsrGraph.fromEdgeList(4, new Uint32Array([0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 3, 0]), undefined, true, true);
const lRstar = mod.Layout2d.fromCoords(new Float32Array([0, 0, 1, 0, 0, 1, -1, 0]));
const clRstar = mod.crosslessness(gRstar, lRstar);

const out = {
  delCsr: { n: delCsr.numNodes(), e: delCsr.numEdges() },
  rectOv,
  ceLen: ce.length,
  ce0: [ce[0].edgeA, ce[0].edgeB],
  cetLen: cet.length,
  cs: { count: cs.count, minAngle: cs.minAngleRad, cosSq: cs.angleCosSqSum },
  gg, rngE, apsp: Array.from(apsp), sssp, snsDist, ielUnit,
  evalIel: evalRep.idealEdgeLengths, evalSns: evalRep.scaleNormalizedStress,
  evalDef: { sns: evalDefRep.sampledScaleNormalizedStress, cr: evalDefRep.sampledCrossingNumber, snsStandalone: sampledSnsDef, crStandalone: sampledCrDef },
  pdEdges: [pdEdges.sigmaDist, pdEdges.sigmaDisp],
  quality_metrics: {
    cl: qCl, scl: qScl, ma: qMa, maSum: qMaSum.toRecord(), deg: qDeg.toRecord(),
    nig: qNig, nigMs: qNigMs, rta: qRta, rtaDet: qRtaDet.toRecord(), rtaTbl: qRtaTbl.toRecord(), eval: qEval.toRecord(),
    clDir4, sclDir4, clRstar,
  },
};

gDir4.free?.(); lDir4.free?.(); gRstar.free?.(); lRstar.free?.();
qMaSum.free?.(); qDeg.free?.(); qRtaDet.free?.(); qRtaTbl.free?.(); qEval.free?.();
for (const c of ce) c.free?.();
for (const tc of cet) tc.free?.();
cs.free?.(); evalRep.free?.(); evalDefRep.free?.(); pdEdges.free?.();
lGridAdj.free?.(); lX.free?.(); gX.free?.(); lRect.free?.(); delCsr.free?.(); lGrid.free?.(); gGrid.free?.();

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

    assert ws["delCsr"]["n"] == del_csr_py.num_nodes and ws["delCsr"]["e"] == del_csr_py.num_edges
    assert list(rect_ov_py.ravel()) == ws["rectOv"]
    assert len(ce_py) == ws["ceLen"] == 1
    assert [ce_py[0]["edge_a"], ce_py[0]["edge_b"]] == ws["ce0"]
    assert len(cet_py) == ws["cetLen"]
    assert cs_py["count"] == ws["cs"]["count"]
    check_close("crossing_summary/min_angle", [cs_py["min_angle_rad"]], [ws["cs"]["minAngle"]], 1e-5)
    check_close("crossing_summary/cos_sq", [cs_py["angle_cos_sq_sum"]], [ws["cs"]["cosSq"]], 1e-5)
    assert list(gg_py.ravel()) == ws["gg"]
    assert list(rng_py.ravel()) == ws["rngE"]
    check_close("all_pairs_bfs", apsp_py, ws["apsp"], 1e-12, is_f32=False)
    check_close("bfs_single_source", sssp_py, ws["sssp"], 1e-12, is_f32=False)
    check_close("scale_normalized_stress_from_distances", [sns_dist_py], [ws["snsDist"]], 1e-5)
    check_close("ideal_edge_lengths_with_unit", [iel_unit_py], [ws["ielUnit"]], 1e-5)
    check_close("evaluate_metrics/ideal_edge_lengths", [eval_iel_py["ideal_edge_lengths"]], [ws["evalIel"]], 1e-5)
    check_close("evaluate_metrics/sns", [eval_iel_py["scale_normalized_stress"]], [ws["evalSns"]], 1e-5)
    check_close("evaluate_metrics/default_sampled_sns", [eval_default_py["sampled_scale_normalized_stress"]], [ws["evalDef"]["sns"]], 1e-12, is_f32=False)
    check_close("evaluate_metrics/default_sampled_crossing", [eval_default_py["sampled_crossing_number"]], [ws["evalDef"]["cr"]], 1e-12, is_f32=False)
    check_close("sampled_scale_normalized_stress/default", [sampled_sns_default_py], [ws["evalDef"]["snsStandalone"]], 1e-12, is_f32=False)
    check_close("sampled_crossing_number/default", [sampled_cr_default_py], [ws["evalDef"]["crStandalone"]], 1e-12, is_f32=False)
    check_close("prism_distortion_with_edges", pd_edges_py, ws["pdEdges"], 1e-5)

    qm = ws["quality_metrics"]
    check_close("crosslessness", [cl_py], [qm["cl"]], 1e-12, is_f32=False)
    check_close("sampled_crosslessness", [scl_py], [qm["scl"]], 1e-12, is_f32=False)
    check_close("minimum_angle", [ma_py], [qm["ma"]], 1e-12, is_f32=False)
    check_close("minimum_angle_summary", [ma_sum_py["a1"], ma_sum_py["a2"]], [qm["maSum"]["a1"], qm["maSum"]["a2"]], 1e-12, is_f32=False)
    assert deg_py["num_nodes"] == qm["deg"]["numNodes"]
    assert deg_py["num_edges"] == qm["deg"]["numEdges"]
    assert deg_py["duplicate_nodes_f32"] == qm["deg"]["duplicateNodesF32"]
    assert deg_py["duplicate_nodes_f64"] == qm["deg"]["duplicateNodesF64"]
    assert deg_py["zero_length_edges"] == qm["deg"]["zeroLengthEdges"]
    assert deg_py["is_bbox_degenerate"] == qm["deg"]["isBboxDegenerate"]
    check_close(
        "layout_degeneracy/ratios_bbox",
        [deg_py["duplicate_ratio_f32"], deg_py["duplicate_ratio_f64"], deg_py["zero_length_edge_ratio"], deg_py["bbox_width"], deg_py["bbox_height"]],
        [qm["deg"]["duplicateRatioF32"], qm["deg"]["duplicateRatioF64"], qm["deg"]["zeroLengthEdgeRatio"], qm["deg"]["bboxWidth"], qm["deg"]["bboxHeight"]],
        1e-12, is_f32=False,
    )
    check_close("adjusted_patch_nig", [nig_py], [qm["nig"]], 1e-12, is_f32=False)
    check_close("adjusted_patch_nig_multiscale", [nig_ms_py["patch_nig_r16"], nig_ms_py["patch_nig_r32"]], qm["nigMs"], 1e-12, is_f32=False)
    check_close("landmark_stratified_rta", [rta_py], [qm["rta"]], 1e-12, is_f32=False)
    assert rta_det_py["num_landmarks"] == qm["rtaDet"]["numLandmarks"]
    assert rta_det_py["num_triplets"] == qm["rtaDet"]["numTriplets"]
    assert rta_det_py["num_strata"] == qm["rtaDet"]["numStrata"]
    assert rta_det_py["is_exact"] == qm["rtaDet"]["isExact"]
    check_close("landmark_stratified_rta_detail", [rta_det_py["rta"]], [qm["rtaDet"]["rta"]], 1e-12, is_f32=False)
    assert rta_tbl_py["num_triplets"] == qm["rtaTbl"]["numTriplets"]
    assert rta_tbl_py["is_exact"] == qm["rtaTbl"]["isExact"]
    check_close("landmark_stratified_rta_from_landmarks", [rta_tbl_py["rta"]], [qm["rtaTbl"]["rta"]], 1e-12, is_f32=False)
    check_close(
        "evaluate_metrics/quality",
        [
            quality_eval_py["crosslessness"], quality_eval_py["sampled_crosslessness"],
            quality_eval_py["minimum_angle"], quality_eval_py["minimum_angle_a1"],
            quality_eval_py["layout_degeneracy"], quality_eval_py["adjusted_patch_nig"],
            quality_eval_py["cluster_area_faithfulness"], quality_eval_py["landmark_stratified_rta"],
        ],
        [
            qm["eval"]["crosslessness"], qm["eval"]["sampledCrosslessness"],
            qm["eval"]["minimumAngle"], qm["eval"]["minimumAngleA1"],
            qm["eval"]["layoutDegeneracy"], qm["eval"]["adjustedPatchNig"],
            qm["eval"]["clusterAreaFaithfulness"], qm["eval"]["landmarkStratifiedRta"],
        ],
        1e-12, is_f32=False,
    )
    check_close("crosslessness/dir4_reciprocal_regression", [cl_dir4_py, scl_dir4_py, cl_rstar_py], [qm["clDir4"], qm["sclDir4"], qm["clRstar"]], 1e-12, is_f32=False)

    return [{
        "case": "cross_target_spatial_quality_py_vs_wasm",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
        "max_f32_diff": round(max_f32_diff, 12),
        "max_f64_diff": round(max_f64_diff, 12),
    }]
