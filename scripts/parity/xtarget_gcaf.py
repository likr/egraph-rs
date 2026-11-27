"""Cross-target Python <-> WASM parity checks for G-CAF (Cluster Area Faithfulness)."""

from __future__ import annotations

import math
import pathlib
import time
from typing import Any

from parity.common import check_array_close, resolve_wasm_entries, run_node_json
from parity.fixtures_gcaf import gcaf_small_n_regression_cases


def run_cross_target_gcaf_parity(
    eg: Any, wasm_pkg: pathlib.Path
) -> list[dict[str, Any]]:
    """Verify Python <-> WASM G-CAF parity across grid, imbalanced, weights, boundary, and small-N cases."""
    import numpy as np

    if not hasattr(eg, "cluster_area_faithfulness"):
        return []
    if resolve_wasm_entries(wasm_pkg) is None:
        return []

    t0 = time.perf_counter()

    grid_coords_f32 = np.array(
        [[float(c) + 0.03 * float(r), float(r) - 0.02 * float(c)] for r in range(3) for c in range(3)],
        dtype=np.float32,
    )
    l_grid = eg.Layout2d.from_coords(grid_coords_f32)
    grid_comm_labels = [0, 0, 0, 0, 0, 1, 1, 1, 1]
    gcaf_grid_py = float(eg.cluster_area_faithfulness(l_grid, grid_comm_labels))
    gcaf_grid_det_py = eg.cluster_area_faithfulness_detail(l_grid, grid_comm_labels, return_per_cluster=True)

    imb24_coords_f32 = np.array(
        [
            [
                (-2.0 + 0.5 * (i / 19.0)) if i < 20 else (1.5 + 0.5 * ((i - 20) / 3.0)),
                math.sin(i * 0.7) * 0.2,
            ]
            for i in range(24)
        ],
        dtype=np.float32,
    )
    imb24_labels = [0] * 20 + [1] * 4
    l_imb24_py = eg.Layout2d.from_coords(imb24_coords_f32)
    g_imb24_py = eg.CsrGraph.from_edges(24, [(i, i + 1) for i in range(23)], directed=False, sanitize=True)
    gcaf_imb_def_py = float(eg.cluster_area_faithfulness(l_imb24_py, imb24_labels))
    gcaf_imb_custom_py = float(
        eg.cluster_area_faithfulness(
            l_imb24_py, imb24_labels, g_max=64, sigma_cells=0.9, kappa=8.0, tau_frac=0.2, boundary_r95=2.5, stevens_gamma=0.7
        )
    )
    gcaf_imb_custom_det_py = eg.cluster_area_faithfulness_detail(
        l_imb24_py, imb24_labels, g_max=64, sigma_cells=0.9, kappa=8.0, tau_frac=0.2, boundary_r95=2.5, stevens_gamma=0.7, return_per_cluster=True
    )
    gcaf_imb_inf_det_py = eg.cluster_area_faithfulness_detail(
        l_imb24_py, imb24_labels, boundary_r95=float("inf"), return_per_cluster=True
    )
    gcaf_imb_g05_det_py = eg.cluster_area_faithfulness_detail(
        l_imb24_py, imb24_labels, stevens_gamma=0.5, return_per_cluster=True
    )
    gcaf_imb_g20_det_py = eg.cluster_area_faithfulness_detail(
        l_imb24_py, imb24_labels, stevens_gamma=2.0, return_per_cluster=True
    )
    imb24_nw = [1.5 if lbl == 0 else 3.0 for lbl in imb24_labels]
    imb24_cw = [30.0, 12.0]
    gcaf_imb_nw_det_py = eg.cluster_area_faithfulness_detail(
        l_imb24_py, imb24_labels, stevens_gamma=2.0, node_weights=imb24_nw, return_per_cluster=True
    )
    gcaf_imb_cw_det_py = eg.cluster_area_faithfulness_detail(
        l_imb24_py, imb24_labels, stevens_gamma=2.0, cluster_weights=imb24_cw, return_per_cluster=True
    )
    gcaf_imb_eval_py = eg.evaluate_metrics(
        g_imb24_py, l_imb24_py, ["gcaf"], labels=imb24_labels,
        g_max=64, sigma_cells=0.9, kappa=8.0, tau_frac=0.2, boundary_r95=2.5, stevens_gamma=0.7,
    )
    int20_coords_f64 = np.array(
        [[float(c + (0 if c < 2 else 4)), float(r)] for c in range(4) for r in range(5)], dtype=np.float64
    )
    int20_labels = [0 if c < 2 else 1 for c in range(4) for _r in range(5)]
    gcaf_int20_f64_py = eg.cluster_area_faithfulness_detail(eg.Layout2d.from_coords(int20_coords_f64), int20_labels, return_per_cluster=True)
    gcaf_int20_f32_py = eg.cluster_area_faithfulness_detail(eg.Layout2d.from_coords(int20_coords_f64.astype(np.float32)), int20_labels, return_per_cluster=True)
    gcaf_b0_py = eg.cluster_area_faithfulness_detail(eg.Layout2d.zeros(0), [], return_per_cluster=True)
    gcaf_b1_py = eg.cluster_area_faithfulness_detail(eg.Layout2d.from_coords(np.array([[1.0, 2.0]], dtype=np.float32)), [5], return_per_cluster=True)
    gcaf_bk1_py = eg.cluster_area_faithfulness_detail(eg.Layout2d.from_coords(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)), [3, 3, 3, 3], return_per_cluster=True)
    gcaf_bcol_py = float(eg.cluster_area_faithfulness(eg.Layout2d.from_coords(np.array([[2.0, -1.0]] * 6, dtype=np.float32)), [0, 0, 0, 1, 1, 1]))

    gcaf_small_n_cases = gcaf_small_n_regression_cases()
    gcaf_small_n_py: dict[str, dict[str, Any]] = {}
    for case in gcaf_small_n_cases:
        coords_arr = np.asarray(case["coords"], dtype=np.float32).reshape(-1, 2)
        n_case = coords_arr.shape[0]
        l_case = eg.Layout2d.from_coords(coords_arr)
        g_case = eg.CsrGraph.from_edges(n_case, [(i, i + 1) for i in range(n_case - 1)], directed=False, sanitize=True)
        kwargs: dict[str, Any] = {}
        if case["g_max"] is not None:
            kwargs["g_max"] = case["g_max"]
        if case["sigma_cells"] is not None:
            kwargs["sigma_cells"] = case["sigma_cells"]
        if case["boundary_r95"] is not None:
            kwargs["boundary_r95"] = case["boundary_r95"]
        if case["stevens_gamma"] is not None:
            kwargs["stevens_gamma"] = case["stevens_gamma"]
        gcaf_small_n_py[case["name"]] = {
            "caf": float(eg.cluster_area_faithfulness(l_case, case["labels"], **kwargs)),
            "det": eg.cluster_area_faithfulness_detail(l_case, case["labels"], return_per_cluster=True, **kwargs),
            "eval": eg.evaluate_metrics(g_case, l_case, ["gcaf"], labels=case["labels"], **kwargs),
        }

    spec_payload = {
        "grid_coords": [float(v) for v in grid_coords_f32.ravel()],
        "imb24_coords": [float(v) for v in imb24_coords_f32.ravel()],
        "int20_coords": [float(v) for v in int20_coords_f64.astype(np.float32).ravel()],
        "int20_labels": int20_labels,
        "gcaf_small_n": [
            {
                **c,
                "boundary_r95": (
                    "Infinity"
                    if (c["boundary_r95"] is not None and math.isinf(c["boundary_r95"]))
                    else c["boundary_r95"]
                ),
            }
            for c in gcaf_small_n_cases
        ],
    }

    node_body = """
const lGrid = mod.Layout2d.fromCoords(new Float32Array(spec.grid_coords));
const gridComm = new Uint32Array([0, 0, 0, 0, 0, 1, 1, 1, 1]);
const gcafGrid = mod.clusterAreaFaithfulness(lGrid, gridComm);
const gcafGridDet = mod.clusterAreaFaithfulnessDetail(lGrid, gridComm);
const imb24Labels = new Uint32Array(Array.from({ length: 24 }, (_, i) => (i < 20 ? 0 : 1)));
const lImb24 = mod.Layout2d.fromCoords(new Float32Array(spec.imb24_coords));
const imb24Edges = [];
for (let i = 0; i < 23; i++) imb24Edges.push(i, i + 1);
const gImb24 = mod.CsrGraph.fromEdgeList(24, new Uint32Array(imb24Edges));
const gcafImbDef = mod.clusterAreaFaithfulness(lImb24, imb24Labels);
const gcafImbCustom = mod.clusterAreaFaithfulness(lImb24, imb24Labels, 64, 0.9, 8.0, 0.2, 2.5, 0.7);
const gcafImbCustomDet = mod.clusterAreaFaithfulnessDetail(lImb24, imb24Labels, 64, 0.9, 8.0, 0.2, 2.5, 0.7);
const gcafImbInfDet = mod.clusterAreaFaithfulnessDetail(lImb24, imb24Labels, null, null, null, null, Number.POSITIVE_INFINITY, null);
const gcafImbG05Det = mod.clusterAreaFaithfulnessDetail(lImb24, imb24Labels, null, null, null, null, null, 0.5);
const gcafImbG20Det = mod.clusterAreaFaithfulnessDetail(lImb24, imb24Labels, null, null, null, null, null, 2.0);
const imb24Nw = new Float64Array(24);
for (let i = 0; i < 24; i++) imb24Nw[i] = i < 20 ? 1.5 : 3.0;
const imb24Cw = new Float64Array([30.0, 12.0]);
const gcafImbNwDet = mod.clusterAreaFaithfulnessDetail(lImb24, imb24Labels, null, null, null, null, null, 2.0, imb24Nw, null);
const gcafImbCwDet = mod.clusterAreaFaithfulnessDetail(lImb24, imb24Labels, null, null, null, null, null, 2.0, null, imb24Cw);
const gcafImbEval = mod.evaluateMetrics(gImb24, lImb24, ['gcaf'], null, null, null, null, imb24Labels, null, null, null, null, null, null, 64, 0.9, 8.0, 0.2, 2.5, 0.7);
const lInt20 = mod.Layout2d.fromCoords(new Float32Array(spec.int20_coords));
const gcafInt20Det = mod.clusterAreaFaithfulnessDetail(lInt20, new Uint32Array(spec.int20_labels));
const lGcafB0 = mod.Layout2d.zeros(0);
const gcafB0 = mod.clusterAreaFaithfulnessDetail(lGcafB0, []);
const lGcafB1 = mod.Layout2d.fromCoords(new Float32Array([1, 2]));
const gcafB1 = mod.clusterAreaFaithfulnessDetail(lGcafB1, [5]);
const lGcafBk1 = mod.Layout2d.fromCoords(new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]));
const gcafBk1 = mod.clusterAreaFaithfulnessDetail(lGcafBk1, [3, 3, 3, 3]);
const lGcafCol = mod.Layout2d.fromCoords(new Float32Array([2, -1, 2, -1, 2, -1, 2, -1, 2, -1, 2, -1]));
const gcafBCol = mod.clusterAreaFaithfulness(lGcafCol, [0, 0, 0, 1, 1, 1]);

const readGcafDet = (d) => {
  const rec = d.toRecord();
  return {
    caf: d.caf, purityAdj: d.purityAdj, propVis: d.propVis, resDensity: d.resDensity,
    resDeconv: d.resDeconv, resCombined: d.resCombined, spatialCoherence: d.spatialCoherence,
    inFrac: d.inFrac, effectiveBoundaryR95: d.effectiveBoundaryR95, gEff: d.gEff,
    numClusters: d.numClusters, numNodes: d.numNodes,
    clusterIds: Array.from(d.clusterIds()), counts: Array.from(d.counts()),
    p: Array.from(d.p()), q: Array.from(d.q()), purity: Array.from(d.purity()),
    resolution: Array.from(d.resolution()), resDensityPerCluster: Array.from(d.resDensityPerCluster()),
    resDeconvPerCluster: Array.from(d.resDeconvPerCluster()), cafPerCluster: Array.from(d.cafPerCluster()),
    coherence: Array.from(d.coherence()),
    rec: {
      caf: rec.caf, purityAdj: rec.purityAdj, propVis: rec.propVis, resDensity: rec.resDensity,
      resDeconv: rec.resDeconv, resCombined: rec.resCombined, spatialCoherence: rec.spatialCoherence,
      inFrac: rec.inFrac, effectiveBoundaryR95: rec.effectiveBoundaryR95, gEff: rec.gEff,
      numClusters: rec.numClusters, numNodes: rec.numNodes,
      clusterIds: Array.from(rec.uniqueLabels), counts: Array.from(rec.counts),
      p: Array.from(rec.p), q: Array.from(rec.q), purity: Array.from(rec.purity),
      resolution: Array.from(rec.resolution), resDensityPerCluster: Array.from(rec.resDensityPerCluster),
      resDeconvPerCluster: Array.from(rec.resDeconvPerCluster), cafPerCluster: Array.from(rec.cafPerCluster),
      coherence: Array.from(rec.coherence),
    },
  };
};

const gcafSmallN = {};
for (const c of spec.gcaf_small_n) {
  const nCase = c.labels.length;
  const lCase = mod.Layout2d.fromCoords(new Float32Array(c.coords));
  const cEdges = [];
  for (let i = 0; i + 1 < nCase; i++) cEdges.push(i, i + 1);
  const gCase = mod.CsrGraph.fromEdgeList(nCase, new Uint32Array(cEdges));
  const cLabels = new Uint32Array(c.labels);
  const bR95 = c.boundary_r95 === 'Infinity' ? Infinity : c.boundary_r95;
  const cafVal = mod.clusterAreaFaithfulness(lCase, cLabels, c.g_max, c.sigma_cells, null, null, bR95, c.stevens_gamma);
  const detHandle = mod.clusterAreaFaithfulnessDetail(lCase, cLabels, c.g_max, c.sigma_cells, null, null, bR95, c.stevens_gamma);
  const evalHandle = mod.evaluateMetrics(gCase, lCase, ['gcaf'], null, null, null, null, cLabels, null, null, null, null, null, null, c.g_max, c.sigma_cells, null, null, bR95, c.stevens_gamma);
  const evalRec = evalHandle.toRecord();
  gcafSmallN[c.name] = {
    caf: cafVal, det: readGcafDet(detHandle),
    evalCaf: evalHandle.clusterAreaFaithfulness, evalRecCaf: evalRec.clusterAreaFaithfulness,
  };
  detHandle.free?.(); evalHandle.free?.(); gCase.free?.(); lCase.free?.();
}

const out = {
  grid: gcafGrid, gridDet: readGcafDet(gcafGridDet),
  imbDef: gcafImbDef, imbCustom: gcafImbCustom,
  imbCustomDet: readGcafDet(gcafImbCustomDet), imbInfDet: readGcafDet(gcafImbInfDet),
  imbG05Det: readGcafDet(gcafImbG05Det), imbG20Det: readGcafDet(gcafImbG20Det),
  imbNwDet: readGcafDet(gcafImbNwDet), imbCwDet: readGcafDet(gcafImbCwDet),
  imbEval: gcafImbEval.toRecord(), int20Det: readGcafDet(gcafInt20Det),
  b0: readGcafDet(gcafB0), b1: readGcafDet(gcafB1), bk1: readGcafDet(gcafBk1),
  bCol: gcafBCol, smallN: gcafSmallN,
};

gcafGridDet.free?.(); gcafImbCustomDet.free?.(); gcafImbInfDet.free?.(); gcafImbG05Det.free?.();
gcafImbG20Det.free?.(); gcafImbNwDet.free?.(); gcafImbCwDet.free?.(); gcafImbEval.free?.();
gcafInt20Det.free?.(); gcafB0.free?.(); gcafB1.free?.(); gcafBk1.free?.();
lImb24.free?.(); gImb24.free?.(); lInt20.free?.(); lGcafB0.free?.(); lGcafB1.free?.(); lGcafBk1.free?.(); lGcafCol.free?.();
lGrid.free?.();

console.log(JSON.stringify(out));
"""
    gcaf_ws = run_node_json(wasm_pkg, node_body, spec_payload)

    max_f64_diff = 0.0

    def check_f64(label: str, a: Any, b: Any) -> None:
        nonlocal max_f64_diff
        max_f64_diff = max(max_f64_diff, check_array_close(label, a, b, 1e-12))

    def check_gcaf_detail(label: str, py_det: dict[str, Any], ws_det: dict[str, Any]) -> None:
        scalar_map = (
            ("caf", "caf"), ("purity_adj", "purityAdj"), ("prop_vis", "propVis"),
            ("res_density", "resDensity"), ("res_deconv", "resDeconv"), ("res_combined", "resCombined"),
            ("spatial_coherence", "spatialCoherence"), ("in_frac", "inFrac"),
            ("effective_boundary_r95", "effectiveBoundaryR95"),
        )
        vec_map = (
            ("p", "p"), ("q", "q"), ("purity", "purity"), ("resolution", "resolution"),
            ("res_density_per_cluster", "resDensityPerCluster"), ("res_deconv_per_cluster", "resDeconvPerCluster"),
            ("caf_per_cluster", "cafPerCluster"), ("coherence", "coherence"),
        )
        for src_obj in (ws_det, ws_det["rec"]):
            assert py_det["g_eff"] == src_obj["gEff"], f"{label}: g_eff mismatch"
            assert py_det["num_clusters"] == src_obj["numClusters"], f"{label}: num_clusters mismatch"
            assert py_det["num_nodes"] == src_obj["numNodes"], f"{label}: num_nodes mismatch"
            assert list(py_det["cluster_ids"]) == src_obj["clusterIds"], f"{label}: cluster_ids mismatch"
            assert list(py_det["counts"]) == src_obj["counts"], f"{label}: counts mismatch"
            for py_k, ws_k in scalar_map:
                check_f64(f"{label}/{py_k}", [py_det[py_k]], [src_obj[ws_k]])
            for py_k, ws_k in vec_map:
                check_f64(f"{label}/{py_k}", py_det[py_k], src_obj[ws_k])

    check_f64("gcaf/grid", [gcaf_grid_py], [gcaf_ws["grid"]])
    check_gcaf_detail("gcaf/grid_det", gcaf_grid_det_py, gcaf_ws["gridDet"])
    check_f64("gcaf/imb_def", [gcaf_imb_def_py], [gcaf_ws["imbDef"]])
    check_f64("gcaf/imb_custom", [gcaf_imb_custom_py, gcaf_imb_eval_py["cluster_area_faithfulness"]], [gcaf_ws["imbCustom"], gcaf_ws["imbEval"]["clusterAreaFaithfulness"]])
    check_gcaf_detail("gcaf/imb_custom_det", gcaf_imb_custom_det_py, gcaf_ws["imbCustomDet"])
    check_gcaf_detail("gcaf/imb_inf_det", gcaf_imb_inf_det_py, gcaf_ws["imbInfDet"])
    check_gcaf_detail("gcaf/imb_g05_det", gcaf_imb_g05_det_py, gcaf_ws["imbG05Det"])
    check_gcaf_detail("gcaf/imb_g20_det", gcaf_imb_g20_det_py, gcaf_ws["imbG20Det"])
    check_gcaf_detail("gcaf/imb_nw_det", gcaf_imb_nw_det_py, gcaf_ws["imbNwDet"])
    check_gcaf_detail("gcaf/imb_cw_det", gcaf_imb_cw_det_py, gcaf_ws["imbCwDet"])
    check_gcaf_detail("gcaf/imb_nw_vs_cw", gcaf_imb_nw_det_py, gcaf_ws["imbCwDet"])
    check_gcaf_detail("gcaf/int20_f32", gcaf_int20_f32_py, gcaf_ws["int20Det"])
    check_gcaf_detail("gcaf/int20_f64", gcaf_int20_f64_py, gcaf_ws["int20Det"])
    check_gcaf_detail("gcaf/b0", gcaf_b0_py, gcaf_ws["b0"])
    check_gcaf_detail("gcaf/b1", gcaf_b1_py, gcaf_ws["b1"])
    check_gcaf_detail("gcaf/bk1", gcaf_bk1_py, gcaf_ws["bk1"])
    check_f64("gcaf/bcol", [gcaf_bcol_py], [gcaf_ws["bCol"]])
    for case in gcaf_small_n_cases:
        cname = case["name"]
        py_c = gcaf_small_n_py[cname]
        ws_c = gcaf_ws["smallN"][cname]
        check_f64(
            f"gcaf/small_n/{cname}/scalar_and_eval",
            [py_c["caf"], py_c["eval"]["cluster_area_faithfulness"], py_c["eval"]["cluster_area_faithfulness"]],
            [ws_c["caf"], ws_c["evalCaf"], ws_c["evalRecCaf"]],
        )
        check_gcaf_detail(f"gcaf/small_n/{cname}/det", py_c["det"], ws_c["det"])

    return [{
        "case": "cross_target_gcaf_py_vs_wasm",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
        "max_f64_diff": round(max_f64_diff, 12),
    }]
