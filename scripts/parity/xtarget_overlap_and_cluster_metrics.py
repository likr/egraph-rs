"""Cross-target Python <-> WASM parity checks for GTree, FORBID, NOR, and cluster metrics."""

from __future__ import annotations

import pathlib
import time
from typing import Any

from parity.common import (
    check_array_close,
    make_asymmetric_coords,
    make_barbell_edges,
    resolve_wasm_entries,
    run_node_json,
)


def run_cross_target_overlap_cluster_metrics_parity(
    eg: Any, wasm_pkg: pathlib.Path
) -> list[dict[str, Any]]:
    """Verify Python float32 <-> WASM f32 parity across GTree, FORBID, NOR, and cluster metrics."""
    import numpy as np

    if not hasattr(eg, "GTreeOverlapRemoval"):
        return []
    if resolve_wasm_entries(wasm_pkg) is None:
        return []

    t0 = time.perf_counter()

    bb_edges = make_barbell_edges(5)
    g_bb = eg.CsrGraph.from_edges(10, bb_edges, directed=False, sanitize=True)

    init10_f32 = np.array(make_asymmetric_coords(10, scale=1.2), dtype=np.float32)
    sep_f32 = np.array(
        [
            [-5.0, 0.0],
            [-4.8, 0.1],
            [-5.1, -0.1],
            [-4.9, 0.2],
            [-5.0, -0.2],
            [5.0, 0.0],
            [5.2, 0.1],
            [4.9, -0.1],
            [5.1, 0.2],
            [5.0, -0.2],
        ],
        dtype=np.float32,
    )
    ov_f32 = (init10_f32 * 0.15).astype(np.float32)
    ov_f32[1] = ov_f32[0]
    comm_labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    pred_labels = [10, 10, 10, 10, 20, 20, 20, 20, 20, 10]
    hw10_f32 = [0.45] * 10
    hh10_f32 = [0.35] * 10

    # GTree & FORBID
    gt_box = eg.GTreeOverlapRemoval(hw10_f32, hh10_f32, max_iterations=25, padding=0.02, seed=42, dtype="float32")
    l_gt_box = eg.Layout2d.from_coords(ov_f32)
    gt_box.run(l_gt_box)
    gt_rad = eg.GTreeOverlapRemoval.from_radii(hw10_f32, max_iterations=25, padding=0.02, seed=42, dtype="float32")
    l_gt_rad = eg.Layout2d.from_coords(ov_f32)
    gt_rad.run(l_gt_rad)

    fb_py: dict[str, dict[str, Any]] = {}
    for var in ("forbid", "forbid_prime", "sparse_forbid"):
        fb = eg.ForbidOverlapRemoval(
            hw10_f32, hh10_f32, variant=var, max_scale_iters=25, sgd_iterations=30, seed=42, dtype="float32"
        )
        lay = eg.Layout2d.from_coords(ov_f32)
        fb.run(lay)
        fb_py[var] = {"coords": lay.coords(copy=True), "ov": int(fb.last_overlap_count)}
    fb_rad = eg.ForbidOverlapRemoval.from_radii(
        hw10_f32, variant="forbid", max_scale_iters=25, sgd_iterations=30, seed=42, dtype="float32"
    )
    l_fb_rad = eg.Layout2d.from_coords(ov_f32)
    fb_rad.run(l_fb_rad)

    # N=80 > sparse_k=30: default ForbidOverlapRemoval matches sparse_forbid & WASM default and diverges from forbid
    ov80_f32 = (np.array(make_asymmetric_coords(80, scale=0.3), dtype=np.float32)).astype(np.float32)
    hw80_f32 = [0.25] * 80
    hh80_f32 = [0.25] * 80
    l_fb80_def = eg.Layout2d.from_coords(ov80_f32)
    l_fb80_sp = eg.Layout2d.from_coords(ov80_f32)
    l_fb80_full = eg.Layout2d.from_coords(ov80_f32)
    l_fb80_x = eg.Layout2d.from_coords(ov80_f32)
    eg.ForbidOverlapRemoval(hw80_f32, hh80_f32, dtype="float32").step(l_fb80_def)
    eg.ForbidOverlapRemoval(hw80_f32, hh80_f32, variant="sparse_forbid", dtype="float32").step(l_fb80_sp)
    eg.ForbidOverlapRemoval(hw80_f32, hh80_f32, variant="forbid", dtype="float32").step(l_fb80_full)
    eg.ForbidOverlapRemoval(hw80_f32, hh80_f32, sgd_iterations=5, dtype="float32").step(l_fb80_x)
    lr_fb80_def = eg.Layout2d.from_coords(ov80_f32)
    lr_fb80_sp = eg.Layout2d.from_coords(ov80_f32)
    lr_fb80_full = eg.Layout2d.from_coords(ov80_f32)
    lr_fb80_x = eg.Layout2d.from_coords(ov80_f32)
    eg.ForbidOverlapRemoval.from_radii(hw80_f32, dtype="float32").step(lr_fb80_def)
    eg.ForbidOverlapRemoval.from_radii(hw80_f32, variant="sparse_forbid", dtype="float32").step(lr_fb80_sp)
    eg.ForbidOverlapRemoval.from_radii(hw80_f32, variant="forbid", dtype="float32").step(lr_fb80_full)
    eg.ForbidOverlapRemoval.from_radii(hw80_f32, sgd_iterations=5, dtype="float32").step(lr_fb80_x)

    # Cluster baselines & evaluate_metrics
    l_sep = eg.Layout2d.from_coords(sep_f32)
    l_ov = eg.Layout2d.from_coords(ov_f32)
    pa_py = eg.partition_agreement(comm_labels, pred_labels)
    cq_km_py = eg.cq_kmeans(l_sep, comm_labels, n_init=5, max_iterations=50, seed=42)
    cq_hac_py = eg.cq_hac(l_sep, comm_labels)
    cdm_py, hdm_py = eg.class_density_measure(l_sep, comm_labels, grid_size=32)
    nor_g_py = eg.nor_metrics(l_ov, l_gt_box, hw10_f32, hh10_f32, graph=g_bb)
    nor_nog_py = eg.nor_metrics(l_ov, l_gt_box, hw10_f32, hh10_f32)
    cluster_metric_targets = [
        "distance_consistency",
        "distribution_consistency",
        "class_density_measure",
        "histogram_density_measure",
        "sepme_knng",
        "sepme_rng",
        "sepme_gab",
        "sepme_gong",
        "neighborhood_hit",
        "silhouette_score",
        "calinski_harabasz_score",
        "davies_bouldin_score",
        "cq_kmeans_ari",
        "cq_kmeans_ami",
        "cq_hac_hom",
        "cq_hac_ami",
        "pairwise_distance_overlap_error",
        "sprawlter_nn",
    ]
    ev_cluster_py = eg.evaluate_metrics(
        g_bb,
        l_sep,
        cluster_metric_targets,
        labels=comm_labels,
        sepme_knng_k=2,
        neighborhood_hit_k=3,
        dc_grid_size=16,
        cdm_grid_size=32,
        num_bins=32,
        gong_gamma=0.35,
        kmeans_n_init=5,
        kmeans_max_iterations=50,
        sprawlter_node_radius=0.5,
        seed=42,
    )
    standalone_cluster_py = {
        "dsc": float(eg.distance_consistency(l_sep, comm_labels)),
        "dc": float(eg.distribution_consistency(l_sep, comm_labels, grid_size=16)),
        "cdm": float(cdm_py),
        "hdm": float(hdm_py),
        "sepme_knng": float(eg.sepme_knng(l_sep, comm_labels, k=2)),
        "sepme_rng": float(eg.sepme_rng(l_sep, comm_labels)),
        "sepme_gab": float(eg.sepme_gab(l_sep, comm_labels)),
        "sepme_gong": float(eg.sepme_gong(l_sep, comm_labels, gamma=0.35)),
        "neighborhood_hit": float(eg.neighborhood_hit(l_sep, comm_labels, k=3)),
        "silhouette": float(eg.silhouette_score(l_sep, comm_labels)),
        "calinski_harabasz": float(eg.calinski_harabasz_score(l_sep, comm_labels)),
        "davies_bouldin": float(eg.davies_bouldin_score(l_sep, comm_labels)),
        "pdoe": float(eg.pairwise_distance_overlap_error(l_sep, comm_labels, num_bins=32)),
        "sprawlter_rect": float(eg.sprawlter_nn(l_ov, hw10_f32, hh10_f32)),
        "sprawlter_rad": float(eg.sprawlter_nn(l_ov, hw10_f32)),
    }

    spec_payload = {
        "bb_edges": [int(v) for edge in bb_edges for v in edge],
        "sep": [float(v) for v in sep_f32.ravel()],
        "ov": [float(v) for v in ov_f32.ravel()],
        "comm_labels": comm_labels,
        "pred_labels": pred_labels,
        "hw10": hw10_f32,
        "hh10": hh10_f32,
        "ov80": [float(v) for v in ov80_f32.ravel()],
        "hw80": hw80_f32,
        "hh80": hh80_f32,
    }

    node_body = """
if (typeof mod.GTreeOverlapRemoval !== 'function') {
  console.log(JSON.stringify({ overlap_cluster_available: false }));
  process.exit(0);
}

const gBb = mod.CsrGraph.fromEdgeList(10, new Uint32Array(spec.bb_edges), null, false, true);
const sep = new Float32Array(spec.sep);
const ov = new Float32Array(spec.ov);
const commLabels = new Uint32Array(spec.comm_labels);
const predLabels = new Uint32Array(spec.pred_labels);
const hw10 = new Float32Array(spec.hw10);
const hh10 = new Float32Array(spec.hh10);

const gtBox = new mod.GTreeOverlapRemoval(hw10, hh10, { maxIterations: 25, padding: 0.02, seed: 42 });
const lGtBox = mod.Layout2d.fromCoords(ov);
gtBox.run(lGtBox);
const gtRad = mod.GTreeOverlapRemoval.fromRadii(hw10, { maxIterations: 25, padding: 0.02, seed: 42 });
const lGtRad = mod.Layout2d.fromCoords(ov);
gtRad.run(lGtRad);

const fb = {};
for (const variant of ['forbid', 'forbid_prime', 'sparse_forbid']) {
  const s = new mod.ForbidOverlapRemoval(hw10, hh10, { variant, maxScaleIters: 25, sgdIterations: 30, seed: 42 });
  const l = mod.Layout2d.fromCoords(ov);
  s.run(l);
  fb[variant] = { coords: Array.from(l.coordsView()), ov: s.lastOverlapCount };
  l.free?.(); s.free?.();
}
const fbRad = mod.ForbidOverlapRemoval.fromRadii(hw10, { variant: 'forbid', maxScaleIters: 25, sgdIterations: 30, seed: 42 });
const lFbRad = mod.Layout2d.fromCoords(ov);
fbRad.run(lFbRad);

const ov80 = new Float32Array(spec.ov80);
const hw80 = new Float32Array(spec.hw80);
const hh80 = new Float32Array(spec.hh80);
const lFb80Def = mod.Layout2d.fromCoords(ov80);
const lFb80Sp = mod.Layout2d.fromCoords(ov80);
const lFb80Full = mod.Layout2d.fromCoords(ov80);
const lFb80X = mod.Layout2d.fromCoords(ov80);
const lFb80RadDef = mod.Layout2d.fromCoords(ov80);
const lFb80RadX = mod.Layout2d.fromCoords(ov80);
new mod.ForbidOverlapRemoval(hw80, hh80).step(lFb80Def);
new mod.ForbidOverlapRemoval(hw80, hh80, { variant: 'sparse_forbid' }).step(lFb80Sp);
new mod.ForbidOverlapRemoval(hw80, hh80, { variant: 'forbid' }).step(lFb80Full);
new mod.ForbidOverlapRemoval(hw80, hh80, { sgdIterations: 5 }).step(lFb80X);
mod.ForbidOverlapRemoval.fromRadii(hw80).step(lFb80RadDef);
mod.ForbidOverlapRemoval.fromRadii(hw80, { sgdIterations: 5 }).step(lFb80RadX);

const lSep = mod.Layout2d.fromCoords(sep);
const lOv = mod.Layout2d.fromCoords(ov);
const pa = mod.partitionAgreement(commLabels, predLabels);
const cqKm = mod.cqKmeans(lSep, commLabels, 5, 50, 42);
const cqHac = mod.cqHac(lSep, commLabels);
const cdmRes = mod.classDensityMeasure(lSep, commLabels, 32, null);
const norG = mod.norMetrics(gBb, lOv, lGtBox, hw10, hh10);
const norNoG = mod.norMetricsWithoutGraph(lOv, lGtBox, hw10, hh10);
const evRep = mod.evaluateMetrics(
  gBb,
  lSep,
  [
    'distance_consistency', 'distribution_consistency', 'class_density_measure', 'histogram_density_measure',
    'sepme_knng', 'sepme_rng', 'sepme_gab', 'sepme_gong',
    'neighborhood_hit', 'silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score',
    'cq_kmeans_ari', 'cq_kmeans_ami', 'cq_hac_hom', 'cq_hac_ami',
    'pairwise_distance_overlap_error', 'sprawlter_nn',
  ],
  {
    labels: commLabels, sepmeKnngK: 2, neighborhoodHitK: 3, dcGridSize: 16, cdmGridSize: 32,
    numBins: 32, gongGamma: 0.35, kmeansNInit: 5, kmeansMaxIterations: 50, sprawlterNodeRadius: 0.5, seed: 42,
  },
);

const out = {
  overlap_cluster_available: true,
  gt: {
    boxCoords: Array.from(lGtBox.coordsView()), boxOv: gtBox.lastOverlapCount,
    radCoords: Array.from(lGtRad.coordsView()), radOv: gtRad.lastOverlapCount,
  },
  fb,
  fbRad: { coords: Array.from(lFbRad.coordsView()), ov: fbRad.lastOverlapCount },
  fb80: {
    defCoords: Array.from(lFb80Def.coordsView()),
    spCoords: Array.from(lFb80Sp.coordsView()),
    fullCoords: Array.from(lFb80Full.coordsView()),
    xCoords: Array.from(lFb80X.coordsView()),
    radDefCoords: Array.from(lFb80RadDef.coordsView()),
    radXCoords: Array.from(lFb80RadX.coordsView()),
  },
  cluster_metrics: {
    pa: pa.toRecord(), cqKm: cqKm.toRecord(), cqHac: cqHac.toRecord(),
    norG: norG.toRecord(), norNoG: norNoG.toRecord(), ev: evRep.toRecord(),
    standalone: {
      dsc: mod.distanceConsistency(lSep, commLabels),
      dc: mod.distributionConsistency(lSep, commLabels, 16),
      cdm: cdmRes.cdm, hdm: cdmRes.hdm,
      sepmeKnng: mod.sepmeKnng(lSep, commLabels, 2),
      sepmeRng: mod.sepmeRng(lSep, commLabels),
      sepmeGab: mod.sepmeGab(lSep, commLabels),
      sepmeGong: mod.sepmeGong(lSep, commLabels, 0.35),
      neighborhoodHit: mod.neighborhoodHit(lSep, commLabels, 3),
      silhouette: mod.silhouetteScore(lSep, commLabels),
      calinskiHarabasz: mod.calinskiHarabaszScore(lSep, commLabels),
      daviesBouldin: mod.daviesBouldinScore(lSep, commLabels),
      pdoe: mod.pairwiseDistanceOverlapError(lSep, commLabels, 32),
      sprawlterRect: mod.sprawlterNn(lOv, hw10, hh10),
      sprawlterRad: mod.sprawlterNnRadii(lOv, hw10),
    },
  },
};

evRep.free?.(); norNoG.free?.(); norG.free?.(); cdmRes.free?.(); cqHac.free?.(); cqKm.free?.(); pa.free?.();
lFb80RadX.free?.(); lFb80RadDef.free?.(); lFb80X.free?.(); lFb80Full.free?.(); lFb80Sp.free?.(); lFb80Def.free?.();
lOv.free?.(); lSep.free?.(); lFbRad.free?.(); fbRad.free?.(); lGtRad.free?.(); gtRad.free?.(); lGtBox.free?.(); gtBox.free?.(); gBb.free?.();

console.log(JSON.stringify(out));
"""
    ws = run_node_json(wasm_pkg, node_body, spec_payload)
    if not ws.get("overlap_cluster_available", False):
        return [{
            "case": "cross_target_overlap_cluster_metrics_py_vs_wasm",
            "status": "skipped_no_wasm_overlap_cluster_metrics",
            "elapsed_sec": round(time.perf_counter() - t0, 4),
        }]

    max_f32_diff = 0.0
    max_f64_diff = 0.0

    def check_close(label: str, a: Any, b: Any, tol: float, is_f32: bool = True) -> None:
        nonlocal max_f32_diff, max_f64_diff
        diff = check_array_close(label, a, b, tol)
        if is_f32:
            max_f32_diff = max(max_f32_diff, diff)
        else:
            max_f64_diff = max(max_f64_diff, diff)

    assert gt_box.last_overlap_count == ws["gt"]["boxOv"] == 0
    assert gt_rad.last_overlap_count == ws["gt"]["radOv"] == 0
    check_close("gtree/box", l_gt_box.coords(copy=True), ws["gt"]["boxCoords"], 1e-4)
    check_close("gtree/radii", l_gt_rad.coords(copy=True), ws["gt"]["radCoords"], 1e-4)
    for var in ("forbid", "forbid_prime", "sparse_forbid"):
        assert fb_py[var]["ov"] == ws["fb"][var]["ov"] == 0
        check_close(f"forbid/{var}", fb_py[var]["coords"], ws["fb"][var]["coords"], 1e-4)
    assert fb_rad.last_overlap_count == ws["fbRad"]["ov"] == 0
    check_close("forbid/radii", l_fb_rad.coords(copy=True), ws["fbRad"]["coords"], 1e-4)

    cm = ws["cluster_metrics"]
    pa_keys = (("ari", "ari"), ("ami", "ami"), ("homogeneity", "homogeneity"), ("completeness", "completeness"), ("v_measure", "vMeasure"), ("fmi", "fmi"))
    for pk, wk in pa_keys:
        check_close(f"partition_agreement/{pk}", [pa_py[pk]], [cm["pa"][wk]], 1e-12, is_f32=False)
        check_close(f"cq_kmeans/{pk}", [cq_km_py[pk]], [cm["cqKm"][wk]], 1e-12, is_f32=False)
        check_close(f"cq_hac/{pk}", [cq_hac_py[pk]], [cm["cqHac"][wk]], 1e-12, is_f32=False)
    for tag, py_n, ws_n in (("with_graph", nor_g_py, cm["norG"]), ("no_graph", nor_nog_py, cm["norNoG"])):
        for pk, wk in (("oo_nni", "ooNni"), ("sp_ch_a", "spChA"), ("gs_bb_iar", "gsBbIar"), ("nm_dm_imse", "nmDmImse"), ("el_rsd", "elRsd")):
            check_close(f"nor/{tag}/{pk}", [py_n[pk]], [ws_n[wk]], 1e-10, is_f32=False)
    for pk, wk in (
        ("dsc", "dsc"), ("dc", "dc"), ("cdm", "cdm"), ("hdm", "hdm"),
        ("sepme_knng", "sepmeKnng"), ("sepme_rng", "sepmeRng"), ("sepme_gab", "sepmeGab"), ("sepme_gong", "sepmeGong"),
        ("neighborhood_hit", "neighborhoodHit"), ("silhouette", "silhouette"),
        ("calinski_harabasz", "calinskiHarabasz"), ("davies_bouldin", "daviesBouldin"),
        ("pdoe", "pdoe"), ("sprawlter_rect", "sprawlterRect"), ("sprawlter_rad", "sprawlterRad"),
    ):
        check_close(f"cluster_standalone/{pk}", [standalone_cluster_py[pk]], [cm["standalone"][wk]], 1e-10, is_f32=False)
    for pk, wk in (
        ("distance_consistency", "distanceConsistency"),
        ("distribution_consistency", "distributionConsistency"),
        ("class_density_measure", "classDensityMeasure"),
        ("histogram_density_measure", "histogramDensityMeasure"),
        ("sepme_knng", "sepmeKnng"), ("sepme_rng", "sepmeRng"), ("sepme_gab", "sepmeGab"), ("sepme_gong", "sepmeGong"),
        ("neighborhood_hit", "neighborhoodHit"),
        ("silhouette_score", "silhouetteScore"),
        ("calinski_harabasz_score", "calinskiHarabaszScore"),
        ("davies_bouldin_score", "daviesBouldinScore"),
        ("cq_kmeans_ari", "cqKmeansAri"), ("cq_kmeans_ami", "cqKmeansAmi"),
        ("cq_hac_hom", "cqHacHom"), ("cq_hac_ami", "cqHacAmi"),
        ("pairwise_distance_overlap_error", "pairwiseDistanceOverlapError"), ("sprawlter_nn", "sprawlterNn"),
    ):
        check_close(f"evaluate_metrics/cluster/{pk}", [ev_cluster_py[pk]], [cm["ev"][wk]], 1e-10, is_f32=False)

    # Verify default ForbidOverlapRemoval matches sparse_forbid & WASM default on N=80 > sparse_k=30
    assert np.array_equal(l_fb80_def.coords(copy=True), l_fb80_sp.coords(copy=True)), (
        "N=80 default ForbidOverlapRemoval must match variant='sparse_forbid'"
    )
    assert np.array_equal(lr_fb80_def.coords(copy=True), lr_fb80_sp.coords(copy=True)), (
        "N=80 default ForbidOverlapRemoval.from_radii must match variant='sparse_forbid'"
    )
    assert np.array_equal(ws["fb80"]["defCoords"], ws["fb80"]["spCoords"]), (
        "N=80 WASM default ForbidOverlapRemoval must match variant='sparse_forbid'"
    )
    check_close("forbid/n80_default_py_vs_wasm", l_fb80_x.coords(copy=True), ws["fb80"]["xCoords"], 1e-4)
    check_close("forbid/n80_radii_default_py_vs_wasm", lr_fb80_x.coords(copy=True), ws["fb80"]["radXCoords"], 1e-4)
    assert not np.allclose(l_fb80_def.coords(copy=True).ravel(), l_fb80_full.coords(copy=True).ravel(), atol=1e-3), (
        "N=80 default ForbidOverlapRemoval must diverge from variant='forbid'"
    )
    assert not np.allclose(lr_fb80_def.coords(copy=True).ravel(), lr_fb80_full.coords(copy=True).ravel(), atol=1e-3), (
        "N=80 default ForbidOverlapRemoval.from_radii must diverge from variant='forbid'"
    )
    assert not np.allclose(ws["fb80"]["defCoords"], ws["fb80"]["fullCoords"], atol=1e-3), (
        "N=80 WASM default ForbidOverlapRemoval must diverge from variant='forbid'"
    )

    return [{
        "case": "cross_target_overlap_cluster_metrics_py_vs_wasm",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
        "max_f32_diff": round(max_f32_diff, 12),
        "max_f64_diff": round(max_f64_diff, 12),
    }]
