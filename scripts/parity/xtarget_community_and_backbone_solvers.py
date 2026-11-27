"""Cross-target Python <-> WASM parity checks for Leiden, LinLog, FR, FA2, backbone, and GroupInABox."""

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


def run_cross_target_community_backbone_parity(
    eg: Any, wasm_pkg: pathlib.Path
) -> list[dict[str, Any]]:
    """Verify Python float32 <-> WASM f32 parity across Leiden, LinLog, FR, backbone, and GroupInABox."""
    import numpy as np

    if not hasattr(eg, "leiden_communities"):
        return []
    if resolve_wasm_entries(wasm_pkg) is None:
        return []

    t0 = time.perf_counter()

    bb_edges = make_barbell_edges(5)
    g_bb = eg.CsrGraph.from_edges(10, bb_edges, directed=False, sanitize=True)

    init10_f32 = np.array(make_asymmetric_coords(10, scale=1.2), dtype=np.float32)
    swapped_f32 = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [0.1, -0.2],
            [0.3, 0.0],
            [10.0, 0.1],
            [10.0, 0.2],
            [10.2, 0.1],
            [10.1, -0.2],
            [10.3, 0.0],
            [0.0, 0.2],
        ],
        dtype=np.float32,
    )
    comm_labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    radii10 = [0.6] * 10

    leiden_mod_py = eg.leiden_communities(g_bb, objective="modularity", resolution=1.0, seed=42)
    leiden_cpm_py = eg.leiden_communities(g_bb, objective="cpm", resolution=0.3, seed=42)

    linlog_py: dict[str, dict[str, Any]] = {}
    for rep in ("edge_repulsion", "node_repulsion"):
        ll = eg.LinLogSolver(
            g_bb, iterations=8, repulsion_model=rep, use_barnes_hut=True, seed=42, dtype="float32"
        )
        lay = eg.Layout2d.from_coords(init10_f32)
        ll.run(lay)
        linlog_py[rep] = {
            "coords": lay.coords(copy=True),
            "energy": float(ll.energy(lay)),
            "max_disp": float(ll.max_displacement),
        }

    fr_py: dict[str, dict[str, Any]] = {}
    for backend in ("exact", "barnes_hut", "grid"):
        fr = eg.FruchtermanReingoldSolver(g_bb, iterations=8, backend=backend, seed=42, dtype="float32")
        lay = eg.Layout2d.from_coords(init10_f32)
        fr.run(lay)
        fr_py[backend] = {
            "coords": lay.coords(copy=True),
            "opt_dist": float(fr.optimal_distance),
        }

    fa2_ov = eg.ForceAtlas2Solver(
        g_bb, iterations=8, prevent_overlap=True, default_node_radius=0.5, node_radii=radii10, seed=42, dtype="float32"
    )
    l_fa2_py = eg.Layout2d.from_coords(init10_f32)
    fa2_ov.run(l_fa2_py)
    fa2_coords_py = l_fa2_py.coords(copy=True)
    fa2_radii_py = fa2_ov.node_radii

    bb_ex_py = eg.extract_quadrilateral_backbone(g_bb, retain_ratio=0.3)
    bb_sol_py: dict[str, Any] = {}
    for mode in ("stress", "sgd"):
        bs = eg.BackboneSolver(g_bb, retain_ratio=0.3, layout_mode=mode, iterations=8, seed=42, dtype="float32")
        lay = eg.Layout2d.from_coords(init10_f32)
        bs.run(lay)
        bb_sol_py[mode] = lay.coords(copy=True)

    cr_py: dict[str, dict[str, Any]] = {}
    for method in ("kmeans", "hac"):
        cs = eg.ClusterRefineSolver.from_graph(
            g_bb, comm_labels, method=method, max_iterations=5, seed=42, dtype="float32"
        )
        lay = eg.Layout2d.from_coords(swapped_f32)
        cs.run(lay)
        cr_py[method] = {
            "coords": lay.coords(copy=True),
            "moved": int(cs.last_moved_count),
            "clusters": int(cs.num_clusters),
        }
    l_cr_fn = eg.Layout2d.from_coords(swapped_f32)
    cr_fn_moved_py = int(eg.refine_cluster_layout(l_cr_fn, comm_labels, method="kmeans", max_iterations=5, seed=42))
    cr_fn_coords_py = l_cr_fn.coords(copy=True)

    treemap_py = eg.squarified_treemap([5.0, 3.0, 2.0], width=100.0, height=50.0, x=5.0, y=10.0)
    gib_py: dict[str, dict[str, Any]] = {}
    for bmode, imode, cw in (("treemap", "fr", [2.0, 1.0]), ("force_directed", "yifan_hu", None)):
        gs = eg.GroupInABoxSolver(
            g_bb, comm_labels, box_mode=bmode, intra_mode=imode, intra_iterations=6,
            fd_iterations=6, cluster_weights=cw, seed=42, dtype="float32",
        )
        lay = eg.Layout2d.from_coords(init10_f32)
        gs.run(lay)
        gib_py[f"{bmode}_{imode}"] = {
            "coords": lay.coords(copy=True),
            "boxes": gs.boxes(),
        }

    spec_payload = {
        "bb_edges": [int(v) for edge in bb_edges for v in edge],
        "init10": [float(v) for v in init10_f32.ravel()],
        "swapped": [float(v) for v in swapped_f32.ravel()],
        "comm_labels": comm_labels,
        "radii10": radii10,
    }

    node_body = """
if (typeof mod.leidenCommunities !== 'function') {
  console.log(JSON.stringify({ community_available: false }));
  process.exit(0);
}

const gBb = mod.CsrGraph.fromEdgeList(10, new Uint32Array(spec.bb_edges), null, false, true);
const init10 = new Float32Array(spec.init10);
const swapped = new Float32Array(spec.swapped);
const commLabels = new Uint32Array(spec.comm_labels);

const lMod = mod.leidenCommunities(gBb, { objective: 'modularity', resolution: 1.0, seed: 42 });
const lCpm = mod.leidenCommunities(gBb, { objective: 'cpm', resolution: 0.3, seed: 42 });

const linlog = {};
for (const rep of ['edge_repulsion', 'node_repulsion']) {
  const s = new mod.LinLogSolver(gBb, { iterations: 8, repulsionModel: rep, useBarnesHut: true, seed: 42 });
  const l = mod.Layout2d.fromCoords(init10);
  s.run(l);
  linlog[rep] = { coords: Array.from(l.coordsView()), energy: s.energy(l), maxDisp: s.maxDisplacement };
  l.free?.(); s.free?.();
}

const fr = {};
for (const backend of ['exact', 'barnes_hut', 'grid']) {
  const s = new mod.FruchtermanReingoldSolver(gBb, { iterations: 8, backend, seed: 42 });
  const l = mod.Layout2d.fromCoords(init10);
  s.run(l);
  fr[backend] = { coords: Array.from(l.coordsView()), optDist: s.optimalDistance };
  l.free?.(); s.free?.();
}

const fa2 = new mod.ForceAtlas2Solver(gBb, {
  iterations: 8, preventOverlap: true, defaultNodeRadius: 0.5, nodeRadii: spec.radii10, seed: 42,
});
const lFa2 = mod.Layout2d.fromCoords(init10);
fa2.run(lFa2);
const fa2Out = { coords: Array.from(lFa2.coordsView()), radii: Array.from(fa2.nodeRadii()) };
lFa2.free?.(); fa2.free?.();

const bbEx = mod.extractQuadrilateralBackbone(gBb, { retainRatio: 0.3 });
const bbCsr = bbEx.backboneCsr();
const bbSol = {};
for (const mode of ['stress', 'sgd']) {
  const s = new mod.BackboneSolver(gBb, { retainRatio: 0.3, layoutMode: mode, iterations: 8, seed: 42 });
  const l = mod.Layout2d.fromCoords(init10);
  s.run(l);
  bbSol[mode] = Array.from(l.coordsView());
  l.free?.(); s.free?.();
}

const cr = {};
for (const method of ['kmeans', 'hac']) {
  const s = mod.ClusterRefineSolver.fromGraph(gBb, commLabels, { method, maxIterations: 5, seed: 42 });
  const l = mod.Layout2d.fromCoords(swapped);
  s.run(l);
  cr[method] = { coords: Array.from(l.coordsView()), moved: s.lastMovedCount, clusters: s.numClusters };
  l.free?.(); s.free?.();
}
const lCrFn = mod.Layout2d.fromCoords(swapped);
const crFnMoved = mod.refineClusterLayout(lCrFn, commLabels, { method: 'kmeans', maxIterations: 5, seed: 42 });
const crFnCoords = Array.from(lCrFn.coordsView());
lCrFn.free?.();

const treemap = Array.from(mod.squarifiedTreemap([5.0, 3.0, 2.0], 100.0, 50.0, 5.0, 10.0));
const gib = {};
for (const [bm, im, cw] of [['treemap', 'fr', [2.0, 1.0]], ['force_directed', 'yifan_hu', null]]) {
  const s = new mod.GroupInABoxSolver(gBb, commLabels, {
    boxMode: bm, intraMode: im, intraIterations: 6, fdIterations: 6, clusterWeights: cw, seed: 42,
  });
  const l = mod.Layout2d.fromCoords(init10);
  s.run(l);
  gib[`${bm}_${im}`] = { coords: Array.from(l.coordsView()), boxes: s.boxes() };
  l.free?.(); s.free?.();
}

const out = {
  community_available: true,
  leidenMod: { labels: Array.from(lMod.labels()), numCommunities: lMod.numCommunities, quality: lMod.quality, modularity: lMod.modularity, passes: lMod.passes },
  leidenCpm: { labels: Array.from(lCpm.labels()), numCommunities: lCpm.numCommunities, quality: lCpm.quality, modularity: lCpm.modularity, passes: lCpm.passes },
  linlog, fr, fa2Out,
  bbEx: {
    umstEdges: Array.from(bbEx.umstEdges()),
    backboneEdges: Array.from(bbEx.backboneEdges()),
    simmelianWeights: Array.from(bbEx.simmelianWeights()),
    edgeWeights: Array.from(bbEx.edgeWeights()),
    csrNodes: bbCsr.numNodes(),
    csrEdges: bbCsr.numEdges(),
  },
  bbSol, cr, crFn: { moved: crFnMoved, coords: crFnCoords }, treemap, gib,
};

bbCsr.free?.(); bbEx.free?.(); lCpm.free?.(); lMod.free?.(); gBb.free?.();
console.log(JSON.stringify(out));
"""
    ws = run_node_json(wasm_pkg, node_body, spec_payload)
    if not ws.get("community_available", False):
        return [{
            "case": "cross_target_community_backbone_py_vs_wasm",
            "status": "skipped_no_wasm_community_backbone",
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

    for tag, py_r, ws_r in (("mod", leiden_mod_py, ws["leidenMod"]), ("cpm", leiden_cpm_py, ws["leidenCpm"])):
        assert list(py_r.labels) == list(ws_r["labels"]), f"leiden/{tag} labels mismatch"
        assert py_r.num_communities == ws_r["numCommunities"]
        assert py_r.passes == ws_r["passes"]
        check_close(f"leiden/{tag}/quality_mod", [py_r.quality, py_r.modularity], [ws_r["quality"], ws_r["modularity"]], 1e-12, is_f32=False)

    for rep in ("edge_repulsion", "node_repulsion"):
        check_close(f"linlog/{rep}/coords", linlog_py[rep]["coords"], ws["linlog"][rep]["coords"], 1e-4)
        check_close(f"linlog/{rep}/scalars", [linlog_py[rep]["energy"], linlog_py[rep]["max_disp"]], [ws["linlog"][rep]["energy"], ws["linlog"][rep]["maxDisp"]], 1e-4)
    for backend in ("exact", "barnes_hut", "grid"):
        check_close(f"fr/{backend}/coords", fr_py[backend]["coords"], ws["fr"][backend]["coords"], 1e-4)
        check_close(f"fr/{backend}/opt_dist", [fr_py[backend]["opt_dist"]], [ws["fr"][backend]["optDist"]], 1e-12, is_f32=False)
    check_close("fa2_overlap/coords", fa2_coords_py, ws["fa2Out"]["coords"], 1e-4)
    check_close("fa2_overlap/radii", fa2_radii_py, ws["fa2Out"]["radii"], 1e-12, is_f32=False)

    assert list(bb_ex_py["umst_edges"].ravel()) == ws["bbEx"]["umstEdges"]
    assert list(bb_ex_py["backbone_edges"].ravel()) == ws["bbEx"]["backboneEdges"]
    assert bb_ex_py["backbone_csr"].num_nodes == ws["bbEx"]["csrNodes"]
    assert bb_ex_py["backbone_csr"].num_edges == ws["bbEx"]["csrEdges"]
    check_close("backbone/simmelian", bb_ex_py["simmelian_weights"], ws["bbEx"]["simmelianWeights"], 1e-12, is_f32=False)
    check_close("backbone/edge_weights", bb_ex_py["edge_weights"], ws["bbEx"]["edgeWeights"], 1e-12, is_f32=False)
    for mode in ("stress", "sgd"):
        check_close(f"backbone_solver/{mode}", bb_sol_py[mode], ws["bbSol"][mode], 1e-4)

    for method in ("kmeans", "hac"):
        assert cr_py[method]["moved"] == ws["cr"][method]["moved"]
        assert cr_py[method]["clusters"] == ws["cr"][method]["clusters"]
        check_close(f"cluster_refine/{method}", cr_py[method]["coords"], ws["cr"][method]["coords"], 1e-4)
    assert cr_fn_moved_py == ws["crFn"]["moved"]
    check_close("refine_cluster_layout", cr_fn_coords_py, ws["crFn"]["coords"], 1e-4)
    check_close("squarified_treemap", treemap_py, ws["treemap"], 1e-12, is_f32=False)
    for key in ("treemap_fr", "force_directed_yifan_hu"):
        check_close(f"gib/{key}/coords", gib_py[key]["coords"], ws["gib"][key]["coords"], 1e-4)
        assert len(gib_py[key]["boxes"]) == len(ws["gib"][key]["boxes"])
        for pb, wb in zip(gib_py[key]["boxes"], ws["gib"][key]["boxes"]):
            assert pb["cluster_id"] == wb["clusterId"]
            assert pb["node_count"] == wb["nodeCount"]
            check_close(
                f"gib/{key}/box_{pb['cluster_id']}",
                [pb["x_min"], pb["x_max"], pb["y_min"], pb["y_max"], pb["weight"]],
                [wb["xMin"], wb["xMax"], wb["yMin"], wb["yMax"], wb["weight"]],
                1e-4,
            )

    return [{
        "case": "cross_target_community_backbone_py_vs_wasm",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
        "max_f32_diff": round(max_f32_diff, 12),
        "max_f64_diff": round(max_f64_diff, 12),
    }]
