"""Cross-target Python <-> WASM parity checks for clustering, coarsening, and manifolds."""

from __future__ import annotations

import pathlib
import time
from typing import Any

from parity.common import (
    check_array_close,
    make_asymmetric_coords,
    make_barbell_edges,
    make_grid_edges,
    resolve_wasm_entries,
    run_node_json,
)


def run_cross_target_clustering_manifold_parity(
    eg: Any, wasm_pkg: pathlib.Path
) -> list[dict[str, Any]]:
    """Verify Python float32/float64 <-> WASM f32 parity across clustering, coarsen, and manifolds."""
    import numpy as np

    if not hasattr(eg, "kmeans_plus_plus"):
        return []
    if resolve_wasm_entries(wasm_pkg) is None:
        return []

    t0 = time.perf_counter()

    # 1. Clustering & coarsening fixtures (12-node barbell)
    pts_km = np.array(
        [
            [10.0, 10.0],
            [10.2, 9.8],
            [9.8, 10.1],
            [-10.0, -10.0],
            [-9.9, -10.2],
            [-10.1, -9.8],
        ],
        dtype=np.float64,
    )
    km_py = eg.kmeans_plus_plus(pts_km, k=2, max_iterations=100, tol=1e-4, n_init=3, seed=42)

    bb_edges = make_barbell_edges(6)
    g_bb = eg.CsrGraph.from_edges(12, bb_edges, directed=False, sanitize=True)
    spec_py = eg.spectral_clustering(
        g_bb, k=2, max_eigen_iterations=300, eigen_tol=1e-4, kmeans_max_iterations=100, kmeans_tol=1e-4, seed=42
    )
    louv_py = eg.louvain_communities(g_bb, max_passes=100, resolution=1.0, min_modularity_gain=1e-7, seed=42)
    lpa_py = eg.label_propagation(g_bb, max_iterations=100, seed=42)
    cg_py, n2c_py, sizes_py, nclust_py = eg.coarsen_graph(g_bb, [100] * 6 + [999] * 6)
    coarse_coords_f32 = np.array([[-2.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    proj_py32 = eg.project_clustered_layout(
        eg.Layout2d.from_coords(coarse_coords_f32), n2c_py, spread_radius=0.2, seed=42
    ).coords(copy=True)
    proj_py64 = eg.project_clustered_layout(
        eg.Layout2d.from_coords(coarse_coords_f32.astype(np.float64)), n2c_py, spread_radius=0.2, seed=42
    ).coords(copy=True)

    # 2. ManifoldLayout2d & SgdSolver manifold stepping (all 3 manifolds)
    g16_edges = make_grid_edges(4, 4)
    g16 = eg.CsrGraph.from_edges(16, g16_edges, directed=False, sanitize=True)
    init16_f32 = np.array(make_asymmetric_coords(16, scale=0.8), dtype=np.float32)
    man_init_f32 = (init16_f32 * 0.25).astype(np.float32)
    man_sph_init_f32 = (man_init_f32 + 1.5).astype(np.float32)
    man_py32: dict[str, dict[str, Any]] = {}
    man_py64: dict[str, dict[str, Any]] = {}
    for kind in ("hyperbolic", "spherical", "torus"):
        m_step_scale = 0.02 if kind == "torus" else (0.2 if kind == "spherical" else 1.0)
        m_init = man_sph_init_f32 if kind == "spherical" else man_init_f32
        ml32 = eg.ManifoldLayout2d.from_coords(kind, m_init)
        sgd_m32 = eg.SgdSolver.full(
            g16, target="manifold2d", iterations=10, eps=0.01, schedule="exponential", step_scale=m_step_scale, seed=42, dtype="float32"
        )
        sgd_m32.run(ml32)
        man_py32[kind] = {
            "coords": ml32.coords(),
            "dist": float(ml32.geodesic_distance(0, 15)),
        }
        ml64 = eg.ManifoldLayout2d.from_coords(kind, m_init.astype(np.float64))
        sgd_m64 = eg.SgdSolver.full(
            g16, target="manifold2d", iterations=10, eps=0.01, schedule="exponential", step_scale=m_step_scale, seed=42, dtype="float64"
        )
        sgd_m64.run(ml64)
        man_py64[kind] = {
            "coords": ml64.coords(),
            "dist": float(ml64.geodesic_distance(0, 15)),
        }

    spec_payload = {
        "pts_km": [float(v) for v in pts_km.ravel()],
        "bb_edges": [int(v) for edge in bb_edges for v in edge],
        "g16_edges": [int(v) for edge in g16_edges for v in edge],
        "coarse_coords": [float(v) for v in coarse_coords_f32.ravel()],
        "man_init16": [float(v) for v in man_init_f32.ravel()],
        "man_sph_init16": [float(v) for v in man_sph_init_f32.ravel()],
    }

    node_body = """
if (typeof mod.kmeansPlusPlus !== 'function') {
  console.log(JSON.stringify({ clustering_available: false }));
  process.exit(0);
}

const km = mod.kmeansPlusPlus(new Float64Array(spec.pts_km), 6, 2, { k: 2, maxIterations: 100, tol: 1e-4, nInit: 3, seed: 42 });
const bb = mod.CsrGraph.fromEdgeList(12, new Uint32Array(spec.bb_edges));
const specRes = mod.spectralClustering(bb, { k: 2, maxEigenIterations: 300, eigenTol: 1e-4, kmeansMaxIterations: 100, kmeansTol: 1e-4, seed: 42 });
const specLabels = Array.from(specRes.labels());
const louv = mod.louvainCommunities(bb, { maxPasses: 100, resolution: 1.0, minModularityGain: 1e-7, seed: 42 });
const lpaRes = mod.labelPropagation(bb, 100, 42);
const lpaLabels = Array.from(lpaRes.labels());
const coarse = mod.coarsenGraph(bb, [100, 100, 100, 100, 100, 100, 999, 999, 999, 999, 999, 999]);
const cg = coarse.graph();
const cl = mod.Layout2d.fromCoords(new Float32Array(spec.coarse_coords));
const proj = mod.projectClusteredLayout(cl, coarse.nodeToCoarse(), { spreadRadius: 0.2, seed: 42 });
const projCoords = Array.from(proj.coordsView());

const g16 = mod.CsrGraph.fromEdgeList(16, new Uint32Array(spec.g16_edges));
const manInit = new Float32Array(spec.man_init16);
const manSphInit = new Float32Array(spec.man_sph_init16);

const manifolds = {};
for (const [pyKind, wasmKind] of [['hyperbolic', 'hyperbolicPoincare'], ['spherical', 'spherical'], ['torus', 'torus']]) {
  const mStepScale = pyKind === 'torus' ? 0.02 : (pyKind === 'spherical' ? 0.2 : 1.0);
  const mInit = pyKind === 'spherical' ? manSphInit : manInit;
  const mlRun = mod.ManifoldLayout2d.fromCoords(wasmKind, mInit);
  const mlStep = mod.ManifoldLayout2d.fromCoords(wasmKind, mInit);
  const sRun = mod.SgdSolver.full(g16, { target: 'manifold2d', iterations: 10, eps: 0.01, kind: 'exponential', stepScale: mStepScale, seed: 42 });
  const sStep = mod.SgdSolver.full(g16, { target: 'manifold2d', iterations: 10, eps: 0.01, kind: 'exponential', stepScale: mStepScale, seed: 42 });
  sRun.run(mlRun);
  sStep.runSteps(mlStep, 4);
  while (!sStep.isFinished) sStep.step(mlStep);
  const cRun = Array.from(mlRun.coordsView());
  const cStep = Array.from(mlStep.coordsView());
  if (JSON.stringify(cRun) !== JSON.stringify(cStep)) {
    throw new Error(`WASM SgdSolver.step != run on ManifoldLayout2d for ${wasmKind}`);
  }
  manifolds[pyKind] = { coords: cRun, dist: mlRun.geodesicDistance(0, 15) };
  mlRun.free?.();
  mlStep.free?.();
  sRun.free?.();
  sStep.free?.();
}

const out = {
  clustering_available: true,
  km: {
    labels: Array.from(km.labels()),
    centroids: Array.from(km.centroids()),
    inertia: km.inertia,
    iterations: km.iterations,
    converged: km.converged,
  },
  spectral: specLabels,
  louvain: {
    labels: Array.from(louv.labels()),
    numCommunities: louv.numCommunities,
    modularity: louv.modularity,
    passes: louv.passes,
  },
  lpa: lpaLabels,
  coarsen: {
    numNodes: cg.numNodes(),
    numEdges: cg.numEdges(),
    weights: Array.from(cg.weights() ?? []),
    nodeToCoarse: Array.from(coarse.nodeToCoarse()),
    coarseSizes: Array.from(coarse.coarseSizes()),
    numClusters: coarse.numClusters,
  },
  projCoords,
  manifolds,
};

g16.free?.(); proj.free?.(); cl.free?.(); cg.free?.(); coarse.free?.(); lpaRes.free?.(); louv.free?.(); specRes.free?.(); km.free?.(); bb.free?.();

console.log(JSON.stringify(out));
"""
    ws = run_node_json(wasm_pkg, node_body, spec_payload)
    if not ws.get("clustering_available", False):
        return [{
            "case": "cross_target_clustering_manifold_py_vs_wasm",
            "status": "skipped_no_wasm_clustering",
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

    assert list(km_py.labels) == ws["km"]["labels"]
    assert bool(km_py.converged) == bool(ws["km"]["converged"])
    assert int(km_py.iterations) == int(ws["km"]["iterations"])
    check_close("kmeans/centroids", km_py.centroids, ws["km"]["centroids"], 1e-12, is_f32=False)
    check_close("kmeans/inertia", [km_py.inertia], [ws["km"]["inertia"]], 1e-12, is_f32=False)
    assert list(spec_py.labels) == ws["spectral"]
    assert list(louv_py.labels) == ws["louvain"]["labels"]
    assert int(louv_py.num_communities) == int(ws["louvain"]["numCommunities"])
    assert int(louv_py.passes) == int(ws["louvain"]["passes"])
    check_close("louvain/modularity", [louv_py.modularity], [ws["louvain"]["modularity"]], 1e-12, is_f32=False)
    assert list(lpa_py.labels) == ws["lpa"]
    assert cg_py.num_nodes == ws["coarsen"]["numNodes"]
    assert cg_py.num_edges == ws["coarsen"]["numEdges"]
    assert list(n2c_py) == ws["coarsen"]["nodeToCoarse"]
    assert list(sizes_py) == ws["coarsen"]["coarseSizes"]
    assert nclust_py == ws["coarsen"]["numClusters"]
    check_close("project_clustered_layout/f32", proj_py32, ws["projCoords"], 1e-4)
    check_close("project_clustered_layout/f64_vs_f32", proj_py64, ws["projCoords"], 1e-3, is_f32=False)

    for kind in ("hyperbolic", "spherical", "torus"):
        check_close(f"manifold_{kind}/coords_f32", man_py32[kind]["coords"], ws["manifolds"][kind]["coords"], 1e-4)
        check_close(f"manifold_{kind}/dist_f32", [man_py32[kind]["dist"]], [ws["manifolds"][kind]["dist"]], 1e-4)
        check_close(f"manifold_{kind}/coords_f64_vs_f32", man_py64[kind]["coords"], ws["manifolds"][kind]["coords"], 1e-3, is_f32=False)

    return [{
        "case": "cross_target_clustering_manifold_py_vs_wasm",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
        "max_f32_diff": round(max_f32_diff, 12),
        "max_f64_diff": round(max_f64_diff, 12),
    }]
