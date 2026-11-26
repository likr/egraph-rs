"""Cross-target Python <-> WASM parity checks for CsrGraph, datasets, Layout2d/Nd, and manifolds."""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import math
import pathlib
import time
from typing import Any

from parity.common import check_array_close, resolve_wasm_entries, run_node_json


def run_cross_target_core_dataset_manifold_parity(
    eg: Any, wasm_pkg: pathlib.Path
) -> list[dict[str, Any]]:
    """Verify Python <-> WASM parity across CsrGraph, datasets, Layout2d/Nd, manifolds, and __version__."""
    import numpy as np

    assert hasattr(eg, "LayoutNd"), "Python missing LayoutNd"
    assert hasattr(eg, "load_dataset"), "Python missing load_dataset"
    assert resolve_wasm_entries(wasm_pkg) is not None, f"Missing WASM index.js in {wasm_pkg}"

    t0 = time.perf_counter()

    py_pkg_ver = importlib_metadata.version("egraph")
    assert eg.__version__ == eg._core.__version__ == py_pkg_ver, (
        f"__version__ mismatch: {eg.__version__!r} vs {eg._core.__version__!r} vs {py_pkg_ver!r}"
    )

    # 1. CsrGraph methods & LCC + dataset/CSV
    g_disc = eg.parse_csr_csv("6,5\n0,1\n2,3\n3,4\n4,5\n5,2\n", directed=False, sanitize=True)
    assert g_disc.nnz == 10
    assert g_disc.is_weighted is False
    assert g_disc.degree(2) == 2
    assert list(g_disc.neighbors(2)) == [3, 5]
    sub_py = g_disc.induced_subgraph([2, 3, 4])
    lcc_py, lcc_ids_py = g_disc.largest_connected_component()
    karate_py = eg.load_dataset("karate")

    # 2. Layout2d methods & LayoutNd
    l2_empty = eg.Layout2d.zeros(0, dtype="float32")
    assert l2_empty.is_empty() is True
    l2_py = eg.Layout2d.zeros(4, dtype="float32")
    assert l2_py.num_nodes == 4
    assert l2_py.is_empty() is False
    l2_py.set_x(0, 1.0)
    l2_py.set_y(0, -1.0)
    l2_py.set_x(1, 3.5)
    l2_py.set_y(1, 2.5)
    l2_x1, l2_y1 = l2_py.x(1), l2_py.y(1)
    assert math.isclose(l2_x1, 3.5, rel_tol=1e-6)
    assert math.isclose(l2_y1, 2.5, rel_tol=1e-6)
    l2_py.normalize_std(target_std=1.5)
    l2_norm_coords = l2_py.copy_coords()

    l2_init_py = eg.Layout2d.init(5, seed=42, dtype="float32")
    assert l2_init_py.num_nodes == 5
    l2_init_coords_py = l2_init_py.copy_coords()

    nd_zero = eg.LayoutNd.zeros(3, 4, dtype="float32")
    assert nd_zero.num_nodes == 3 and nd_zero.dim == 4 and nd_zero.is_all_finite() and not nd_zero.is_empty()
    nd_arr_f32 = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    nd_py = eg.LayoutNd.from_coords(nd_arr_f32)
    assert nd_py.num_nodes == 3 and nd_py.dim == 4 and nd_py.is_all_finite()

    # 3. ManifoldLayout2d methods & geodesic helpers
    ml_hyp_py = eg.ManifoldLayout2d.hyperbolic_poincare(2, dtype="float32")
    ml_sph_py = eg.ManifoldLayout2d.spherical(2, dtype="float32")
    ml_tor_py = eg.ManifoldLayout2d.torus_2d(2, dtype="float32")
    assert ml_hyp_py.manifold == "hyperbolic" and ml_hyp_py.num_nodes == 2
    assert ml_sph_py.manifold == "spherical" and ml_sph_py.num_nodes == 2
    assert ml_tor_py.manifold == "torus" and ml_tor_py.num_nodes == 2

    ml_py = eg.ManifoldLayout2d.zeros("hyperbolic", 2, dtype="float32")
    ml_py.set_point(0, 0.1, 0.2)
    ml_py.set_point(1, -0.1, 0.05)
    ml_x0, ml_y0 = ml_py.x(0), ml_py.y(0)
    td_vx, td_vy = ml_py.tangent_delta(0, 1)
    ml_py.apply_tangent_step(0, td_vx * 0.1, td_vy * 0.1)
    ml_coords_after = ml_py.coords()
    ml_rand_py = eg.ManifoldLayout2d.random_init("hyperbolic", 4, scale=0.25, seed=42, dtype="float32")
    ml_rand_py.center_poincare_centroid()
    ml_rand_cent_py = ml_rand_py.poincare_centroid()
    ml_scaled_py = eg.ManifoldLayout2d.from_layout_scaled(
        "hyperbolic", eg.Layout2d.from_coords(np.array([[10.0, -5.0], [20.0, 5.0], [30.0, 15.0]], dtype=np.float32)), max_radius=0.7
    )
    ml_scaled_py.recenter_poincare(0.1, -0.1)
    ml_scaled_py.recenter_poincare_at(1)
    geo_dists_py = [
        float(eg.poincare_distance(0.0, 0.0, 0.3, 0.4)),
        float(eg.spherical_distance(0.0, 0.0, 0.0, math.pi / 2.0)),
        float(eg.torus_distance(0.05, 0.05, 0.95, 0.05)),
        *[float(v) for v in eg.wrap_torus_point(1.25, -0.25)],
        *[float(v) for v in eg.poincare_mobius_add(0.2, -0.1, -0.15, 0.25)],
    ]

    spec_payload = {"nd_coords": [float(v) for v in nd_arr_f32.ravel()]}

    node_body = """
assert.equal(typeof mod.LayoutNd, 'function', 'WASM missing LayoutNd');
assert.equal(typeof mod.loadDataset, 'undefined', 'WASM must not expose embedded loadDataset');
assert.equal(typeof indexMod.runProgressive, 'function', 'WASM index.js missing runProgressive');

const gDisc = mod.parseCsrCsv("6,5\\n0,1\\n2,3\\n3,4\\n4,5\\n5,2\\n", false, true);
const sub = gDisc.inducedSubgraph(new Uint32Array([2, 3, 4]));
const lcc = gDisc.largestConnectedComponent();
const lccG = lcc.graph();

const l2Empty = mod.Layout2d.zeros(0);
assert.equal(l2Empty.isEmpty(), true);
const l2 = mod.Layout2d.zeros(4);
assert.equal(l2.numNodes, 4);
assert.equal(l2.isEmpty(), false);
l2.setX(0, 1.0); l2.setY(0, -1.0);
l2.setX(1, 3.5); l2.setY(1, 2.5);
const l2x1 = l2.x(1);
const l2y1 = l2.y(1);
l2.normalizeStd(1.5);
const l2NormCoords = Array.from(l2.copyCoords());
const l2Init = mod.Layout2d.init(5, 42);
assert.equal(l2Init.numNodes, 5);
const l2InitCoords = Array.from(l2Init.copyCoords());

const ndZero = mod.LayoutNd.zeros(3, 4);
assert.equal(ndZero.numNodes, 3);
assert.equal(ndZero.dim, 4);
assert.equal(ndZero.isAllFinite(), true);
assert.equal(ndZero.isEmpty(), false);
const lNd = mod.LayoutNd.fromCoords(3, 4, new Float32Array(spec.nd_coords));
const ndCoords = Array.from(lNd.copyCoords());

const mlHyp = mod.ManifoldLayout2d.hyperbolicPoincare(2);
const mlSph = mod.ManifoldLayout2d.spherical(2);
const mlTor = mod.ManifoldLayout2d.torus2d(2);
assert.equal(mlHyp.manifold, 'hyperbolic');
assert.equal(mlHyp.numNodes, 2);
assert.equal(mlSph.manifold, 'spherical');
assert.equal(mlSph.numNodes, 2);
assert.equal(mlTor.manifold, 'torus');
assert.equal(mlTor.numNodes, 2);
const ml = mod.ManifoldLayout2d.hyperbolicPoincare(2);
ml.setPoint(0, 0.1, 0.2);
ml.setPoint(1, -0.1, 0.05);
const mlx0 = ml.x(0);
const mly0 = ml.y(0);
const td = ml.tangentDelta(0, 1);
ml.applyTangentStep(0, td[0] * 0.1, td[1] * 0.1);
const mlCoords = Array.from(ml.coords());
const mlRand = mod.ManifoldLayout2d.randomInit('hyperbolic', 4, 0.25, 42);
mlRand.centerPoincareCentroid();
const mlRandCent = Array.from(mlRand.poincareCentroid());
const lScaleIn = mod.Layout2d.fromCoords(new Float32Array([10, -5, 20, 5, 30, 15]));
const mlScaled = mod.ManifoldLayout2d.fromLayoutScaled('hyperbolic', lScaleIn, 0.7);
mlScaled.recenterPoincare(0.1, -0.1);
mlScaled.recenterPoincareAt(1);
const mlScaledCoords = Array.from(mlScaled.coords());
const mlRandCoords = Array.from(mlRand.coords());
const geoDists = [
  mod.poincareDistance(0, 0, 0.3, 0.4),
  mod.sphericalDistance(0, 0, 0, Math.PI / 2),
  mod.torusDistance(0.05, 0.05, 0.95, 0.05),
  ...Array.from(mod.wrapTorusPoint(1.25, -0.25)),
  ...Array.from(mod.poincareMobiusAdd(0.2, -0.1, -0.15, 0.25)),
];

const out = {
  disc: { nnz: gDisc.nnz(), isW: gDisc.isWeighted(), deg2: gDisc.degree(2), nbr2: Array.from(gDisc.neighbors(2)), subN: sub.numNodes(), subE: sub.numEdges(), lccN: lccG.numNodes(), lccIds: Array.from(lcc.originalNodeIds()) },
  l2: { x1: l2x1, y1: l2y1, normCoords: l2NormCoords, initCoords: l2InitCoords },
  ndCoords,
  ml: { x0: mlx0, y0: mly0, td: Array.from(td), coords: mlCoords, geoDists, randCoords: mlRandCoords, randCent: mlRandCent, scaledCoords: mlScaledCoords },
};

mlScaled.free?.(); lScaleIn.free?.(); mlRand.free?.();
ml.free?.(); mlHyp.free?.(); mlSph.free?.(); mlTor.free?.(); lNd.free?.(); ndZero.free?.();
l2Init.free?.(); l2.free?.(); l2Empty.free?.(); lccG.free?.(); lcc.free?.(); sub.free?.(); gDisc.free?.();

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

    assert ws["disc"]["nnz"] == g_disc.nnz
    assert ws["disc"]["isW"] == g_disc.is_weighted
    assert ws["disc"]["deg2"] == g_disc.degree(2)
    assert ws["disc"]["nbr2"] == list(g_disc.neighbors(2))
    assert ws["disc"]["subN"] == sub_py.num_nodes and ws["disc"]["subE"] == sub_py.num_edges
    assert ws["disc"]["lccN"] == lcc_py.num_nodes and ws["disc"]["lccIds"] == list(lcc_ids_py)
    assert karate_py.num_nodes == 34 and karate_py.num_edges >= 78

    check_close("Layout2d.x/y", [l2_x1, l2_y1], [ws["l2"]["x1"], ws["l2"]["y1"]], 1e-6)
    check_close("Layout2d.normalize_std", l2_norm_coords, ws["l2"]["normCoords"], 1e-5)
    check_close("Layout2d.init", l2_init_coords_py, ws["l2"]["initCoords"], 1e-6)
    check_close("LayoutNd.from_coords", nd_py.coords(copy=True), ws["ndCoords"], 1e-6)

    check_close("ManifoldLayout2d.x/y", [ml_x0, ml_y0], [ws["ml"]["x0"], ws["ml"]["y0"]], 1e-6)
    check_close("ManifoldLayout2d.tangent_delta", [td_vx, td_vy], ws["ml"]["td"], 1e-5)
    check_close("ManifoldLayout2d.apply_tangent_step", ml_coords_after, ws["ml"]["coords"], 1e-5)
    check_close("ManifoldLayout2d.random_init+center_poincare_centroid", ml_rand_py.coords(), ws["ml"]["randCoords"], 1e-5)
    check_close("ManifoldLayout2d.poincare_centroid", ml_rand_cent_py, ws["ml"]["randCent"], 1e-5)
    check_close("ManifoldLayout2d.from_layout_scaled+recenter", ml_scaled_py.coords(), ws["ml"]["scaledCoords"], 1e-5)
    check_close("geodesic_distances", geo_dists_py, ws["ml"]["geoDists"], 1e-12, is_f32=False)

    return [{
        "case": "cross_target_core_dataset_manifold_py_vs_wasm",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
        "max_f32_diff": round(max_f32_diff, 12),
        "max_f64_diff": round(max_f64_diff, 12),
    }]
