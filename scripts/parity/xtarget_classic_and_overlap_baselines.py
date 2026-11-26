"""Cross-target Python <-> WASM parity checks for MaxentStress, GRIP, GEM, SEL, PFS', and SFC."""

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


def run_cross_target_classic_overlap_baselines_parity(
    eg: Any, wasm_pkg: pathlib.Path
) -> list[dict[str, Any]]:
    """Verify Python float32 <-> WASM f32 parity across MaxentStress, GRIP, GEM, SEL, PFS', and SFC."""
    import numpy as np

    if not hasattr(eg, "MaxentStressSolver"):
        return []
    if resolve_wasm_entries(wasm_pkg) is None:
        return []

    t0 = time.perf_counter()

    bb_edges = make_barbell_edges(5)
    g_bb = eg.CsrGraph.from_edges(10, bb_edges, directed=False, sanitize=True)
    grid_edges = make_grid_edges(4, 4)
    g_grid = eg.CsrGraph.from_edges(16, grid_edges, directed=False, sanitize=True)
    star_edges = [(0, i) for i in range(1, 10)]
    g_star = eg.CsrGraph.from_edges(10, star_edges, directed=False, sanitize=True)

    init10_f32 = np.array(make_asymmetric_coords(10, scale=1.2), dtype=np.float32)
    ov_f32 = (init10_f32 * 0.15).astype(np.float32)
    ov_f32[1] = ov_f32[0]
    col_f32 = np.column_stack([
        np.linspace(-1.0, 1.0, 10, dtype=np.float32),
        np.zeros(10, dtype=np.float32),
    ]).astype(np.float32)
    hw10_f32 = [0.45] * 10
    hh10_f32 = [0.35] * 10
    radii10_f32 = [0.50] * 10

    maxent_py: dict[str, dict[str, Any]] = {}
    for use_bh in (True, False):
        tag = "bh" if use_bh else "exact"
        ms = eg.MaxentStressSolver(
            g_bb, q=0.4, stages=2, steps_per_stage=5, use_barnes_hut=use_bh, seed=42, dtype="float32"
        )
        lay = eg.Layout2d.from_coords(init10_f32)
        ms.run(lay)
        maxent_py[tag] = {
            "coords": lay.coords(copy=True),
            "q": float(ms.effective_q),
            "alpha": float(ms.current_alpha),
        }
    maxent_auto_grid_py = float(eg.MaxentStressSolver(g_grid, dtype="float32").effective_q)
    maxent_auto_star_py = float(eg.MaxentStressSolver(g_star, dtype="float32").effective_q)

    grip_s = eg.GripSolver(g_bb, rounds_min=3, rounds_max=8, seed=42, dtype="float32")
    l_grip = eg.Layout2d.from_coords(init10_f32)
    grip_s.run(l_grip)
    grip_py = {"coords": l_grip.coords(copy=True), "num_levels": int(grip_s.num_levels)}

    gem_s = eg.GemSolver(g_bb, max_rounds=12, seed=42, dtype="float32")
    l_gem = eg.Layout2d.from_coords(init10_f32)
    gem_s.run(l_gem)
    gem_py = l_gem.coords(copy=True)

    sel_py: dict[str, dict[str, Any]] = {}
    for mapping, backend in (("sigmoid", "fr"), ("tanh", "t_fdp")):
        ss = eg.SelLayoutSolver(
            g_bb, mapping=mapping, layout_backend=backend, iterations=8, seed=42, dtype="float32"
        )
        lay = eg.Layout2d.from_coords(init10_f32)
        ss.run(lay)
        sel_py[f"{mapping}_{backend}"] = {
            "coords": lay.coords(copy=True),
            "se_target": float(ss.se_target),
            "weights": ss.modulated_weights(),
        }

    pfs_box = eg.PfsPrimeOverlapRemoval(hw10_f32, hh10_f32, padding=0.02, seed=42, dtype="float32")
    l_pfs_box = eg.Layout2d.from_coords(ov_f32)
    pfs_box.run(l_pfs_box)
    pfs_rad = eg.PfsPrimeOverlapRemoval.from_radii(radii10_f32, padding=0.0, seed=42, dtype="float32")
    l_col_orig = eg.Layout2d.from_coords(col_f32)
    l_pfs_rad = eg.Layout2d.from_coords(col_f32)
    pfs_rad.run(l_pfs_rad)
    pfs_nor_py = eg.nor_metrics(l_col_orig, l_pfs_rad, radii10_f32, radii10_f32, graph=g_bb)

    sfc_py: dict[str, dict[str, Any]] = {}
    for curve in ("hilbert", "morton"):
        sc = eg.SfcLayoutSolver(g_bb, curve=curve, scale=80.0, seed=42, dtype="float32")
        lay = eg.Layout2d.zeros(10, dtype="float32")
        sc.run(lay)
        sfc_py[curve] = {
            "coords": lay.coords(copy=True),
            "order": sc.vertex_order().tolist(),
            "dups": int(eg.layout_degeneracy(g_bb, lay)["duplicate_nodes_f32"]),
        }

    spec_payload = {
        "bb_edges": [int(v) for edge in bb_edges for v in edge],
        "grid_edges": [int(v) for edge in grid_edges for v in edge],
        "star_edges": [int(v) for edge in star_edges for v in edge],
        "init10": [float(v) for v in init10_f32.ravel()],
        "ov": [float(v) for v in ov_f32.ravel()],
        "col": [float(v) for v in col_f32.ravel()],
        "hw10": hw10_f32,
        "hh10": hh10_f32,
        "radii10": radii10_f32,
    }

    node_body = """
if (typeof mod.MaxentStressSolver !== 'function') {
  console.log(JSON.stringify({ classic_available: false }));
  process.exit(0);
}

const gBb = mod.CsrGraph.fromEdgeList(10, new Uint32Array(spec.bb_edges), null, false, true);
const gGrid = mod.CsrGraph.fromEdgeList(16, new Uint32Array(spec.grid_edges), null, false, true);
const gStar = mod.CsrGraph.fromEdgeList(10, new Uint32Array(spec.star_edges), null, false, true);
const init10 = new Float32Array(spec.init10);
const ov = new Float32Array(spec.ov);
const col = new Float32Array(spec.col);
const hw10 = new Float32Array(spec.hw10);
const hh10 = new Float32Array(spec.hh10);
const radii10 = new Float32Array(spec.radii10);

const maxentOut = {};
for (const useBh of [true, false]) {
  const tag = useBh ? 'bh' : 'exact';
  const s1 = new mod.MaxentStressSolver(gBb, { q: 0.4, stages: 2, stepsPerStage: 5, useBarnesHut: useBh, seed: 42 });
  const s2 = new mod.MaxentStressSolver(gBb, { q: 0.4, stages: 2, stepsPerStage: 5, useBarnesHut: useBh, seed: 42 });
  const l1 = mod.Layout2d.fromCoords(init10);
  const l2 = mod.Layout2d.fromCoords(init10);
  s1.run(l1); s2.step(l2); s2.runSteps(l2, 3); s2.run(l2);
  const c1 = Array.from(l1.coordsView());
  const c2 = Array.from(l2.coordsView());
  if (!arrEq(c1, c2)) throw new Error(`WASM Maxent step()!=run() for ${tag}`);
  maxentOut[tag] = { coords: c1, q: s1.effectiveQ, alpha: s1.currentAlpha };
  l1.free?.(); l2.free?.(); s1.free?.(); s2.free?.();
}
const mAutoGrid = new mod.MaxentStressSolver(gGrid, {});
const mAutoStar = new mod.MaxentStressSolver(gStar, {});
const maxentAutoQ = { grid: mAutoGrid.effectiveQ, star: mAutoStar.effectiveQ };
mAutoGrid.free?.(); mAutoStar.free?.();

const grip1 = new mod.GripSolver(gBb, { roundsMin: 3, roundsMax: 8, seed: 42 });
const grip2 = new mod.GripSolver(gBb, { roundsMin: 3, roundsMax: 8, seed: 42 });
const lg1 = mod.Layout2d.fromCoords(init10);
const lg2 = mod.Layout2d.fromCoords(init10);
grip1.run(lg1); grip2.step(lg2); grip2.runSteps(lg2, 2); grip2.run(lg2);
const gripCoords = Array.from(lg1.coordsView());
if (!arrEq(gripCoords, Array.from(lg2.coordsView()))) throw new Error('WASM Grip step()!=run()');
const gripNumLevels = grip1.numLevels;
lg1.free?.(); lg2.free?.(); grip1.free?.(); grip2.free?.();

const gem1 = new mod.GemSolver(gBb, { maxRounds: 12, seed: 42 });
const gem2 = new mod.GemSolver(gBb, { maxRounds: 12, seed: 42 });
const lm1 = mod.Layout2d.fromCoords(init10);
const lm2 = mod.Layout2d.fromCoords(init10);
gem1.run(lm1); gem2.step(lm2); gem2.runSteps(lm2, 3); gem2.run(lm2);
const gemCoords = Array.from(lm1.coordsView());
if (!arrEq(gemCoords, Array.from(lm2.coordsView()))) throw new Error('WASM Gem step()!=run()');
lm1.free?.(); lm2.free?.(); gem1.free?.(); gem2.free?.();

const selOut = {};
for (const [mapping, layoutBackend] of [['sigmoid', 'fr'], ['tanh', 't_fdp']]) {
  const s1 = new mod.SelLayoutSolver(gBb, { mapping, layoutBackend, iterations: 8, seed: 42 });
  const s2 = new mod.SelLayoutSolver(gBb, { mapping, layoutBackend, iterations: 8, seed: 42 });
  const l1 = mod.Layout2d.fromCoords(init10);
  const l2 = mod.Layout2d.fromCoords(init10);
  s1.run(l1); s2.step(l2); s2.runSteps(l2, 2); s2.run(l2);
  const c1 = Array.from(l1.coordsView());
  if (!arrEq(c1, Array.from(l2.coordsView()))) throw new Error(`WASM SEL step()!=run() for ${mapping}/${layoutBackend}`);
  selOut[`${mapping}_${layoutBackend}`] = { coords: c1, seTarget: s1.seTarget, weights: Array.from(s1.modulatedWeights()) };
  l1.free?.(); l2.free?.(); s1.free?.(); s2.free?.();
}

const pfBox1 = new mod.PfsPrimeOverlapRemoval(hw10, hh10, { padding: 0.02, seed: 42 });
const pfBox2 = new mod.PfsPrimeOverlapRemoval(hw10, hh10, { padding: 0.02, seed: 42 });
const lPf1 = mod.Layout2d.fromCoords(ov);
const lPf2 = mod.Layout2d.fromCoords(ov);
pfBox1.run(lPf1); pfBox2.step(lPf2);
const pfBoxCoords = Array.from(lPf1.coordsView());
if (!arrEq(pfBoxCoords, Array.from(lPf2.coordsView()))) throw new Error('WASM PFS step()!=run()');
const pfBoxOv = pfBox1.lastOverlapCount;

const pfRad = mod.PfsPrimeOverlapRemoval.fromRadii(radii10, { padding: 0.0, seed: 42 });
const lColOrig = mod.Layout2d.fromCoords(col);
const lPfRad = mod.Layout2d.fromCoords(col);
pfRad.run(lPfRad);
const pfRadCoords = Array.from(lPfRad.coordsView());
const pfRadOv = pfRad.lastOverlapCount;
const norG = mod.norMetrics(gBb, lColOrig, lPfRad, radii10, radii10);
const pfOoNni = norG.ooNni;
norG.free?.(); lColOrig.free?.(); lPfRad.free?.(); pfRad.free?.();
lPf1.free?.(); lPf2.free?.(); pfBox1.free?.(); pfBox2.free?.();

const sfcOut = {};
for (const curve of ['hilbert', 'morton']) {
  const s1 = new mod.SfcLayoutSolver(gBb, { curve, scale: 80.0, seed: 42 });
  const s2 = new mod.SfcLayoutSolver(gBb, { curve, scale: 80.0, seed: 42 });
  const l1 = mod.Layout2d.zeros(10);
  const l2 = mod.Layout2d.zeros(10);
  s1.run(l1); s2.step(l2);
  const c1 = Array.from(l1.coordsView());
  if (!arrEq(c1, Array.from(l2.coordsView()))) throw new Error(`WASM SFC step()!=run() for ${curve}`);
  const deg = mod.layoutDegeneracy(gBb, l1);
  sfcOut[curve] = { coords: c1, order: Array.from(s1.vertexOrder()), dups: deg.duplicateNodesF32 };
  deg.free?.(); l1.free?.(); l2.free?.(); s1.free?.(); s2.free?.();
}

const gEmpty = mod.CsrGraph.fromEdgeList(0, new Uint32Array([]), null, false, true);
const gSingle = mod.CsrGraph.fromEdgeList(1, new Uint32Array([]), null, false, true);
const gDisc = mod.CsrGraph.fromEdgeList(6, new Uint32Array([0, 1, 1, 2, 0, 2, 3, 4]), null, false, true);
const gDirRec = mod.CsrGraph.fromEdgeList(4, new Uint32Array([0, 1, 1, 0, 1, 2, 2, 1, 2, 3]), null, true, false);
for (const gDeg of [gEmpty, gSingle, gDisc, gDirRec]) {
  const m = gDeg.numNodes();
  for (const s of [
    new mod.MaxentStressSolver(gDeg, { stages: 2, stepsPerStage: 3, seed: 42 }),
    new mod.GripSolver(gDeg, { roundsMin: 2, roundsMax: 4, seed: 42 }),
    new mod.GemSolver(gDeg, { maxRounds: 4, seed: 42 }),
    new mod.SelLayoutSolver(gDeg, { iterations: 4, seed: 42 }),
    mod.PfsPrimeOverlapRemoval.fromRadii(new Float32Array(m).fill(0.25), { seed: 42 }),
    new mod.SfcLayoutSolver(gDeg, { seed: 42 }),
  ]) {
    const ld = mod.Layout2d.zeros(m);
    s.run(ld);
    if (!ld.isAllFinite()) throw new Error('WASM baseline solver non-finite on degenerate graph');
    ld.free?.(); s.free?.();
  }
}
gEmpty.free?.(); gSingle.free?.(); gDisc.free?.(); gDirRec.free?.();

assertNonPanicError('wasm/maxent/q_bad', () => new mod.MaxentStressSolver(gBb, { q: -2 }));
assertNonPanicError('wasm/grip/rounds_inv', () => new mod.GripSolver(gBb, { roundsMin: 10, roundsMax: 5 }));
assertNonPanicError('wasm/gem/edge_len_0', () => new mod.GemSolver(gBb, { edgeLength: 0 }));
assertNonPanicError('wasm/sel/bad_mapping', () => new mod.SelLayoutSolver(gBb, { mapping: 'bad' }));
assertNonPanicError('wasm/pfs/neg_padding', () => new mod.PfsPrimeOverlapRemoval(hw10, hh10, { padding: -0.1 }));
assertNonPanicError('wasm/sfc/bad_curve', () => new mod.SfcLayoutSolver(gBb, { curve: 'peano' }));

gBb.free?.(); gGrid.free?.(); gStar.free?.();
console.log(JSON.stringify({
  classic_available: true, maxentOut, maxentAutoQ, gripCoords, gripNumLevels, gemCoords,
  selOut, pfBoxCoords, pfBoxOv, pfRadCoords, pfRadOv, pfOoNni, sfcOut,
}));
"""
    ws = run_node_json(wasm_pkg, node_body, spec_payload)
    if not ws.get("classic_available", False):
        return [{
            "case": "cross_target_classic_overlap_baselines_py_vs_wasm",
            "status": "skipped_no_wasm_classic_baselines",
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

    for tag in ("bh", "exact"):
        check_close(f"maxent/{tag}/coords", maxent_py[tag]["coords"], ws["maxentOut"][tag]["coords"], 1e-4)
        check_close(
            f"maxent/{tag}/scalars",
            [maxent_py[tag]["q"], maxent_py[tag]["alpha"]],
            [ws["maxentOut"][tag]["q"], ws["maxentOut"][tag]["alpha"]],
            1e-12, is_f32=False,
        )
    check_close("maxent/auto_q_grid", [maxent_auto_grid_py], [ws["maxentAutoQ"]["grid"]], 1e-12, is_f32=False)
    check_close("maxent/auto_q_star", [maxent_auto_star_py], [ws["maxentAutoQ"]["star"]], 1e-12, is_f32=False)

    assert grip_py["num_levels"] == ws["gripNumLevels"]
    check_close("grip/coords", grip_py["coords"], ws["gripCoords"], 1e-4)
    check_close("gem/coords", gem_py, ws["gemCoords"], 1e-4)

    for key, py_sel in sel_py.items():
        ws_sel = ws["selOut"][key]
        check_close(f"sel/{key}/coords", py_sel["coords"], ws_sel["coords"], 1e-4)
        check_close(f"sel/{key}/se_target", [py_sel["se_target"]], [ws_sel["seTarget"]], 1e-12, is_f32=False)
        check_close(f"sel/{key}/weights", py_sel["weights"], ws_sel["weights"], 1e-12, is_f32=False)

    assert pfs_box.last_overlap_count == ws["pfBoxOv"] == 0
    assert pfs_rad.last_overlap_count == ws["pfRadOv"] == 0
    check_close("pfs_prime/box", l_pfs_box.coords(copy=True), ws["pfBoxCoords"], 1e-4)
    check_close("pfs_prime/radii", l_pfs_rad.coords(copy=True), ws["pfRadCoords"], 1e-4)
    check_close("pfs_prime/oo_nni", [pfs_nor_py["oo_nni"], 0.0], [ws["pfOoNni"], 0.0], 1e-12, is_f32=False)

    for curve in ("hilbert", "morton"):
        assert sfc_py[curve]["order"] == ws["sfcOut"][curve]["order"]
        assert sfc_py[curve]["dups"] == ws["sfcOut"][curve]["dups"] == 0
        check_close(f"sfc/{curve}/coords", sfc_py[curve]["coords"], ws["sfcOut"][curve]["coords"], 1e-4)

    return [{
        "case": "cross_target_classic_overlap_baselines_py_vs_wasm",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
        "max_f32_diff": round(max_f32_diff, 12),
        "max_f64_diff": round(max_f64_diff, 12),
    }]
