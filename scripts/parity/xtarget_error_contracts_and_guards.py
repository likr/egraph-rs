"""Cross-target Python <-> WASM parity for Option 1 backend guards, error hierarchies, strict tokens, and 64-bit seeds."""

from __future__ import annotations

import time
from typing import Any

from parity.common import check_array_close, make_grid_edges, resolve_wasm_entries, run_node_json


def _assert_py_error(label: str, exc_cls: type[BaseException], fn: Any) -> BaseException:
    try:
        fn()
    except BaseException as exc:
        assert isinstance(exc, exc_cls), f"{label}: expected {exc_cls.__name__}, got {type(exc).__name__}: {exc}"
        return exc
    raise AssertionError(f"{label}: expected {exc_cls.__name__} but succeeded")


def run_cross_target_error_contracts_parity(
    eg: Any, wasm_pkg: Any
) -> list[dict[str, Any]]:
    """Verify illegal (target, backend) fail-fast, Python EgraphError/TypeError, WASM error codes, camelCase, weightDtype, and BigInt seeds."""
    import numpy as np

    if not hasattr(eg, "EgraphError") or not hasattr(eg, "Layout3d"):
        return []
    if resolve_wasm_entries(wasm_pkg) is None:
        return []

    t0 = time.perf_counter()
    g8_edges = [(i, (i + 1) % 8) for i in range(8)]
    g8 = eg.CsrGraph.from_edges(8, g8_edges, directed=False, sanitize=True)
    g16 = eg.CsrGraph.from_edges(16, make_grid_edges(4, 4), directed=False, sanitize=True)
    g2 = eg.CsrGraph.from_edges(2, [], directed=False, sanitize=False)

    # 1. Option 1 illegal (target, backend) fail-fast checks in Python -> UnsupportedOperationError
    illegal_pairs: list[tuple[str, Any]] = [
        ("tsnet/2d/auto3d", lambda: eg.TsNetSolver.pure_2hop(g8, target="layout2d", backend="auto3d")),
        ("tsnet/3d/auto2d", lambda: eg.TsNetSolver.pure_2hop(g8, target="layout3d", backend="auto2d")),
        ("tsnet/3d/fit_fft", lambda: eg.TsNetSolver.pure_2hop(g8, target="layout3d", backend="fit_fft")),
        ("tsnet/3d/linear_fft", lambda: eg.TsNetSolver.pure_2hop(g8, target="layout3d", backend="linear_fft")),
        ("tfdp/3d/ib_fft", lambda: eg.TFdpSolver(g8, target="layout3d", backend="ib_fft")),
        ("tsnet/nd/auto2d", lambda: eg.TsNetSolver.pure_2hop(g8, target="layout_nd", backend="auto2d")),
        ("tsnet/nd/auto3d", lambda: eg.TsNetSolver.pure_2hop(g8, target="layout_nd", backend="auto3d")),
        ("tsnet/nd/bh", lambda: eg.TsNetSolver.pure_2hop(g8, target="layout_nd", backend="barnes_hut")),
        ("tsnet/nd/fit_fft", lambda: eg.TsNetSolver.pure_2hop(g8, target="layout_nd", backend="fit_fft")),
        ("tsnet/nd/linear_fft", lambda: eg.TsNetSolver.pure_2hop(g8, target="layout_nd", backend="linear_fft")),
        ("tsnet/mani/auto2d", lambda: eg.TsNetSolver.pure_2hop(g8, target="manifold2d", backend="auto2d")),
        ("tsnet/mani/auto3d", lambda: eg.TsNetSolver.pure_2hop(g8, target="manifold2d", backend="auto3d")),
        ("fr/nd/bh", lambda: eg.FruchtermanReingoldSolver(g8, target="layout_nd", backend="barnes_hut")),
        ("fr/nd/grid", lambda: eg.FruchtermanReingoldSolver(g8, target="layout_nd", backend="grid")),
        ("fa2/nd/bh", lambda: eg.ForceAtlas2Solver(g8, target="layout_nd", backend="barnes_hut")),
        ("linlog/nd/bh", lambda: eg.LinLogSolver(g8, target="layout_nd", backend="barnes_hut")),
        ("yh/nd/bh", lambda: eg.YifanHuSolver(g8, target="layout_nd", backend="barnes_hut")),
        ("maxent/nd/bh", lambda: eg.MaxentStressSolver(g8, target="layout_nd", backend="barnes_hut")),
        ("tfdp/nd/bh", lambda: eg.TFdpSolver(g8, target="layout_nd", backend="barnes_hut")),
        ("fr/mani/bh", lambda: eg.FruchtermanReingoldSolver(g8, target="manifold2d", backend="barnes_hut")),
        ("sm/mani/euc", lambda: eg.StressMajorizationSolver.from_graph(g8, target="manifold2d", backend="euclidean_pcg")),
        ("ibumap/3d", lambda: eg.IbUmapSolver(g8, target="layout3d")),
        ("ibumap/nd", lambda: eg.IbUmapSolver(g8, target="layout_nd")),
        ("ibumap/mani", lambda: eg.IbUmapSolver(g8, target="manifold2d")),
    ]
    for lbl, fn in illegal_pairs:
        _assert_py_error(f"py/illegal/{lbl}", eg.UnsupportedOperationError, fn)

    # 2. Python EgraphError hierarchy & all 7 subclasses
    subclasses = (
        eg.InvalidInputError,
        eg.DimensionMismatchError,
        eg.IndexOutOfBoundsError,
        eg.InvalidCsrError,
        eg.NonFiniteValueError,
        eg.UnsupportedOperationError,
        eg.NumericalBreakdownError,
    )
    for sub in subclasses:
        assert issubclass(sub, eg.EgraphError)
        assert issubclass(sub, ValueError)
    assert issubclass(eg.IndexOutOfBoundsError, IndexError)

    _assert_py_error("py/err/invalid_input", eg.InvalidInputError, lambda: eg.PivotMds(g8, num_pivots=0).generate())
    _assert_py_error("py/err/invalid_input_scale", eg.InvalidInputError, lambda: eg.ManifoldLayout2d.init("hyperbolic_poincare", 4).set_scale(0.0))
    _assert_py_error("py/err/dim_mismatch_nodes", eg.DimensionMismatchError, lambda: eg.SgdSolver.full(g8).run(eg.Layout2d.zeros(4)))
    _assert_py_error("py/err/dim_mismatch_target", eg.DimensionMismatchError, lambda: eg.SgdSolver.full(g8, target="layout2d").run(eg.Layout3d.zeros(8)))
    _assert_py_error("py/err/dim_mismatch_cols", eg.DimensionMismatchError, lambda: eg.Layout3d.from_coords(np.zeros((4, 2), dtype=np.float32)))

    oob_dijk = _assert_py_error("py/err/oob_dijkstra", eg.IndexOutOfBoundsError, lambda: eg.dijkstra_single_source(g8, 99))
    assert isinstance(oob_dijk, eg.EgraphError) and isinstance(oob_dijk, IndexError)
    oob_l3 = _assert_py_error("py/err/oob_l3", eg.IndexOutOfBoundsError, lambda: eg.Layout3d.zeros(4).x(99))
    assert isinstance(oob_l3, eg.EgraphError) and isinstance(oob_l3, IndexError)

    _assert_py_error("py/err/invalid_csr", eg.InvalidCsrError, lambda: eg.CsrGraph.from_csr([0, 5], [0]))
    _assert_py_error("py/err/non_finite_l3", eg.NonFiniteValueError, lambda: eg.Layout3d.from_coords(np.array([[float("nan"), 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)))
    _assert_py_error("py/err/unsupported", eg.UnsupportedOperationError, lambda: eg.IbUmapSolver(g8, target="layout3d"))

    fr_fail = eg.FruchtermanReingoldSolver(g2, iterations=5, initial_temperature=1e38, k=1e58, dtype="float32")
    l_fail = eg.Layout2d.from_coords(np.array([[-3.4e38, 0.0], [-2.4e38, 0.0]], dtype=np.float32))
    _assert_py_error("py/err/breakdown_fr", eg.NumericalBreakdownError, lambda: fr_fail.step(l_fail))
    assert fr_fail.status == "failed" and fr_fail.is_finished

    pts_fail = np.array([[3.0e38, -3.0e38], [3.0e38, 3.0e38], [3.1e38, -3.0e38], [3.1e38, 3.0e38]], dtype=np.float32)
    poly_fail = eg.EdgePolylines.from_parts([0, 2, 4], eg.Layout2d.from_coords(pts_fail))
    fd_fail = eg.FdebSolver(poly_fail, cycles=2, s0=1e38, steps_per_cycle=4, minimum_edge_compatibility=0.0)
    _assert_py_error("py/err/breakdown_fdeb", eg.NumericalBreakdownError, lambda: fd_fail.step(poly_fail))
    assert fd_fail.status == "failed" and fd_fail.is_finished

    # 3. Keyword-only '*' TypeError enforcement in Python
    kw_only_calls: list[tuple[str, Any]] = [
        ("PivotMds", lambda: eg.PivotMds(g8, "layout2d")),
        ("ClassicalMds", lambda: eg.ClassicalMds(g8, "layout2d")),
        ("Hde", lambda: eg.Hde(g8, "layout2d")),
        ("RdMds", lambda: eg.RdMds(g8, "layout2d")),
        ("KorenSpectral", lambda: eg.KorenSpectral(g8, "layout2d")),
        ("CoarseBackbone", lambda: eg.CoarseBackbone(g8, 4)),
        ("SfcLayout", lambda: eg.SfcLayout(g8, "hilbert")),
        ("FruchtermanReingoldSolver", lambda: eg.FruchtermanReingoldSolver(g8, 5)),
        ("FdebSolver", lambda: eg.FdebSolver(poly_fail, 1)),
        ("from_graph_partial_bfs", lambda: eg.TsNetSolver.from_graph_partial_bfs(g8, 15, 15.0)),
        ("generate_hyperbolic_tree", lambda: eg.generate_hyperbolic_tree(2, 3)),
    ]
    for lbl, fn in kw_only_calls:
        _assert_py_error(f"py/kw_only/{lbl}", TypeError, fn)

    # 4. 64-bit seed > 2^53 - 1 parity baseline in Python
    seed64_a = 9007199254740993  # 2^53 + 1
    seed64_b = 12345678901234567891  # > 2^63
    l2_seed_py = eg.Layout2d.init(16, seed=seed64_a, scale=0.8, dtype="float32").coords(copy=True).ravel()
    l3_seed_py = eg.Layout3d.init(16, seed=seed64_b, scale=0.8, dtype="float32").coords(copy=True).ravel()

    l2_sgd = eg.Layout2d.init(16, seed=17, dtype="float32")
    eg.SgdSolver.full(g16, iterations=5, seed=seed64_a, dtype="float32").run(l2_sgd)
    sgd_seed_py = l2_sgd.coords(copy=True).ravel()

    l2_ts = eg.Layout2d.init(16, seed=17, dtype="float32")
    eg.TsNetSolver.pure_2hop(g16, backend="exact", stage_iterations=(2, 2, 2), seed=seed64_b, dtype="float32").run(l2_ts)
    ts_seed_py = l2_ts.coords(copy=True).ravel()

    l2_ts_auto2d = eg.Layout2d.init(16, seed=17, dtype="float32")
    eg.TsNetSolver.pure_2hop(g16, target="layout2d", backend="auto2d", stage_iterations=(2, 2, 2), seed=42, dtype="float32").run(l2_ts_auto2d)
    ts_auto2d_py = l2_ts_auto2d.coords(copy=True).ravel()

    l3_ts_auto3d = eg.Layout3d.init(16, seed=17, dtype="float32")
    eg.TsNetSolver.pure_2hop(g16, target="layout3d", backend="auto3d", stage_iterations=(2, 2, 2), seed=42, dtype="float32").run(l3_ts_auto3d)
    ts_auto3d_py = l3_ts_auto3d.coords(copy=True).ravel()

    # 5. SelLayoutSolver nested t_fdp_config / fr_config fail-fast & backend differentiation in Python
    _assert_py_error("py/sel/tfdp/min_grid_dim", eg.InvalidInputError, lambda: eg.SelLayoutSolver(g8, layout_backend="t_fdp", t_fdp_config={"min_grid_dim": 50}))
    _assert_py_error("py/sel/tfdp/theta", eg.InvalidInputError, lambda: eg.SelLayoutSolver(g8, layout_backend="t_fdp", t_fdp_config={"theta": -1.0}))
    _assert_py_error("py/sel/tfdp/backend_bogus", eg.UnsupportedOperationError, lambda: eg.SelLayoutSolver(g8, layout_backend="t_fdp", t_fdp_config={"backend": "bogus"}))
    _assert_py_error("py/sel/fr/theta", eg.InvalidInputError, lambda: eg.SelLayoutSolver(g8, layout_backend="fr", fr_config={"theta": -1.0}))
    _assert_py_error("py/sel/fr/backend_bogus", eg.UnsupportedOperationError, lambda: eg.SelLayoutSolver(g8, layout_backend="fr", fr_config={"backend": "bogus"}))
    l2_sel_ex = eg.Layout2d.init(16, seed=17, dtype="float32")
    eg.SelLayoutSolver(g16, layout_backend="t_fdp", t_fdp_config={"backend": "exact", "iterations": 20, "seed": 42}, dtype="float32").run(l2_sel_ex)
    l2_sel_fft = eg.Layout2d.init(16, seed=17, dtype="float32")
    eg.SelLayoutSolver(g16, layout_backend="t_fdp", t_fdp_config={"backend": "ib_fft", "iterations": 20, "seed": 42}, dtype="float32").run(l2_sel_fft)
    l2_sel_bh = eg.Layout2d.init(16, seed=17, dtype="float32")
    eg.SelLayoutSolver(g16, layout_backend="t_fdp", t_fdp_config={"backend": "barnes_hut", "iterations": 20, "seed": 42}, dtype="float32").run(l2_sel_bh)
    sel_ex_py = l2_sel_ex.coords(copy=True).ravel()
    sel_fft_py = l2_sel_fft.coords(copy=True).ravel()
    sel_bh_py = l2_sel_bh.coords(copy=True).ravel()
    assert not np.allclose(sel_ex_py, sel_fft_py), "py/sel: exact vs ib_fft should diverge"

    spec_payload = {"g8_edges": [int(v) for e in g8_edges for v in e], "g16_edges": [int(v) for e in make_grid_edges(4, 4) for v in e]}
    node_body = """
const g8 = mod.CsrGraph.fromEdgeList(8, new Uint32Array(spec.g8_edges));
const g16 = mod.CsrGraph.fromEdgeList(16, new Uint32Array(spec.g16_edges));
const g2 = mod.CsrGraph.fromEdgeList(2, new Uint32Array([]));

const expectCode = (label, expectedCode, fn) => {
  try {
    fn();
  } catch (err) {
    assert.ok(!(err instanceof WebAssembly.RuntimeError), `${label} triggered RuntimeError: ${err}`);
    assert.ok(err instanceof Error, `${label} must throw Error`);
    assert.equal(err.code, expectedCode, `${label}: expected code=${expectedCode}, got ${err.code} (${err.message})`);
    return err;
  }
  assert.fail(`${label}: expected error ${expectedCode} but succeeded`);
};

// Option 1 illegal (target, backend) -> UnsupportedOperation
const illegalWasm = [
  ['tsnet/2d/auto3d', () => mod.TsNetSolver.pure2Hop(g8, { target: 'layout2d', backend: 'auto3d' })],
  ['tsnet/3d/auto2d', () => mod.TsNetSolver.pure2Hop(g8, { target: 'layout3d', backend: 'auto2d' })],
  ['tsnet/3d/fitFft', () => mod.TsNetSolver.pure2Hop(g8, { target: 'layout3d', backend: 'fitFft' })],
  ['tsnet/3d/linearFft', () => mod.TsNetSolver.pure2Hop(g8, { target: 'layout3d', backend: 'linearFft' })],
  ['tfdp/3d/ibFft', () => new mod.TFdpSolver(g8, { target: 'layout3d', backend: 'ibFft' })],
  ['tsnet/nd/auto2d', () => mod.TsNetSolver.pure2Hop(g8, { target: 'layoutNd', backend: 'auto2d' })],
  ['tsnet/nd/auto3d', () => mod.TsNetSolver.pure2Hop(g8, { target: 'layoutNd', backend: 'auto3d' })],
  ['tsnet/nd/bh', () => mod.TsNetSolver.pure2Hop(g8, { target: 'layoutNd', backend: 'barnesHut' })],
  ['tsnet/nd/fitFft', () => mod.TsNetSolver.pure2Hop(g8, { target: 'layoutNd', backend: 'fitFft' })],
  ['tsnet/nd/linearFft', () => mod.TsNetSolver.pure2Hop(g8, { target: 'layoutNd', backend: 'linearFft' })],
  ['tsnet/mani/auto2d', () => mod.TsNetSolver.pure2Hop(g8, { target: 'manifold2d', backend: 'auto2d' })],
  ['tsnet/mani/auto3d', () => mod.TsNetSolver.pure2Hop(g8, { target: 'manifold2d', backend: 'auto3d' })],
  ['fr/nd/bh', () => new mod.FruchtermanReingoldSolver(g8, { target: 'layoutNd', backend: 'barnesHut' })],
  ['fr/nd/grid', () => new mod.FruchtermanReingoldSolver(g8, { target: 'layoutNd', backend: 'grid' })],
  ['fa2/nd/bh', () => new mod.ForceAtlas2Solver(g8, { target: 'layoutNd', backend: 'barnesHut' })],
  ['linlog/nd/bh', () => new mod.LinLogSolver(g8, { target: 'layoutNd', backend: 'barnesHut' })],
  ['yh/nd/bh', () => new mod.YifanHuSolver(g8, { target: 'layoutNd', backend: 'barnesHut' })],
  ['maxent/nd/bh', () => new mod.MaxentStressSolver(g8, { target: 'layoutNd', backend: 'barnesHut' })],
  ['tfdp/nd/bh', () => new mod.TFdpSolver(g8, { target: 'layoutNd', backend: 'barnesHut' })],
  ['fr/mani/bh', () => new mod.FruchtermanReingoldSolver(g8, { target: 'manifold2d', backend: 'barnesHut' })],
  ['sm/mani/euc', () => mod.StressMajorizationSolver.fromGraph(g8, { target: 'manifold2d', backend: 'euclideanPcg' })],
  ['ibumap/3d', () => new mod.IbUmapSolver(g8, { target: 'layout3d' })],
  ['ibumap/nd', () => new mod.IbUmapSolver(g8, { target: 'layoutNd' })],
  ['ibumap/mani', () => new mod.IbUmapSolver(g8, { target: 'manifold2d' })],
];
for (const [lbl, fn] of illegalWasm) {
  expectCode(`wasm/illegal/${lbl}`, 'UnsupportedOperation', fn);
}

// All 7 EgraphErrorCode variants
expectCode('wasm/code/InvalidInput', 'InvalidInput', () => new mod.PivotMds({ numPivots: 0 }).generate(g8));
expectCode('wasm/code/DimensionMismatch', 'DimensionMismatch', () => mod.Layout3d.fromCoords(new Float32Array([1, 2, 3, 4])));
expectCode('wasm/code/IndexOutOfBounds', 'IndexOutOfBounds', () => mod.dijkstraSingleSource(g8, 99));
expectCode('wasm/code/InvalidCsr', 'InvalidCsr', () => mod.CsrGraph.fromCsr(new Uint32Array([0, 5]), new Uint32Array([0])));
expectCode('wasm/code/NonFiniteValue', 'NonFiniteValue', () => mod.Layout3d.fromCoords(new Float32Array([Number.NaN, 0, 0, 1, 0, 0])));
expectCode('wasm/code/UnsupportedOperation', 'UnsupportedOperation', () => new mod.IbUmapSolver(g8, { target: 'layout3d' }));

const frFail = new mod.FruchtermanReingoldSolver(g2, { iterations: 5, initialTemperature: 1e38, k: 1e58 });
const lFail = mod.Layout2d.fromCoords(new Float32Array([-3.4e38, 0, -2.4e38, 0]));
expectCode('wasm/code/NumericalBreakdown_fr', 'NumericalBreakdown', () => frFail.step(lFail));
assert.equal(frFail.status, 'failed');
assert.equal(frFail.isFinished, true);
frFail.free?.(); lFail.free?.();

const ptsFail = mod.Layout2d.fromCoords(new Float32Array([3.0e38, -3.0e38, 3.0e38, 3.0e38, 3.1e38, -3.0e38, 3.1e38, 3.0e38]));
const polyFail = mod.EdgePolylines.fromParts(new Uint32Array([0, 2, 4]), ptsFail);
const fdFail = new mod.FdebSolver(polyFail, { cycles: 2, s0: 1e38, stepsPerCycle: 4, minimumEdgeCompatibility: 0.0 });
expectCode('wasm/code/NumericalBreakdown_fdeb', 'NumericalBreakdown', () => fdFail.step(polyFail));
assert.equal(fdFail.status, 'failed');
assert.equal(fdFail.isFinished, true);
fdFail.free?.(); polyFail.free?.(); ptsFail.free?.();

// Strict snake_case rejection -> InvalidInput
expectCode('wasm/snake/target', 'InvalidInput', () => mod.TsNetSolver.pure2Hop(g8, { target: 'layout_nd' }));
expectCode('wasm/snake/backend_fft', 'InvalidInput', () => mod.TsNetSolver.pure2Hop(g8, { backend: 'fit_fft' }));
expectCode('wasm/snake/backend_bh', 'InvalidInput', () => new mod.FruchtermanReingoldSolver(g8, { backend: 'barnes_hut' }));
expectCode('wasm/snake/manifold', 'InvalidInput', () => mod.ManifoldLayout2d.init('hyperbolic_poincare', 4));
expectCode('wasm/snake/sm_backend', 'InvalidInput', () => mod.StressMajorizationSolver.fromGraph(g8, { backend: 'euclidean_pcg' }));

// Strict weightDtype: "float32" rejection of "float64"
expectCode('wasm/weightDtype/fromEdgeList_f64', 'InvalidInput', () => mod.CsrGraph.fromEdgeList(4, new Uint32Array([0, 1, 1, 2]), new Float32Array([1, 1]), { weightDtype: 'float64' }));
expectCode('wasm/weightDtype/fromCsr_f64', 'InvalidInput', () => mod.CsrGraph.fromCsr(new Uint32Array([0, 1, 2]), new Uint32Array([1, 0]), new Float32Array([1, 1]), { weightDtype: 'float64' }));
const gOkF32 = mod.CsrGraph.fromEdgeList(4, new Uint32Array([0, 1, 1, 2]), new Float32Array([1, 1]), { weightDtype: 'float32' });
assert.equal(gOkF32.weightDtype, 'float32');
gOkF32.free?.();

// BigInt 64-bit seeds (> 2^53 - 1) matching Python u64 seeds
const sA = 9007199254740993n;
const sB = 12345678901234567891n;
const l2Seed = mod.Layout2d.init(16, { seed: sA, scale: 0.8 });
const l3Seed = mod.Layout3d.init(16, { seed: sB, scale: 0.8 });
const l2Sgd = mod.Layout2d.init(16, { seed: 17n });
const svSgd = mod.SgdSolver.full(g16, { iterations: 5, seed: sA });
svSgd.run(l2Sgd);
const l2Ts = mod.Layout2d.init(16, { seed: 17n });
const svTs = mod.TsNetSolver.pure2Hop(g16, { backend: 'exact', stageIterations: [2, 2, 2], seed: sB });
svTs.run(l2Ts);

const l2Auto2d = mod.Layout2d.init(16, { seed: 17n });
const svAuto2d = mod.TsNetSolver.pure2Hop(g16, { target: 'layout2d', backend: 'auto2d', stageIterations: [2, 2, 2], seed: 42 });
svAuto2d.run(l2Auto2d);
const l3Auto3d = mod.Layout3d.init(16, { seed: 17n });
const svAuto3d = mod.TsNetSolver.pure2Hop(g16, { target: 'layout3d', backend: 'auto3d', stageIterations: [2, 2, 2], seed: 42 });
svAuto3d.run(l3Auto3d);

// SelLayoutSolver nested tFdpConfig / frConfig fail-fast & backend differentiation
expectCode('wasm/sel/tfdp/minGridDim', 'InvalidInput', () => new mod.SelLayoutSolver(g8, { layoutBackend: 'tFdp', tFdpConfig: { minGridDim: 50 } }));
expectCode('wasm/sel/tfdp/theta', 'InvalidInput', () => new mod.SelLayoutSolver(g8, { layoutBackend: 'tFdp', tFdpConfig: { theta: -1.0 } }));
expectCode('wasm/sel/tfdp/backend_bogus', 'InvalidInput', () => new mod.SelLayoutSolver(g8, { layoutBackend: 'tFdp', tFdpConfig: { backend: 'bogus' } }));
expectCode('wasm/sel/fr/theta', 'InvalidInput', () => new mod.SelLayoutSolver(g8, { layoutBackend: 'fr', frConfig: { theta: -1.0 } }));
expectCode('wasm/sel/fr/backend_bogus', 'InvalidInput', () => new mod.SelLayoutSolver(g8, { layoutBackend: 'fr', frConfig: { backend: 'bogus' } }));

const l2SelEx = mod.Layout2d.init(16, { seed: 17n });
const svSelEx = new mod.SelLayoutSolver(g16, { layoutBackend: 'tFdp', tFdpConfig: { backend: 'exact', iterations: 20, seed: 42 } });
svSelEx.run(l2SelEx);
const l2SelFft = mod.Layout2d.init(16, { seed: 17n });
const svSelFft = new mod.SelLayoutSolver(g16, { layoutBackend: 'tFdp', tFdpConfig: { backend: 'ibFft', iterations: 20, seed: 42 } });
svSelFft.run(l2SelFft);
const l2SelBh = mod.Layout2d.init(16, { seed: 17n });
const svSelBh = new mod.SelLayoutSolver(g16, { layoutBackend: 'tFdp', tFdpConfig: { backend: 'barnesHut', iterations: 20, seed: 42 } });
svSelBh.run(l2SelBh);
const selExArr = Array.from(l2SelEx.coordsView());
const selFftArr = Array.from(l2SelFft.coordsView());
const selBhArr = Array.from(l2SelBh.coordsView());
assert.ok(selExArr.some((v, i) => Math.abs(v - selFftArr[i]) > 1e-6), 'wasm/sel: exact vs ibFft should diverge');

const out = {
  l2Seed: Array.from(l2Seed.coordsView()),
  l3Seed: Array.from(l3Seed.coordsView()),
  sgdSeed: Array.from(l2Sgd.coordsView()),
  tsSeed: Array.from(l2Ts.coordsView()),
  tsAuto2d: Array.from(l2Auto2d.coordsView()),
  tsAuto3d: Array.from(l3Auto3d.coordsView()),
  selEx: selExArr,
  selFft: selFftArr,
  selBh: selBhArr,
};
svSelBh.free?.(); l2SelBh.free?.();
svSelFft.free?.(); l2SelFft.free?.(); svSelEx.free?.(); l2SelEx.free?.();
svAuto3d.free?.(); l3Auto3d.free?.(); svAuto2d.free?.(); l2Auto2d.free?.();
svTs.free?.(); l2Ts.free?.(); svSgd.free?.(); l2Sgd.free?.(); l3Seed.free?.(); l2Seed.free?.();
g2.free?.(); g16.free?.(); g8.free?.();
console.log(JSON.stringify(out));
"""
    ws = run_node_json(wasm_pkg, node_body, spec_payload)

    max_f32_diff = 0.0
    max_f32_diff = max(max_f32_diff, check_array_close("bigint/l2_init", l2_seed_py, ws["l2Seed"], 1e-6))
    max_f32_diff = max(max_f32_diff, check_array_close("bigint/l3_init", l3_seed_py, ws["l3Seed"], 1e-6))
    max_f32_diff = max(max_f32_diff, check_array_close("bigint/sgd", sgd_seed_py, ws["sgdSeed"], 1e-5))
    max_f32_diff = max(max_f32_diff, check_array_close("bigint/tsnet", ts_seed_py, ws["tsSeed"], 1e-5))
    max_f32_diff = max(max_f32_diff, check_array_close("tsnet/auto2d", ts_auto2d_py, ws["tsAuto2d"], 1e-5))
    max_f32_diff = max(max_f32_diff, check_array_close("tsnet/auto3d", ts_auto3d_py, ws["tsAuto3d"], 1e-5))
    max_f32_diff = max(max_f32_diff, check_array_close("sel/tfdp/exact", sel_ex_py, ws["selEx"], 1e-5))
    max_f32_diff = max(max_f32_diff, check_array_close("sel/tfdp/ib_fft", sel_fft_py, ws["selFft"], 1e-5))
    max_f32_diff = max(max_f32_diff, check_array_close("sel/tfdp/barnes_hut", sel_bh_py, ws["selBh"], 1e-5))

    return [{
        "case": "cross_target_error_contracts_and_guards_py_vs_wasm",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
        "max_f32_diff": round(max_f32_diff, 12),
    }]
