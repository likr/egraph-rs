import assert from "node:assert/strict";

export function runCoreBuffersSmoke({ eg, webPkg }) {
  const { CsrGraph, Layout2d, wasmMemory } = eg;

  assert.equal(typeof webPkg.Layout2d.prototype.coordsView, "function");
  assert.equal(typeof Layout2d.prototype.coordsView, "function");

  // 1. CsrGraph.fromCsr: strict valid undirected weighted graph
  {
    const indptr = new Uint32Array([0, 1, 3, 4]);
    const indices = new Uint32Array([1, 0, 2, 1]);
    const weights = new Float32Array([1.5, 1.5, 2.5, 2.5]);

    const g = CsrGraph.fromCsr(indptr, indices, weights, {
      directed: false,
      sanitize: false,
    });
    assert.equal(g.numNodes(), 3);
    assert.equal(g.numEdges(), 2);
    assert.equal(g.isDirected(), false);
    assert.deepEqual(Array.from(g.indptr()), [0, 1, 3, 4]);
    assert.deepEqual(Array.from(g.indices()), [1, 0, 2, 1]);
    assert.deepEqual(Array.from(g.weights()), [1.5, 1.5, 2.5, 2.5]);
    g.free();
  }

  // 2. CsrGraph.fromCsr: default sanitization + boolean trailing params
  {
    const dirtyIndptr = new Uint32Array([0, 4, 5, 5]);
    const dirtyIndices = new Uint32Array([0, 2, 1, 2, 0]);
    const dirtyWeights = new Float32Array([9.0, 2.0, 1.5, 4.0, 1.5]);

    assert.throws(
      () => CsrGraph.fromCsr(dirtyIndptr, dirtyIndices, dirtyWeights, false, false),
      Error,
    );

    const sanitized = CsrGraph.fromCsr(dirtyIndptr, dirtyIndices, dirtyWeights);
    assert.equal(sanitized.numNodes(), 3);
    assert.equal(sanitized.numEdges(), 2);
    assert.deepEqual(Array.from(sanitized.indptr()), [0, 2, 3, 4]);
    assert.deepEqual(Array.from(sanitized.indices()), [1, 2, 0, 0]);
    assert.deepEqual(Array.from(sanitized.weights()), [1.5, 4.0, 1.5, 4.0]);
    sanitized.free();
  }

  // 3. CsrGraph.fromEdgeList: directed and undirected
  {
    const dg = CsrGraph.fromEdgeList(
      3,
      new Uint32Array([0, 1, 0, 2]),
      undefined,
      { directed: true, sanitize: false },
    );
    assert.equal(dg.isDirected(), true);
    assert.equal(dg.numNodes(), 3);
    assert.equal(dg.numEdges(), 2);
    assert.deepEqual(Array.from(dg.indptr()), [0, 2, 2, 2]);
    assert.deepEqual(Array.from(dg.indices()), [1, 2]);
    assert.equal(dg.weights(), undefined);
    dg.free();

    const ug = CsrGraph.fromEdgeList(
      3,
      new Uint32Array([0, 1, 1, 2]),
      new Float32Array([0.5, 1.25]),
    );
    assert.equal(ug.isDirected(), false);
    assert.equal(ug.numEdges(), 2);
    assert.deepEqual(Array.from(ug.indptr()), [0, 1, 3, 4]);
    assert.deepEqual(Array.from(ug.indices()), [1, 0, 2, 1]);
    assert.deepEqual(Array.from(ug.weights()), [0.5, 0.5, 1.25, 1.25]);
    ug.free();
  }

  // 4. Layout2d zero-copy coordsView() + memory.grow resilience
  {
    const empty = Layout2d.zeros(0);
    assert.equal(empty.len(), 0);
    assert.equal(empty.isEmpty(), true);
    assert.equal(empty.coordsLen(), 0);
    assert.equal(empty.coordsView().length, 0);
    empty.free();

    const layout = Layout2d.zeros(4);
    assert.equal(layout.len(), 4);
    assert.equal(layout.isEmpty(), false);
    assert.equal(layout.coordsLen(), 8);
    assert.equal(layout.isAllFinite(), true);

    const view1 = layout.coordsView();
    const view2 = layout.coordsView();
    assert(view1 instanceof Float32Array);
    assert.equal(view1.length, 8);
    assert.equal(view1, view2, "coordsView() must cache Float32Array while buffer is unchanged");

    // Rust -> JS zero-copy visibility
    layout.setX(0, 1.5);
    layout.setY(0, 2.5);
    assert.equal(view1[0], 1.5);
    assert.equal(view1[1], 2.5);

    // JS -> Rust zero-copy visibility
    view1[2] = 7.5;
    view1[3] = -3.25;
    assert.equal(layout.x(1), 7.5);
    assert.equal(layout.y(1), -3.25);

    // Force WebAssembly linear memory growth to detach the old ArrayBuffer
    const oldBuf = view1.buffer;
    wasmMemory().grow(1);
    assert.equal(oldBuf.byteLength, 0, "expected old ArrayBuffer to be detached after memory.grow");

    const viewAfterGrow = layout.coordsView();
    assert.notEqual(viewAfterGrow, view1);
    assert.notEqual(viewAfterGrow.buffer, oldBuf);
    assert(viewAfterGrow.buffer.byteLength > 0);
    assert.equal(viewAfterGrow[0], 1.5);
    assert.equal(viewAfterGrow[1], 2.5);
    assert.equal(viewAfterGrow[2], 7.5);
    assert.equal(viewAfterGrow[3], -3.25);

    viewAfterGrow[4] = 9.0;
    viewAfterGrow[5] = -8.5;
    assert.equal(layout.x(2), 9.0);
    assert.equal(layout.y(2), -8.5);

    // Finiteness checks
    layout.setX(3, Number.NaN);
    assert.equal(layout.isAllFinite(), false);
    layout.setX(3, 0.0);
    assert.equal(layout.isAllFinite(), true);
    layout.setY(3, Number.POSITIVE_INFINITY);
    assert.equal(layout.isAllFinite(), false);

    layout.free();
  }

  // 5. Layout2d.fromCoords round-trip
  {
    const layout = Layout2d.fromCoords(new Float32Array([1.0, -2.0, 3.5, 4.5]));
    assert.equal(layout.len(), 2);
    assert.equal(layout.x(0), 1.0);
    assert.equal(layout.y(0), -2.0);
    assert.equal(layout.x(1), 3.5);
    assert.equal(layout.y(1), 4.5);
    assert.deepEqual(Array.from(layout.coordsView()), [1.0, -2.0, 3.5, 4.5]);
    layout.free();
  }

  // 6. Adversarial inputs must throw Error (never crash/unreachable trap)
  assert.throws(
    () => CsrGraph.fromCsr(new Uint32Array([]), new Uint32Array([])),
    Error,
  );
  assert.throws(
    () => CsrGraph.fromCsr(new Uint32Array([0, 1, 1]), new Uint32Array([5])),
    Error,
  );
  assert.throws(
    () =>
      CsrGraph.fromCsr(
        new Uint32Array([0, 1, 2]),
        new Uint32Array([1, 0]),
        new Float32Array([Number.NaN, 1.0]),
      ),
    Error,
  );
  assert.throws(
    () =>
      CsrGraph.fromCsr(
        new Uint32Array([0, 1, 2, 3]),
        new Uint32Array([1, 2, 0]),
        undefined,
        { directed: false, sanitize: false },
      ),
    Error,
  );
  assert.throws(
    () => CsrGraph.fromCsr(new Uint32Array([0, 0]), new Uint32Array([]), undefined, { directed: "yes" }),
    Error,
  );
  assert.throws(
    () => CsrGraph.fromEdgeList(3, new Uint32Array([0, 1, 2])),
    Error,
  );
  assert.throws(
    () => CsrGraph.fromEdgeList(-1, new Uint32Array([])),
    Error,
  );
  assert.throws(
    () => CsrGraph.fromEdgeList(1.5, new Uint32Array([])),
    Error,
  );
  assert.throws(
    () => Layout2d.fromCoords(new Float32Array([1.0, 2.0, 3.0])),
    Error,
  );
  assert.throws(() => Layout2d.zeros(-1), Error);
  assert.throws(() => Layout2d.zeros(1.5), Error);
  assert.throws(() => Layout2d.zeros(Number.NaN), Error);
  {
    const l = Layout2d.zeros(2);
    assert.throws(() => l.x(2), Error);
    assert.throws(() => l.y(-1), Error);
    assert.throws(() => l.setX(1.5, 0), Error);
    assert.throws(() => l.setY(99, 0), Error);
    l.free();
  }
}

export function runLegacyAbsenceSmoke({ eg }) {
  // Verify legacy exports are absent by default and InfoMap is omitted
  assert.equal(typeof eg.InfoMap, "undefined");
  if (typeof eg.Graph === "function") {
    const legacyGraph = new eg.Graph();
    const u = legacyGraph.addNode(0);
    const v = legacyGraph.addNode(1);
    legacyGraph.addEdge(u, v, 1);
    assert.equal(legacyGraph.nodeCount(), 2);
    assert.equal(legacyGraph.edgeCount(), 1);
    legacyGraph.free();
  }
}
