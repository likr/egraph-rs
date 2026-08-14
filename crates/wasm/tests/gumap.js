const assert = require("assert");
const eg = require("wasm-bindgen-test");
const helpers = require("./util/test_helpers");

/**
 * Test basic instantiation of GuMap class and GuMapBuilder
 */
exports.testGuMapConstructor = function () {
  const builder = new eg.GuMapBuilder();
  assert(builder instanceof eg.GuMapBuilder, "Should create an instance of GuMapBuilder");
  const gumapBuilt = builder
    .k(10)
    .minDist(0.05)
    .spread(1.2)
    .nEpochs(200)
    .negativeSampleRate(3)
    .learningRate(0.8)
    .build();
  assert(gumapBuilt instanceof eg.GuMap, "Should build GuMap instance");

  const gumap = new eg.GuMap();
  assert(gumap instanceof eg.GuMap, "Should create an instance of GuMap");
  assert.strictEqual(gumap.k, 15, "default k should be 15");
  assert.ok(Math.abs(gumap.minDist - 0.1) < 1e-5, "default minDist should be 0.1");
  assert.ok(Math.abs(gumap.spread - 1.0) < 1e-5, "default spread should be 1.0");
  assert.strictEqual(gumap.nEpochs, 300, "default nEpochs should be 300");
  assert.strictEqual(gumap.negativeSampleRate, 5, "default negativeSampleRate should be 5");
  assert.ok(Math.abs(gumap.learningRate - 1.0) < 1e-5, "default learningRate should be 1.0");
};

/**
 * Test running GuMap on graph
 */
exports.testGuMapRun = function () {
  const { graph } = helpers.createCycleGraph(6);
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);
  const distanceMatrix = eg.DistanceMatrix.allSourcesDijkstra(graph, () => 1.0);
  const rng = eg.Rng.seedFrom(BigInt(42));

  const gumap = new eg.GuMapBuilder()
    .k(3)
    .nEpochs(50)
    .build();

  gumap.run(drawing, distanceMatrix, rng);

  helpers.verifyFiniteCoordinates2d(drawing, graph);
};
