const assert = require("assert");
const eg = require("wasm-bindgen-test");
const helpers = require("./util/test_helpers");

/**
 * Test basic instantiation of TsNet class
 */
exports.testTsNetConstructor = function () {
  const tsNet = new eg.TsNet();
  assert(tsNet instanceof eg.TsNet, "Should create an instance of TsNet");
  
  // Test parameters getters and setters
  assert(typeof tsNet.perplexity === "number", "perplexity should be a number");
  assert.strictEqual(tsNet.perplexity, 30.0, "default perplexity should be 30");
  
  tsNet.perplexity = 25.0;
  assert.strictEqual(tsNet.perplexity, 25.0, "perplexity should be updated");
  
  assert.strictEqual(tsNet.iterationsStage2, 250, "default iterationsStage2 should be 250");
  tsNet.iterationsStage2 = 100;
  assert.strictEqual(tsNet.iterationsStage2, 100, "iterationsStage2 should be updated");

  assert.strictEqual(tsNet.iterationsStage3, 250, "default iterationsStage3 should be 250");
  tsNet.iterationsStage3 = 150;
  assert.strictEqual(tsNet.iterationsStage3, 150, "iterationsStage3 should be updated");

  assert.strictEqual(tsNet.learningRate, 200.0, "default learningRate should be 200");
  tsNet.learningRate = 100.0;
  assert.strictEqual(tsNet.learningRate, 100.0, "learningRate should be updated");

  assert.ok(Math.abs(tsNet.momentum - 0.8) < 1e-6, "default momentum should be 0.8");
  tsNet.momentum = 0.9;
  assert.ok(Math.abs(tsNet.momentum - 0.9) < 1e-6, "momentum should be updated");

  assert.ok(Math.abs(tsNet.epsilonD - 0.01) < 1e-6, "default epsilonD should be 0.01");
  tsNet.epsilonD = 0.02;
  assert.ok(Math.abs(tsNet.epsilonD - 0.02) < 1e-6, "epsilonD should be updated");

  assert.ok(Math.abs(tsNet.epsilonR - 0.05) < 1e-6, "default epsilonR should be 0.05");
  tsNet.epsilonR = 0.06;
  assert.ok(Math.abs(tsNet.epsilonR - 0.06) < 1e-6, "epsilonR should be updated");
};

/**
 * Test running TsNet with simple DistanceMatrix
 */
exports.testTsNetRun = function () {
  const { graph } = helpers.createCycleGraph(4);
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);
  
  // Create distance matrix using all sources dijkstra
  const distanceMatrix = eg.DistanceMatrix.allSourcesDijkstra(graph, () => 1.0);
  
  const tsNet = new eg.TsNet();
  tsNet.learningRate = 2.0;
  tsNet.iterationsStage2 = 10;
  tsNet.iterationsStage3 = 10;
  
  tsNet.run(drawing, distanceMatrix);
  
  helpers.verifyFiniteCoordinates2d(drawing, graph);
};

/**
 * Test running TsNet with Diffusion distance matrix
 */
exports.testTsNetWithDiffusionDistance = function () {
  const { graph } = helpers.createCycleGraph(4);
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);
  const rng = new eg.Rng();
  
  const kernel = new eg.DiffusionKernel(graph, () => 1.0, 1000.0, 10, 50, rng);
  const distanceMatrix = eg.DistanceMatrix.diffusion(graph, kernel, 1e-3);
  
  const tsNet = new eg.TsNet();
  tsNet.learningRate = 2.0;
  tsNet.iterationsStage2 = 10;
  tsNet.iterationsStage3 = 10;
  
  tsNet.run(drawing, distanceMatrix);
  
  helpers.verifyFiniteCoordinates2d(drawing, graph);
};

/**
 * Test running TsNet with Embedding distance matrix
 */
exports.testTsNetWithEmbeddingDistance = function () {
  const { graph } = helpers.createCycleGraph(4);
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);
  
  const coordinates = new Float32Array([
    0.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 1.0, 0.0,
    0.0, 1.0, 1.0
  ]);
  const distanceMatrix = eg.DistanceMatrix.embedding(graph, coordinates, 3, 1e-3);
  
  const tsNet = new eg.TsNet();
  tsNet.learningRate = 2.0;
  tsNet.iterationsStage2 = 10;
  tsNet.iterationsStage3 = 10;
  
  tsNet.run(drawing, distanceMatrix);
  
  helpers.verifyFiniteCoordinates2d(drawing, graph);
};

/**
 * Test running TsNet with Kernel distance matrix
 */
exports.testTsNetWithKernelDistance = function () {
  const { graph } = helpers.createCycleGraph(4);
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);
  
  const baseDistance = eg.DistanceMatrix.allSourcesDijkstra(graph, () => 1.0);
  const distanceMatrix = eg.DistanceMatrix.kernel(baseDistance, 0.5);
  
  const tsNet = new eg.TsNet();
  tsNet.learningRate = 2.0;
  tsNet.iterationsStage2 = 10;
  tsNet.iterationsStage3 = 10;
  
  tsNet.run(drawing, distanceMatrix);
  
  helpers.verifyFiniteCoordinates2d(drawing, graph);
};
