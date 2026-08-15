const assert = require("assert");
const eg = require("wasm-bindgen-test");
const helpers = require("./util/test_helpers");

/**
 * Test basic instantiation of TsNet class and TsNetBuilder
 */
exports.testTsNetConstructor = function () {
  // Test builder
  const builder = new eg.TsNetBuilder();
  assert(
    builder instanceof eg.TsNetBuilder,
    "Should create an instance of TsNetBuilder",
  );
  const tsNetBuilt = builder
    .perplexity(25.0)
    .nEpochs(100)
    .learningRate(100.0)
    .momentum(0.9)
    .initialLC(1.2)
    .finalLC(0.01)
    .initialLR(0.0)
    .finalLR(0.6)
    .rEps(0.06)
    .autostop(1e-6)
    .windowSize(30)
    .build();
  assert(tsNetBuilt instanceof eg.TsNet, "Should build TsNet instance");

  const starBuilder = eg.TsNetBuilder.newStar();
  assert(
    starBuilder instanceof eg.TsNetBuilder,
    "Should create TsNetBuilder for star",
  );
  const tsNetStar = starBuilder.build();
  assert(tsNetStar instanceof eg.TsNet, "Should build TsNet star instance");

  const tsNetStarDirect = eg.TsNet.newStar();
  assert(
    tsNetStarDirect instanceof eg.TsNet,
    "Should create TsNet star instance directly",
  );

  const tsNet = new eg.TsNet();
  assert(tsNet instanceof eg.TsNet, "Should create an instance of TsNet");

  // Test parameters getters and setters
  assert(typeof tsNet.perplexity === "number", "perplexity should be a number");
  assert.strictEqual(tsNet.perplexity, 40.0, "default perplexity should be 40");

  tsNet.perplexity = 25.0;
  assert.strictEqual(tsNet.perplexity, 25.0, "perplexity should be updated");

  assert.strictEqual(tsNet.nEpochs, 5000, "default nEpochs should be 5000");
  tsNet.nEpochs = 200;
  assert.strictEqual(tsNet.nEpochs, 200, "nEpochs should be updated");

  assert.strictEqual(tsNet.initialLr, 50.0, "default initialLr should be 50");
  tsNet.initialLr = 100.0;
  assert.strictEqual(tsNet.initialLr, 100.0, "initialLr should be updated");

  assert.strictEqual(tsNet.finalLr, 50.0, "default finalLr should be 50");
  tsNet.finalLr = 10.0;
  assert.strictEqual(tsNet.finalLr, 10.0, "finalLr should be updated");

  assert.ok(
    Math.abs(tsNet.initialMomentum - 0.5) < 1e-6,
    "default initialMomentum should be 0.5",
  );
  tsNet.initialMomentum = 0.9;
  assert.ok(
    Math.abs(tsNet.initialMomentum - 0.9) < 1e-6,
    "initialMomentum should be updated",
  );

  assert.ok(Math.abs(tsNet.rEps - 0.05) < 1e-6, "default rEps should be 0.05");
  tsNet.rEps = 0.06;
  assert.ok(Math.abs(tsNet.rEps - 0.06) < 1e-6, "rEps should be updated");

  assert.ok(
    Math.abs(tsNet.autostop - 1e-7) < 1e-12,
    "default autostop should be 1e-7",
  );
  tsNet.autostop = 1e-5;
  assert.ok(
    Math.abs(tsNet.autostop - 1e-5) < 1e-12,
    "autostop should be updated",
  );

  assert.strictEqual(tsNet.windowSize, 40, "default windowSize should be 40");
  tsNet.windowSize = 20;
  assert.strictEqual(tsNet.windowSize, 20, "windowSize should be updated");
};

/**
 * Test running TsNet with simple DistanceMatrix
 */
exports.testTsNetRun = function () {
  const { graph } = helpers.createCycleGraph(4);
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Create distance matrix using all sources dijkstra
  const distanceMatrix = eg.DistanceMatrix.allSourcesDijkstra(graph, () => 1.0);

  const tsNet = new eg.TsNetBuilder().learningRate(2.0).nEpochs(20).build();

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

  const tsNet = new eg.TsNetBuilder().learningRate(2.0).nEpochs(20).build();

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
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
  ]);
  const distanceMatrix = eg.DistanceMatrix.embedding(
    graph,
    coordinates,
    3,
    1e-3,
  );

  const tsNet = new eg.TsNetBuilder().learningRate(2.0).nEpochs(20).build();

  tsNet.run(drawing, distanceMatrix);

  helpers.verifyFiniteCoordinates2d(drawing, graph);
};

/**
 * Test running TsNet with Kernel distance matrix
 */
exports.testTsNetWithKernelDistance = function () {
  const { graph } = helpers.createCycleGraph(4);
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  const coordinates = new Float32Array([
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
  ]);
  const baseDistance = eg.DistanceMatrix.embedding(graph, coordinates, 3, 1e-3);
  const distanceMatrix = eg.DistanceMatrix.kernel(baseDistance, 0.5);

  const tsNet = new eg.TsNetBuilder()
    .learningRate(2.0)
    .nEpochs(20)
    .autostop(1e-7)
    .windowSize(40)
    .build();

  tsNet.run(drawing, distanceMatrix);

  helpers.verifyFiniteCoordinates2d(drawing, graph);
};

/**
 * Test basic instantiation of BhTsNet class and BhTsNetBuilder
 */
exports.testBhTsNetConstructor = function () {
  const builder = new eg.BhTsNetBuilder();
  assert(
    builder instanceof eg.BhTsNetBuilder,
    "Should create an instance of BhTsNetBuilder",
  );
  const bhTsNetBuilt = builder
    .perplexity(25.0)
    .theta(0.6)
    .k(50)
    .iterationsStage2(100)
    .iterationsStage3(150)
    .lambdaCStage2(1.2)
    .lambdaCStage3(0.01)
    .lambdaRStage3(0.6)
    .learningRate(100.0)
    .momentum(0.9)
    .epsilonR(0.06)
    .build();
  assert(bhTsNetBuilt instanceof eg.BhTsNet, "Should build BhTsNet instance");

  const bhTsNet = new eg.BhTsNet();
  assert(bhTsNet instanceof eg.BhTsNet, "Should create an instance of BhTsNet");
  assert.strictEqual(
    bhTsNet.perplexity,
    40.0,
    "default perplexity should be 40",
  );
  assert.strictEqual(bhTsNet.theta, 0.5, "default theta should be 0.5");
};

/**
 * Test running BhTsNet on graph
 */
exports.testBhTsNetRun = function () {
  const { graph } = helpers.createCycleGraph(6);
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);
  const rng = eg.Rng.seedFrom(BigInt(42));

  const bhTsNet = new eg.BhTsNetBuilder()
    .perplexity(2.0)
    .k(4)
    .theta(0.5)
    .learningRate(2.0)
    .iterationsStage1(10)
    .iterationsStage2(10)
    .iterationsStage3(10)
    .build();

  bhTsNet.run(drawing, graph, rng);

  helpers.verifyFiniteCoordinates2d(drawing, graph);
};

/**
 * Test basic instantiation of FitTsNet class and FitTsNetBuilder
 */
exports.testFitTsNetConstructor = function () {
  const builder = new eg.FitTsNetBuilder();
  assert(
    builder instanceof eg.FitTsNetBuilder,
    "Should create an instance of FitTsNetBuilder",
  );
  const fitTsNetBuilt = builder
    .intervals(20)
    .interpolationPoints(3)
    .perplexity(25.0)
    .theta(0.6)
    .k(50)
    .iterationsStage2(100)
    .iterationsStage3(150)
    .lambdaCStage2(1.2)
    .lambdaCStage3(0.01)
    .lambdaRStage3(0.6)
    .learningRate(100.0)
    .momentum(0.9)
    .epsilonR(0.06)
    .build();
  assert(
    fitTsNetBuilt instanceof eg.FitTsNet,
    "Should build FitTsNet instance",
  );

  const fitTsNet = new eg.FitTsNet();
  assert(
    fitTsNet instanceof eg.FitTsNet,
    "Should create an instance of FitTsNet",
  );
  assert.strictEqual(
    fitTsNet.perplexity,
    40.0,
    "default perplexity should be 40",
  );
  assert.strictEqual(fitTsNet.intervals, 25, "default intervals should be 25");
  assert.strictEqual(
    fitTsNet.interpolationPoints,
    3,
    "default interpolationPoints should be 3",
  );
};

/**
 * Test running FitTsNet on graph
 */
exports.testFitTsNetRun = function () {
  const { graph } = helpers.createCycleGraph(6);
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);
  const rng = eg.Rng.seedFrom(BigInt(42));

  const fitTsNet = new eg.FitTsNetBuilder()
    .intervals(10)
    .interpolationPoints(3)
    .perplexity(2.0)
    .k(4)
    .theta(0.5)
    .learningRate(2.0)
    .iterationsStage1(10)
    .iterationsStage2(10)
    .iterationsStage3(10)
    .build();

  fitTsNet.run(drawing, graph, rng);

  helpers.verifyFiniteCoordinates2d(drawing, graph);
};

/**
 * Test basic instantiation of LTsNet class and LTsNetBuilder
 */
exports.testLTsNetConstructor = function () {
  const builder = new eg.LTsNetBuilder();
  assert(
    builder instanceof eg.LTsNetBuilder,
    "Should create an instance of LTsNetBuilder",
  );
  const lTsNetBuilt = builder
    .intervals(20)
    .interpolationPoints(3)
    .perplexity(25.0)
    .k(50)
    .iterationsStage2(100)
    .iterationsStage3(150)
    .lambdaCStage2(1.2)
    .lambdaCStage3(0.01)
    .lambdaRStage3(0.6)
    .learningRate(100.0)
    .momentum(0.9)
    .epsilonR(0.06)
    .build();
  assert(lTsNetBuilt instanceof eg.LTsNet, "Should build LTsNet instance");

  const lTsNet = new eg.LTsNet();
  assert(lTsNet instanceof eg.LTsNet, "Should create an instance of LTsNet");
  assert.strictEqual(
    lTsNet.perplexity,
    40.0,
    "default perplexity should be 40",
  );
  assert.strictEqual(lTsNet.intervals, 25, "default intervals should be 25");
  assert.strictEqual(
    lTsNet.interpolationPoints,
    3,
    "default interpolationPoints should be 3",
  );
};

/**
 * Test running LTsNet on graph
 */
exports.testLTsNetRun = function () {
  const { graph } = helpers.createCycleGraph(6);
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);
  const rng = eg.Rng.seedFrom(BigInt(42));

  const lTsNet = new eg.LTsNetBuilder()
    .intervals(10)
    .interpolationPoints(3)
    .perplexity(2.0)
    .k(4)
    .learningRate(2.0)
    .iterationsStage1(10)
    .iterationsStage2(10)
    .iterationsStage3(10)
    .build();

  lTsNet.run(drawing, graph, rng);

  helpers.verifyFiniteCoordinates2d(drawing, graph);
};
