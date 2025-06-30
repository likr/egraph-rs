const assert = require("assert");
const eg = require("wasm-bindgen-test");
const helpers = require("./util/test_helpers");

/**
 * Test scheduler creation methods
 */
exports.testSgdSchedulers = function () {
  // Test creating different types of schedulers
  const constantScheduler = new eg.SchedulerConstant(100);
  assert(
    constantScheduler instanceof eg.SchedulerConstant,
    "Should create a constant scheduler"
  );

  const linearScheduler = new eg.SchedulerLinear(100);
  assert(
    linearScheduler instanceof eg.SchedulerLinear,
    "Should create a linear scheduler"
  );

  const quadraticScheduler = new eg.SchedulerQuadratic(100);
  assert(
    quadraticScheduler instanceof eg.SchedulerQuadratic,
    "Should create a quadratic scheduler"
  );

  const exponentialScheduler = new eg.SchedulerExponential(100);
  assert(
    exponentialScheduler instanceof eg.SchedulerExponential,
    "Should create an exponential scheduler"
  );

  const reciprocalScheduler = new eg.SchedulerReciprocal(100);
  assert(
    reciprocalScheduler instanceof eg.SchedulerReciprocal,
    "Should create a reciprocal scheduler"
  );

  // Test scheduler execution
  let callCount = 0;
  const scheduler = new eg.SchedulerExponential(5);
  helpers.runScheduler(scheduler, (eta) => {
    assert(typeof eta === "number", "Learning rate should be a number");
    assert(eta > 0, "Learning rate should be positive");
    callCount++;
  });
  assert.strictEqual(callCount, 5, "Scheduler should run exactly 5 times");

  // Test step-by-step execution
  callCount = 0;
  const stepScheduler = new eg.SchedulerExponential(5);
  helpers.runSchedulerStepByStep(stepScheduler, (eta) => {
    assert(typeof eta === "number", "Learning rate should be a number");
    assert(eta > 0, "Learning rate should be positive");
    callCount++;
  });
  assert.strictEqual(callCount, 5, "Step scheduler should run exactly 5 times");
};

/**
 * Test applying SGD to Euclidean 2D drawings
 */
exports.testSgdWithEuclidean2d = function () {
  // Create a test graph
  const { graph } = helpers.createLineGraph(3);

  // Create a Sgd instance
  const sgd = new eg.FullSgd().build(graph, () => 100);

  // Create a drawing
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Record initial positions
  const initialPositions = helpers.recordInitialPositions2d(drawing, graph);

  // Apply SGD
  sgd.applyWithDrawingEuclidean2d(drawing, 0.1);

  // Verify that positions have changed
  helpers.verifyPositionsChanged2d(
    drawing,
    graph,
    initialPositions,
    "SGD should change node positions"
  );

  // Verify that all coordinates are finite numbers
  helpers.verifyFiniteCoordinates2d(drawing, graph);
};

/**
 * Test applying SGD to Hyperbolic 2D drawings
 */
exports.testSgdWithHyperbolic2d = function () {
  // Create a test graph
  const { graph } = helpers.createLineGraph(3);

  // Create a Sgd instance
  const sgd = new eg.FullSgd().build(graph, () => 100);

  // Create a drawing
  const drawing = eg.DrawingHyperbolic2d.initialPlacement(graph);

  // Record initial positions
  const initialPositions = helpers.recordInitialPositions2d(drawing, graph);

  // Apply SGD
  sgd.applyWithDrawingHyperbolic2d(drawing, 0.1);

  // Verify that positions have changed
  helpers.verifyPositionsChanged2d(
    drawing,
    graph,
    initialPositions,
    "SGD should change node positions"
  );

  // Verify that all coordinates are finite numbers
  helpers.verifyFiniteCoordinates2d(drawing, graph);

  // Verify that all nodes are within the Poincaré disc
  helpers.verifyHyperbolicCoordinateRange(drawing, graph);
};

/**
 * Test applying SGD to Spherical 2D drawings
 */
exports.testSgdWithSpherical2d = function () {
  // Create a test graph
  const { graph } = helpers.createLineGraph(3);

  // Create a Sgd instance
  const sgd = new eg.FullSgd().build(graph, () => 100);

  // Create a drawing
  const drawing = eg.DrawingSpherical2d.initialPlacement(graph);

  // Record initial positions
  const initialPositions = helpers.recordInitialSphericalPositions(
    drawing,
    graph
  );

  // Apply SGD
  sgd.applyWithDrawingSpherical2d(drawing, 0.1);

  // Verify that positions have changed
  helpers.verifySphericalPositionsChanged(
    drawing,
    graph,
    initialPositions,
    "SGD should change node positions"
  );

  // Verify that all coordinates are finite numbers
  helpers.verifyFiniteSphericalCoordinates(drawing, graph);
};

/**
 * Test applying SGD to Torus 2D drawings
 */
exports.testSgdWithTorus2d = function () {
  // Create a test graph
  const { graph } = helpers.createLineGraph(3);

  // Create a Sgd instance
  const sgd = new eg.FullSgd().build(graph, () => 100);

  // Create a drawing
  const drawing = eg.DrawingTorus2d.initialPlacement(graph);

  // Record initial positions
  const initialPositions = helpers.recordInitialPositions2d(drawing, graph);

  // Apply SGD
  sgd.applyWithDrawingTorus2d(drawing, 0.1);

  // Verify that positions have changed
  helpers.verifyPositionsChanged2d(
    drawing,
    graph,
    initialPositions,
    "SGD should change node positions"
  );

  // Verify that all coordinates are finite numbers
  helpers.verifyFiniteCoordinates2d(drawing, graph);

  // Verify that coordinates are within the torus range (0 to 1)
  helpers.verifyTorusCoordinateRange(drawing, graph);
};

/**
 * Test applying SGD to n-dimensional Euclidean drawings
 */
exports.testSgdWithEuclidean = function () {
  // Create a test graph
  const { graph } = helpers.createLineGraph(3);

  // Create a Sgd instance
  const sgd = new eg.FullSgd().build(graph, () => 100);

  // Create a 3D drawing using ClassicalMds
  const mds = new eg.ClassicalMds(graph, () => 1.0);
  const drawing = mds.run(3);
  const dimensions = 3;

  // Record initial positions
  const initialPositions = helpers.recordInitialPositionsNd(
    drawing,
    graph,
    dimensions
  );

  // Apply SGD
  sgd.applyWithDrawingEuclidean(drawing, 0.1);

  // Verify that positions have changed
  helpers.verifyPositionsChangedNd(
    drawing,
    graph,
    initialPositions,
    dimensions,
    "SGD should change node positions"
  );

  // Verify that all coordinates are finite numbers
  helpers.verifyFiniteCoordinatesNd(drawing, graph, dimensions);
};

/**
 * Test updating distance function
 */
exports.testSgdUpdateDistance = function () {
  // Create a test graph
  const { graph } = helpers.createLineGraph(2);

  // Create a Sgd instance
  const sgd = new eg.FullSgd().build(graph, () => 100);

  // Update distance function to double the distance
  sgd.updateDistance((i, j, dij, wij) => dij * 2);
};

/**
 * Test updating weight function
 */
exports.testSgdUpdateWeight = function () {
  // Create a test graph
  const { graph } = helpers.createLineGraph(2);

  // Create a Sgd instance
  const sgd = new eg.FullSgd().build(graph, () => 100);

  // Update weight function to double the weight
  sgd.updateWeight((i, j, d, w) => w * 2);
};

/**
 * Test shuffling node pairs
 */
exports.testSgdShuffle = function () {
  // Create a test graph
  const { graph } = helpers.createLineGraph(3);

  // Create an RNG
  const rng = new eg.Rng();

  // Create a Sgd instance
  const sgd = new eg.FullSgd().build(graph, () => 100);

  // Create a drawing
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Apply SGD
  sgd.shuffle(rng);
  sgd.applyWithDrawingEuclidean2d(drawing, 0.1);

  // Verify that all coordinates are finite numbers
  helpers.verifyFiniteCoordinates2d(drawing, graph);

  // Test that the same seed produces the same shuffle result
  const rng1 = helpers.createSeededRng(42n);
  const rng2 = helpers.createSeededRng(42n);

  const drawing1 = eg.DrawingEuclidean2d.initialPlacement(graph);
  const drawing2 = eg.DrawingEuclidean2d.initialPlacement(graph);

  const sgd1 = new eg.FullSgd().build(graph, () => 100);
  const sgd2 = new eg.FullSgd().build(graph, () => 100);

  sgd1.shuffle(rng1);
  sgd2.shuffle(rng2);

  sgd1.applyWithDrawingEuclidean2d(drawing1, 0.1);
  sgd2.applyWithDrawingEuclidean2d(drawing2, 0.1);

  // Verify that the results are identical
  for (const u of graph.nodeIndices()) {
    assert.strictEqual(
      drawing1.x(u),
      drawing2.x(u),
      "X coordinates should be identical with same seed"
    );
    assert.strictEqual(
      drawing1.y(u),
      drawing2.y(u),
      "Y coordinates should be identical with same seed"
    );
  }
};
