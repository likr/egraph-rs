const assert = require("assert");
const eg = require("wasm-bindgen-test");
const helpers = require("./util/test_helpers");

/**
 * Test basic instantiation of FullSgd class
 */
exports.testFullSgdConstructor = function () {
  // Create a test graph
  const { graph } = helpers.createLineGraph(2);

  // Create a FullSgd instance with a simple length function
  const sgd = new eg.FullSgd(graph, () => 100);

  // Verify that the SGD instance exists
  assert(sgd instanceof eg.FullSgd, "Should create an instance of FullSgd");
};

/**
 * Test scheduler creation methods
 */
exports.testFullSgdSchedulers = function () {
  // Create a test graph
  const { graph } = helpers.createLineGraph(2);

  // Create a FullSgd instance
  const sgd = new eg.FullSgd(graph, () => 100);

  // Test creating different types of schedulers
  const constantScheduler = sgd.schedulerConstant(100, 0.1);
  assert(
    constantScheduler instanceof eg.SchedulerConstant,
    "Should create a constant scheduler"
  );

  const linearScheduler = sgd.schedulerLinear(100, 0.1);
  assert(
    linearScheduler instanceof eg.SchedulerLinear,
    "Should create a linear scheduler"
  );

  const quadraticScheduler = sgd.schedulerQuadratic(100, 0.1);
  assert(
    quadraticScheduler instanceof eg.SchedulerQuadratic,
    "Should create a quadratic scheduler"
  );

  const exponentialScheduler = sgd.schedulerExponential(100, 0.1);
  assert(
    exponentialScheduler instanceof eg.SchedulerExponential,
    "Should create an exponential scheduler"
  );

  const reciprocalScheduler = sgd.schedulerReciprocal(100, 0.1);
  assert(
    reciprocalScheduler instanceof eg.SchedulerReciprocal,
    "Should create a reciprocal scheduler"
  );

  // Test the default scheduler (should be exponential)
  const defaultScheduler = sgd.scheduler(100, 0.1);
  assert(
    defaultScheduler instanceof eg.SchedulerExponential,
    "Default scheduler should be exponential"
  );

  // Test scheduler execution
  let callCount = 0;
  const scheduler = sgd.scheduler(5, 0.1);
  helpers.runScheduler(scheduler, (eta) => {
    assert(typeof eta === "number", "Learning rate should be a number");
    assert(eta > 0, "Learning rate should be positive");
    callCount++;
  });
  assert.strictEqual(callCount, 5, "Scheduler should run exactly 5 times");

  // Test step-by-step execution
  callCount = 0;
  const stepScheduler = sgd.scheduler(5, 0.1);
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
exports.testFullSgdWithEuclidean2d = function () {
  // Create a test graph
  const { graph } = helpers.createLineGraph(3);

  // Create a FullSgd instance
  const sgd = new eg.FullSgd(graph, () => 100);

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
exports.testFullSgdWithHyperbolic2d = function () {
  // Create a test graph
  const { graph } = helpers.createLineGraph(3);

  // Create a FullSgd instance
  const sgd = new eg.FullSgd(graph, () => 100);

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
exports.testFullSgdWithSpherical2d = function () {
  // Create a test graph
  const { graph } = helpers.createLineGraph(3);

  // Create a FullSgd instance
  const sgd = new eg.FullSgd(graph, () => 100);

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

  // Verify that latitude is within valid range (-π/2 to π/2)
  helpers.verifySphericalCoordinateRange(drawing, graph);
};

/**
 * Test applying SGD to Torus 2D drawings
 */
exports.testFullSgdWithTorus2d = function () {
  // Create a test graph
  const { graph } = helpers.createLineGraph(3);

  // Create a FullSgd instance
  const sgd = new eg.FullSgd(graph, () => 100);

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
exports.testFullSgdWithEuclidean = function () {
  // Create a test graph
  const { graph } = helpers.createLineGraph(3);

  // Create a FullSgd instance
  const sgd = new eg.FullSgd(graph, () => 100);

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
exports.testFullSgdUpdateDistance = function () {
  // Create a test graph
  const { graph } = helpers.createLineGraph(2);

  // Create a FullSgd instance
  const sgd = new eg.FullSgd(graph, () => 100);

  // Create a drawing
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Apply SGD with default distance function
  sgd.applyWithDrawingEuclidean2d(drawing, 0.1);
  const positionsAfterDefault = helpers.recordInitialPositions2d(
    drawing,
    graph
  );

  // Reset drawing
  const resetDrawing = eg.DrawingEuclidean2d.initialPlacement(graph);
  for (const u of graph.nodeIndices()) {
    drawing.setX(u, resetDrawing.x(u));
    drawing.setY(u, resetDrawing.y(u));
  }

  // Update distance function to double the distance
  sgd.updateDistance((i, j, dij, wij) => dij * 2);

  // Apply SGD with modified distance function
  sgd.applyWithDrawingEuclidean2d(drawing, 0.1);

  // Verify that positions are different from the default distance function
  let positionsDifferent = false;
  for (const u of graph.nodeIndices()) {
    if (
      Math.abs(drawing.x(u) - positionsAfterDefault[u].x) > 1e-10 ||
      Math.abs(drawing.y(u) - positionsAfterDefault[u].y) > 1e-10
    ) {
      positionsDifferent = true;
      break;
    }
  }
  assert(
    positionsDifferent,
    "Updating distance function should change the layout behavior"
  );
};

/**
 * Test updating weight function
 */
exports.testFullSgdUpdateWeight = function () {
  // Create a test graph
  const { graph } = helpers.createLineGraph(2);

  // Create a FullSgd instance
  const sgd = new eg.FullSgd(graph, () => 100);

  // Create a drawing
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Apply SGD with default weight function
  sgd.applyWithDrawingEuclidean2d(drawing, 0.1);
  const positionsAfterDefault = helpers.recordInitialPositions2d(
    drawing,
    graph
  );

  // Reset drawing
  const resetDrawing = eg.DrawingEuclidean2d.initialPlacement(graph);
  for (const u of graph.nodeIndices()) {
    drawing.setX(u, resetDrawing.x(u));
    drawing.setY(u, resetDrawing.y(u));
  }

  // Update weight function to double the weight
  sgd.updateWeight((i, j, d, w) => w * 2);

  // Apply SGD with modified weight function
  sgd.applyWithDrawingEuclidean2d(drawing, 0.1);

  // Verify that positions are different from the default weight function
  let positionsDifferent = false;
  for (const u of graph.nodeIndices()) {
    if (
      Math.abs(drawing.x(u) - positionsAfterDefault[u].x) > 1e-10 ||
      Math.abs(drawing.y(u) - positionsAfterDefault[u].y) > 1e-10
    ) {
      positionsDifferent = true;
      break;
    }
  }
  assert(
    positionsDifferent,
    "Updating weight function should change the layout behavior"
  );
};

/**
 * Test shuffling node pairs
 */
exports.testFullSgdShuffle = function () {
  // Create a test graph
  const { graph } = helpers.createLineGraph(3);

  // Create a FullSgd instance
  const sgd = new eg.FullSgd(graph, () => 100);

  // Create a drawing
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Create an RNG
  const rng = new eg.Rng();

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

  const sgd1 = new eg.FullSgd(graph, () => 100);
  const sgd2 = new eg.FullSgd(graph, () => 100);

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

/**
 * Test integration with other components
 */
exports.testFullSgdIntegration = function () {
  // Create a custom graph with cross edges
  const graph = new eg.Graph();
  const nodes = [];

  // Create nodes
  for (let i = 0; i < 10; i++) {
    nodes.push(graph.addNode({ id: i }));
  }

  // Create a path
  for (let i = 0; i < 9; i++) {
    graph.addEdge(nodes[i], nodes[i + 1], {});
  }

  // Add some cross edges
  graph.addEdge(nodes[0], nodes[5], {});
  graph.addEdge(nodes[2], nodes[7], {});
  graph.addEdge(nodes[3], nodes[8], {});

  // Create a drawing
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Create an RNG with a seed for reproducibility
  const rng = helpers.createSeededRng(123n);

  // Create a FullSgd instance
  const sgd = new eg.FullSgd(graph, () => 100);

  // Run a complete layout process
  const scheduler = sgd.scheduler(10, 0.1);
  helpers.runScheduler(scheduler, (eta) => {
    sgd.shuffle(rng);
    sgd.applyWithDrawingEuclidean2d(drawing, eta);
  });

  // Verify that all coordinates are finite numbers
  helpers.verifyFiniteCoordinates2d(drawing, graph);

  // Verify that connected nodes are positioned relatively close to each other
  helpers.verifyConnectedNodesCloser(graph, drawing);
};
