const assert = require("assert");
const eg = require("wasm-bindgen-test");
const helpers = require("./util/test_helpers");

/**
 * Test basic instantiation of SparseSgd class
 */
exports.testSparseSgdConstructor = function () {
  // Create a test graph
  const { graph } = helpers.createTestGraph("line", 2);

  // Create an RNG
  const rng = new eg.Rng();

  // Create a SparseSgd instance with a simple length function and 1 pivot node
  const sgd = new eg.SparseSgd(graph, () => 100, 1, rng);

  // Verify that the SGD instance exists
  assert(sgd instanceof eg.SparseSgd, "Should create an instance of SparseSgd");
};

/**
 * Test SparseSgd with different numbers of pivot nodes
 */
exports.testSparseSgdWithDifferentPivots = function () {
  // Create a larger graph
  const graph = new eg.Graph();
  const nodes = [];
  for (let i = 0; i < 10; i++) {
    nodes.push(graph.addNode({}));
  }
  // Add some edges to create a connected graph
  for (let i = 0; i < 9; i++) {
    graph.addEdge(nodes[i], nodes[i + 1], {});
  }
  // Add some cross edges
  graph.addEdge(nodes[0], nodes[5], {});
  graph.addEdge(nodes[2], nodes[7], {});
  graph.addEdge(nodes[3], nodes[8], {});

  // Create an RNG with a seed for reproducibility
  const rng = eg.Rng.seedFrom(123n);

  // Test with different numbers of pivot nodes
  const pivotCounts = [1, 3, 5];
  const drawings = [];

  for (const pivotCount of pivotCounts) {
    // Create a new RNG with the same seed for each test
    const testRng = eg.Rng.seedFrom(123n);

    // Create a SparseSgd instance with the specified number of pivot nodes
    const sgd = new eg.SparseSgd(graph, () => 100, pivotCount, testRng);

    // Create a drawing
    const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

    // Apply SGD
    const scheduler = sgd.scheduler(10, 0.1);
    scheduler.run((eta) => {
      // Create a new RNG with the same seed for shuffling
      const shuffleRng = eg.Rng.seedFrom(456n);
      sgd.shuffle(shuffleRng);
      sgd.applyWithDrawingEuclidean2d(drawing, eta);
    });

    // Verify that all coordinates are finite numbers
    for (const u of graph.nodeIndices()) {
      assert(
        Number.isFinite(drawing.x(u)),
        `X coordinate should be a finite number with ${pivotCount} pivots`
      );
      assert(
        Number.isFinite(drawing.y(u)),
        `Y coordinate should be a finite number with ${pivotCount} pivots`
      );
    }

    drawings.push(drawing);
  }

  // Verify that different pivot counts produce different layouts
  // Compare the first and last layouts (most different pivot counts)
  let positionsDifferent = false;
  for (const u of graph.nodeIndices()) {
    if (
      Math.abs(drawings[0].x(u) - drawings[drawings.length - 1].x(u)) > 1e-10 ||
      Math.abs(drawings[0].y(u) - drawings[drawings.length - 1].y(u)) > 1e-10
    ) {
      positionsDifferent = true;
      break;
    }
  }
  assert(
    positionsDifferent,
    "Different pivot counts should produce different layouts"
  );
};

/**
 * Test scheduler creation methods
 */
exports.testSparseSgdSchedulers = function () {
  // Create a test graph
  const { graph } = helpers.createTestGraph("line", 2);

  // Create an RNG
  const rng = new eg.Rng();

  // Create a SparseSgd instance
  const sgd = new eg.SparseSgd(graph, () => 100, 1, rng);

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
exports.testSparseSgdWithEuclidean2d = function () {
  // Create a test graph
  const { graph } = helpers.createTestGraph("line", 3);

  // Create an RNG
  const rng = new eg.Rng();

  // Create a SparseSgd instance
  const sgd = new eg.SparseSgd(graph, () => 100, 2, rng);

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
exports.testSparseSgdWithHyperbolic2d = function () {
  // Create a test graph
  const { graph } = helpers.createTestGraph("line", 3);

  // Create an RNG
  const rng = new eg.Rng();

  // Create a SparseSgd instance
  const sgd = new eg.SparseSgd(graph, () => 100, 2, rng);

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
exports.testSparseSgdWithSpherical2d = function () {
  // Create a test graph
  const { graph } = helpers.createTestGraph("line", 3);

  // Create an RNG
  const rng = helpers.createSeededRng(43n);

  // Create a SparseSgd instance
  const sgd = new eg.SparseSgd(graph, () => 0.5, 2, rng);

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
exports.testSparseSgdWithTorus2d = function () {
  // Create a test graph
  const { graph } = helpers.createTestGraph("line", 3);

  // Create an RNG
  const rng = new eg.Rng();

  // Create a SparseSgd instance
  const sgd = new eg.SparseSgd(graph, () => 100, 2, rng);

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
exports.testSparseSgdWithEuclidean = function () {
  // Create a test graph
  const { graph } = helpers.createTestGraph("line", 3);

  // Create an RNG
  const rng = new eg.Rng();

  // Create a SparseSgd instance
  const sgd = new eg.SparseSgd(graph, () => 100, 2, rng);

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
exports.testSparseSgdUpdateDistance = function () {
  // Create a test graph
  const { graph } = helpers.createTestGraph("line", 2);

  // Create an RNG
  const rng = new eg.Rng();

  // Create a SparseSgd instance
  const sgd = new eg.SparseSgd(graph, () => 100, 1, rng);

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
exports.testSparseSgdUpdateWeight = function () {
  // Create a test graph
  const { graph } = helpers.createTestGraph("line", 2);

  // Create an RNG
  const rng = new eg.Rng();

  // Create a SparseSgd instance
  const sgd = new eg.SparseSgd(graph, () => 100, 1, rng);

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
exports.testSparseSgdShuffle = function () {
  // Create a test graph
  const { graph } = helpers.createTestGraph("line", 3);

  // Create an RNG
  const rng = new eg.Rng();

  // Create a SparseSgd instance
  const sgd = new eg.SparseSgd(graph, () => 100, 2, rng);

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

  const sgd1 = new eg.SparseSgd(graph, () => 100, 2, rng1);
  const sgd2 = new eg.SparseSgd(graph, () => 100, 2, rng2);

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
exports.testSparseSgdIntegration = function () {
  // Create a custom graph with cross edges
  const { graph, nodes } = helpers.createTestGraph(
    "custom",
    10,
    (graph, nodes) => {
      // Create a path
      for (let i = 0; i < 9; i++) {
        graph.addEdge(nodes[i], nodes[i + 1], {});
      }
      // Add some cross edges
      graph.addEdge(nodes[0], nodes[5], {});
      graph.addEdge(nodes[2], nodes[7], {});
      graph.addEdge(nodes[3], nodes[8], {});
    }
  );

  // Create a drawing
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Create an RNG with a seed for reproducibility
  const rng = helpers.createSeededRng(123n);

  // Create a SparseSgd instance with 3 pivot nodes
  const sgd = new eg.SparseSgd(graph, () => 100, 3, rng);

  // Run a complete layout process
  const scheduler = sgd.scheduler(10, 0.1);
  helpers.runScheduler(scheduler, (eta) => {
    // Create a new RNG for each shuffle with the same seed
    const shuffleRng = helpers.createSeededRng(456n);
    sgd.shuffle(shuffleRng);
    sgd.applyWithDrawingEuclidean2d(drawing, eta);
  });

  // Verify that all coordinates are finite numbers
  helpers.verifyFiniteCoordinates2d(drawing, graph);

  // Verify that connected nodes are positioned relatively close to each other
  helpers.verifyConnectedNodesCloser(graph, drawing);

  // Compare with FullSgd to verify that SparseSgd produces reasonable results
  const fullDrawing = eg.DrawingEuclidean2d.initialPlacement(graph);
  const fullSgd = new eg.FullSgd(graph, () => 100);
  const fullScheduler = fullSgd.scheduler(10, 0.1);

  // Use the same RNG for shuffling to make the comparison more fair
  const fullRng = eg.Rng.seedFrom(123n);

  fullScheduler.run((eta) => {
    // Create a new RNG for each shuffle with the same seed
    const shuffleRng = eg.Rng.seedFrom(456n);
    fullSgd.shuffle(shuffleRng);
    fullSgd.applyWithDrawingEuclidean2d(fullDrawing, eta);
  });

  // Calculate stress for both layouts
  let sparseStress = 0;
  let fullStress = 0;

  // Get node indices array
  const nodeIndices = Array.from(graph.nodeIndices());

  for (let i = 0; i < nodeIndices.length; i++) {
    for (let j = i + 1; j < nodeIndices.length; j++) {
      const u = nodeIndices[i];
      const v = nodeIndices[j];

      // Calculate Euclidean distance in the layout
      const sparseDx = drawing.x(u) - drawing.x(v);
      const sparseDy = drawing.y(u) - drawing.y(v);
      const sparseDistance = Math.sqrt(
        sparseDx * sparseDx + sparseDy * sparseDy
      );

      const fullDx = fullDrawing.x(u) - fullDrawing.x(v);
      const fullDy = fullDrawing.y(u) - fullDrawing.y(v);
      const fullDistance = Math.sqrt(fullDx * fullDx + fullDy * fullDy);

      // For simplicity, we'll use a basic stress calculation
      // In a real stress calculation, we'd use graph-theoretic distances
      sparseStress += Math.pow(sparseDistance - 1.0, 2);
      fullStress += Math.pow(fullDistance - 1.0, 2);
    }
  }

  // We don't expect SparseSgd to be better than FullSgd, but it should be reasonable
  assert(
    sparseStress < fullStress * 3, // Allow SparseSgd to have up to 3x the stress of FullSgd
    "SparseSgd should produce layouts with reasonable stress compared to FullSgd"
  );
};
