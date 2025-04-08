const assert = require("assert");
const eg = require("wasm-bindgen-test");
const helpers = require("./util/test_helpers");

/**
 * Test basic instantiation of StressMajorization class
 */
exports.testStressMajorizationConstructor = function () {
  // Create a simple graph
  const { graph } = helpers.createTestGraph("line", 2);

  // Create a drawing
  const drawing = helpers.createDrawing(graph, "euclidean2d");

  // Create a StressMajorization instance with a simple distance function
  const layout = new eg.StressMajorization(graph, drawing, (e) => {
    return { distance: 1.0 };
  });

  // Verify that the layout instance exists
  assert(
    layout instanceof eg.StressMajorization,
    "Should create an instance of StressMajorization"
  );
};

/**
 * Test applying a single iteration of the stress majorization algorithm
 */
exports.testStressMajorizationApply = function () {
  // Create a simple graph
  const { graph } = helpers.createTestGraph("line", 3);

  // Create a drawing with initial positions
  const drawing = helpers.createDrawing(graph, "euclidean2d");

  // Set specific positions for testing
  drawing.setX(0, 0.0);
  drawing.setY(0, 0.0);
  drawing.setX(1, 1.0);
  drawing.setY(1, 0.0);
  drawing.setX(2, 0.5);
  drawing.setY(2, 1.0);

  // Create a StressMajorization instance
  const layout = new eg.StressMajorization(graph, drawing, (e) => {
    return { distance: 1.0 };
  });

  // Record initial positions
  const initialPositions = helpers.recordInitialPositions2d(drawing, graph);

  // Apply a single iteration
  const stress = layout.apply(drawing);

  // Verify that the stress value is a finite number
  assert(Number.isFinite(stress), "Stress value should be a finite number");

  // Verify that positions have changed
  helpers.verifyPositionsChanged2d(
    drawing,
    graph,
    initialPositions,
    "At least one node position should change after applying the algorithm"
  );

  // Verify that all coordinates are finite numbers
  helpers.verifyFiniteCoordinates2d(drawing, graph);
};

/**
 * Test running the complete stress majorization algorithm
 */
exports.testStressMajorizationRun = function () {
  // Create a cycle graph
  const { graph } = helpers.createTestGraph("cycle", 4);

  // Create a drawing and initial positions
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);
  const initialPositions = helpers.recordInitialPositions2d(drawing, graph);

  // Apply StressMajorization layout
  const sm = new eg.StressMajorization(graph, drawing, () => ({ distance: 1 }));
  sm.run(drawing);

  // Verify that positions have changed
  helpers.verifyPositionsChanged2d(
    drawing,
    graph,
    initialPositions,
    "Node positions should change after running the algorithm"
  );

  // Verify that all coordinates are finite numbers
  helpers.verifyFiniteCoordinates2d(drawing, graph);

  // For a cycle graph with uniform edge lengths, the layout should
  // approximate a regular polygon. Check that nodes are roughly
  // equidistant from the center.
  const centerX =
    (drawing.x(0) + drawing.x(1) + drawing.x(2) + drawing.x(3)) / 4;
  const centerY =
    (drawing.y(0) + drawing.y(1) + drawing.y(2) + drawing.y(3)) / 4;

  const distances = [];
  for (const u of graph.nodeIndices()) {
    const dx = drawing.x(u) - centerX;
    const dy = drawing.y(u) - centerY;
    distances.push(Math.sqrt(dx * dx + dy * dy));
  }

  // Calculate standard deviation of distances
  const avgDistance = distances.reduce((a, b) => a + b, 0) / distances.length;
  const variance =
    distances.reduce((a, b) => a + Math.pow(b - avgDistance, 2), 0) /
    distances.length;
  const stdDev = Math.sqrt(variance);

  // Check that the standard deviation is small relative to the average distance
  assert(
    stdDev / avgDistance < 0.2,
    "Nodes should be roughly equidistant from the center in a cycle graph"
  );
};

/**
 * Test getter and setter methods for epsilon and max_iterations
 */
exports.testStressMajorizationParameters = function () {
  // Create a simple graph
  const { graph } = helpers.createTestGraph("line", 3);

  // Create a drawing
  const drawing = helpers.createDrawing(graph, "euclidean2d");

  // Create a StressMajorization instance
  const layout = new eg.StressMajorization(graph, drawing, (e) => {
    return { distance: 1.0 };
  });

  // Check default values
  assert.strictEqual(
    typeof layout.epsilon,
    "number",
    "epsilon should be a number"
  );
  assert.strictEqual(
    typeof layout.max_iterations,
    "number",
    "max_iterations should be a number"
  );

  // Default values should be finite numbers
  assert(
    Number.isFinite(layout.epsilon),
    "Default epsilon should be a finite number"
  );
  assert(
    Number.isFinite(layout.max_iterations),
    "Default max_iterations should be a finite number"
  );

  // Test setters
  const newEpsilon = 1e-6;
  const newMaxIterations = 200;

  layout.epsilon = newEpsilon;
  layout.max_iterations = newMaxIterations;

  // Verify values were updated - use approximate comparison for floating point
  const epsilon = 1e-10; // Small value to account for floating point precision
  assert(
    Math.abs(layout.epsilon - newEpsilon) < epsilon,
    "epsilon should be approximately updated to the new value"
  );
  assert.strictEqual(
    layout.max_iterations,
    newMaxIterations,
    "max_iterations should be updated to the new value"
  );
};

/**
 * Test integration with other components and stress reduction
 */
exports.testStressMajorizationIntegration = function () {
  // Create a more complex graph
  const { graph } = helpers.createTestGraph("custom", 10, (graph, nodes) => {
    // Create a path
    for (let i = 0; i < 9; i++) {
      graph.addEdge(nodes[i], nodes[i + 1], {});
    }
    // Add some cross edges
    graph.addEdge(nodes[0], nodes[5], {});
    graph.addEdge(nodes[2], nodes[7], {});
    graph.addEdge(nodes[3], nodes[8], {});
  });

  // Create a drawing with initial positions
  const drawing = helpers.createDrawing(graph, "euclidean2d");

  // Set specific positions instead of using random values
  let i = 0;
  for (const u of graph.nodeIndices()) {
    // Use a simple pattern to distribute nodes
    const x = (i % 3) * 2 - 2; // -2, 0, 2, -2, 0, 2, ...
    const y = Math.floor(i / 3) * 2 - 2; // -2, -2, -2, 0, 0, 0, ...
    drawing.setX(u, x);
    drawing.setY(u, y);
    i++;
  }

  // Calculate initial stress using the stress metric
  const initialStress = eg.stress(graph, drawing);

  // Custom distance function that uses node IDs
  const customDistanceFunc = (e) => {
    const endpoints = graph.edgeEndpoints(e);
    const u = endpoints[0];
    const v = endpoints[1];
    // Use node IDs to determine distance
    const uId = graph.nodeWeight(u).id;
    const vId = graph.nodeWeight(v).id;
    return { distance: Math.abs(uId - vId) };
  };

  // Apply StressMajorization layout
  helpers.applyLayout("stress_majorization", graph, drawing, {
    distanceFunc: customDistanceFunc,
    iterations: 100,
  });

  // Calculate final stress
  const finalStress = eg.stress(graph, drawing);

  // Verify that stress has been reduced
  assert(
    finalStress < initialStress,
    "Stress should be reduced after running the algorithm"
  );

  // Verify that all coordinates are finite numbers
  helpers.verifyFiniteCoordinates2d(drawing, graph);

  // Verify that connected nodes with smaller ideal distances are positioned
  // closer together than those with larger ideal distances
  let shortEdgesCount = 0;
  let shortEdgesDistance = 0;
  let longEdgesCount = 0;
  let longEdgesDistance = 0;

  for (const e of graph.edgeIndices()) {
    const endpoints = graph.edgeEndpoints(e);
    const u = endpoints[0];
    const v = endpoints[1];
    const uId = graph.nodeWeight(u).id;
    const vId = graph.nodeWeight(v).id;
    const idealDistance = Math.abs(uId - vId);

    const dx = drawing.x(u) - drawing.x(v);
    const dy = drawing.y(u) - drawing.y(v);
    const distance = Math.sqrt(dx * dx + dy * dy);

    if (idealDistance <= 2) {
      shortEdgesDistance += distance;
      shortEdgesCount++;
    } else {
      longEdgesDistance += distance;
      longEdgesCount++;
    }
  }

  if (shortEdgesCount > 0 && longEdgesCount > 0) {
    const avgShortDistance = shortEdgesDistance / shortEdgesCount;
    const avgLongDistance = longEdgesDistance / longEdgesCount;

    assert(
      avgShortDistance < avgLongDistance,
      "Edges with smaller ideal distances should be positioned closer together"
    );
  }
};
