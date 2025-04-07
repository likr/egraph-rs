const assert = require("assert");
const eg = require("wasm-bindgen-test");

/**
 * Test basic instantiation of FullSgd class
 */
exports.testFullSgdConstructor = function () {
  // Create a simple graph
  const graph = new eg.Graph();
  const node1 = graph.addNode({});
  const node2 = graph.addNode({});
  const edge = graph.addEdge(node1, node2, {});

  // Create a FullSgd instance with a simple length function
  const sgd = new eg.FullSgd(graph, () => 100);

  // Verify that the SGD instance exists
  assert(sgd instanceof eg.FullSgd, "Should create an instance of FullSgd");
};

/**
 * Test scheduler creation methods
 */
exports.testFullSgdSchedulers = function () {
  // Create a simple graph
  const graph = new eg.Graph();
  const node1 = graph.addNode({});
  const node2 = graph.addNode({});
  graph.addEdge(node1, node2, {});

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
  scheduler.run((eta) => {
    assert(typeof eta === "number", "Learning rate should be a number");
    assert(eta > 0, "Learning rate should be positive");
    callCount++;
  });
  assert.strictEqual(callCount, 5, "Scheduler should run exactly 5 times");

  // Test step-by-step execution
  callCount = 0;
  const stepScheduler = sgd.scheduler(5, 0.1);
  while (!stepScheduler.isFinished()) {
    stepScheduler.step((eta) => {
      assert(typeof eta === "number", "Learning rate should be a number");
      assert(eta > 0, "Learning rate should be positive");
      callCount++;
    });
  }
  assert.strictEqual(callCount, 5, "Step scheduler should run exactly 5 times");
};

/**
 * Test applying SGD to Euclidean 2D drawings
 */
exports.testFullSgdWithEuclidean2d = function () {
  // Create a simple graph
  const graph = new eg.Graph();
  const node1 = graph.addNode({});
  const node2 = graph.addNode({});
  const node3 = graph.addNode({});
  graph.addEdge(node1, node2, {});
  graph.addEdge(node2, node3, {});

  // Create a FullSgd instance
  const sgd = new eg.FullSgd(graph, () => 100);

  // Create a drawing
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Record initial positions
  const initialPositions = {};
  for (const u of graph.nodeIndices()) {
    initialPositions[u] = { x: drawing.x(u), y: drawing.y(u) };
  }

  // Apply SGD
  sgd.applyWithDrawingEuclidean2d(drawing, 0.1);

  // Verify that positions have changed
  let positionsChanged = false;
  for (const u of graph.nodeIndices()) {
    if (
      drawing.x(u) !== initialPositions[u].x ||
      drawing.y(u) !== initialPositions[u].y
    ) {
      positionsChanged = true;
      break;
    }
  }
  assert(positionsChanged, "SGD should change node positions");

  // Verify that all coordinates are finite numbers
  for (const u of graph.nodeIndices()) {
    assert(
      Number.isFinite(drawing.x(u)),
      "X coordinate should be a finite number"
    );
    assert(
      Number.isFinite(drawing.y(u)),
      "Y coordinate should be a finite number"
    );
  }
};

/**
 * Test applying SGD to Hyperbolic 2D drawings
 */
exports.testFullSgdWithHyperbolic2d = function () {
  // Create a simple graph
  const graph = new eg.Graph();
  const node1 = graph.addNode({});
  const node2 = graph.addNode({});
  const node3 = graph.addNode({});
  graph.addEdge(node1, node2, {});
  graph.addEdge(node2, node3, {});

  // Create a FullSgd instance
  const sgd = new eg.FullSgd(graph, () => 100);

  // Create a drawing
  const drawing = eg.DrawingHyperbolic2d.initialPlacement(graph);

  // Record initial positions
  const initialPositions = {};
  for (const u of graph.nodeIndices()) {
    initialPositions[u] = { x: drawing.x(u), y: drawing.y(u) };
  }

  // Apply SGD
  sgd.applyWithDrawingHyperbolic2d(drawing, 0.1);

  // Verify that positions have changed
  let positionsChanged = false;
  for (const u of graph.nodeIndices()) {
    if (
      drawing.x(u) !== initialPositions[u].x ||
      drawing.y(u) !== initialPositions[u].y
    ) {
      positionsChanged = true;
      break;
    }
  }
  assert(positionsChanged, "SGD should change node positions");

  // Verify that all coordinates are finite numbers
  for (const u of graph.nodeIndices()) {
    assert(
      Number.isFinite(drawing.x(u)),
      "X coordinate should be a finite number"
    );
    assert(
      Number.isFinite(drawing.y(u)),
      "Y coordinate should be a finite number"
    );
  }

  // Verify that all nodes are within the Poincaré disc (distance from origin <= 1)
  for (const u of graph.nodeIndices()) {
    const distance = Math.sqrt(
      drawing.x(u) * drawing.x(u) + drawing.y(u) * drawing.y(u)
    );
    assert(
      distance < 1.0001, // Allow for small floating-point errors
      "Node should be within the Poincaré disc"
    );
  }
};

/**
 * Test applying SGD to Spherical 2D drawings
 */
exports.testFullSgdWithSpherical2d = function () {
  // Create a simple graph
  const graph = new eg.Graph();
  const node1 = graph.addNode({});
  const node2 = graph.addNode({});
  const node3 = graph.addNode({});
  graph.addEdge(node1, node2, {});
  graph.addEdge(node2, node3, {});

  // Create a FullSgd instance
  const sgd = new eg.FullSgd(graph, () => 100);

  // Create a drawing
  const drawing = eg.DrawingSpherical2d.initialPlacement(graph);

  // Record initial positions
  const initialPositions = {};
  for (const u of graph.nodeIndices()) {
    initialPositions[u] = {
      lon: drawing.lon(u),
      lat: drawing.lat(u),
    };
  }

  // Apply SGD
  sgd.applyWithDrawingSpherical2d(drawing, 0.1);

  // Verify that positions have changed
  let positionsChanged = false;
  for (const u of graph.nodeIndices()) {
    if (
      drawing.lon(u) !== initialPositions[u].lon ||
      drawing.lat(u) !== initialPositions[u].lat
    ) {
      positionsChanged = true;
      break;
    }
  }
  assert(positionsChanged, "SGD should change node positions");

  // Verify that all coordinates are finite numbers
  for (const u of graph.nodeIndices()) {
    assert(
      Number.isFinite(drawing.lon(u)),
      "Longitude should be a finite number"
    );
    assert(
      Number.isFinite(drawing.lat(u)),
      "Latitude should be a finite number"
    );
  }

  // Verify that latitude is within valid range (-π/2 to π/2)
  for (const u of graph.nodeIndices()) {
    assert(
      drawing.lat(u) >= -Math.PI / 2 && drawing.lat(u) <= Math.PI / 2,
      "Latitude should be within valid range"
    );
  }
};

/**
 * Test applying SGD to Torus 2D drawings
 */
exports.testFullSgdWithTorus2d = function () {
  // Create a simple graph
  const graph = new eg.Graph();
  const node1 = graph.addNode({});
  const node2 = graph.addNode({});
  const node3 = graph.addNode({});
  graph.addEdge(node1, node2, {});
  graph.addEdge(node2, node3, {});

  // Create a FullSgd instance
  const sgd = new eg.FullSgd(graph, () => 100);

  // Create a drawing
  const drawing = eg.DrawingTorus2d.initialPlacement(graph);

  // Record initial positions
  const initialPositions = {};
  for (const u of graph.nodeIndices()) {
    initialPositions[u] = { x: drawing.x(u), y: drawing.y(u) };
  }

  // Apply SGD
  sgd.applyWithDrawingTorus2d(drawing, 0.1);

  // Verify that positions have changed
  let positionsChanged = false;
  for (const u of graph.nodeIndices()) {
    if (
      drawing.x(u) !== initialPositions[u].x ||
      drawing.y(u) !== initialPositions[u].y
    ) {
      positionsChanged = true;
      break;
    }
  }
  assert(positionsChanged, "SGD should change node positions");

  // Verify that all coordinates are finite numbers
  for (const u of graph.nodeIndices()) {
    assert(
      Number.isFinite(drawing.x(u)),
      "X coordinate should be a finite number"
    );
    assert(
      Number.isFinite(drawing.y(u)),
      "Y coordinate should be a finite number"
    );
  }

  // Verify that coordinates are within the torus range (0 to 1)
  for (const u of graph.nodeIndices()) {
    assert(
      drawing.x(u) >= 0 && drawing.x(u) <= 1,
      "X coordinate should be within torus range"
    );
    assert(
      drawing.y(u) >= 0 && drawing.y(u) <= 1,
      "Y coordinate should be within torus range"
    );
  }
};

/**
 * Test applying SGD to n-dimensional Euclidean drawings
 */
exports.testFullSgdWithEuclidean = function () {
  // Create a simple graph
  const graph = new eg.Graph();
  const node1 = graph.addNode({});
  const node2 = graph.addNode({});
  const node3 = graph.addNode({});
  graph.addEdge(node1, node2, {});
  graph.addEdge(node2, node3, {});

  // Create a FullSgd instance
  const sgd = new eg.FullSgd(graph, () => 100);

  // Create a 3D drawing using ClassicalMds
  const mds = new eg.ClassicalMds(graph, () => 1.0);
  const drawing = mds.run(3);

  // Record initial positions
  const initialPositions = {};
  for (const u of graph.nodeIndices()) {
    initialPositions[u] = {
      x: drawing.get(u, 0),
      y: drawing.get(u, 1),
      z: drawing.get(u, 2),
    };
  }

  // Apply SGD
  sgd.applyWithDrawingEuclidean(drawing, 0.1);

  // Verify that positions have changed
  let positionsChanged = false;
  for (const u of graph.nodeIndices()) {
    if (
      drawing.get(u, 0) !== initialPositions[u].x ||
      drawing.get(u, 1) !== initialPositions[u].y ||
      drawing.get(u, 2) !== initialPositions[u].z
    ) {
      positionsChanged = true;
      break;
    }
  }
  assert(positionsChanged, "SGD should change node positions");

  // Verify that all coordinates are finite numbers
  for (const u of graph.nodeIndices()) {
    for (let d = 0; d < 3; d++) {
      assert(
        Number.isFinite(drawing.get(u, d)),
        `Coordinate at dimension ${d} should be a finite number`
      );
    }
  }
};

/**
 * Test updating distance function
 */
exports.testFullSgdUpdateDistance = function () {
  // Create a simple graph
  const graph = new eg.Graph();
  const node1 = graph.addNode({});
  const node2 = graph.addNode({});
  graph.addEdge(node1, node2, {});

  // Create a FullSgd instance
  const sgd = new eg.FullSgd(graph, () => 100);

  // Create a drawing
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Apply SGD with default distance function
  sgd.applyWithDrawingEuclidean2d(drawing, 0.1);
  const positionsAfterDefault = {};
  for (const u of graph.nodeIndices()) {
    positionsAfterDefault[u] = { x: drawing.x(u), y: drawing.y(u) };
  }

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
  // Create a simple graph
  const graph = new eg.Graph();
  const node1 = graph.addNode({});
  const node2 = graph.addNode({});
  graph.addEdge(node1, node2, {});

  // Create a FullSgd instance
  const sgd = new eg.FullSgd(graph, () => 100);

  // Create a drawing
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Apply SGD with default weight function
  sgd.applyWithDrawingEuclidean2d(drawing, 0.1);
  const positionsAfterDefault = {};
  for (const u of graph.nodeIndices()) {
    positionsAfterDefault[u] = { x: drawing.x(u), y: drawing.y(u) };
  }

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
  // Create a simple graph
  const graph = new eg.Graph();
  const node1 = graph.addNode({});
  const node2 = graph.addNode({});
  const node3 = graph.addNode({});
  graph.addEdge(node1, node2, {});
  graph.addEdge(node2, node3, {});

  // Create a FullSgd instance
  const sgd = new eg.FullSgd(graph, () => 100);

  // Create a drawing
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Create an RNG
  const rng = new eg.Rng();

  // Apply SGD
  sgd.shuffle(rng);
  sgd.applyWithDrawingEuclidean2d(drawing, 0.1);
  for (const u of graph.nodeIndices()) {
    assert(Number.isFinite(drawing.x(u)));
    assert(Number.isFinite(drawing.y(u)));
  }

  // Test that the same seed produces the same shuffle result
  const rng1 = eg.Rng.seedFrom(42n);
  const rng2 = eg.Rng.seedFrom(42n);

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
  // Create a simple graph
  const graph = new eg.Graph();
  const nodes = [];
  for (let i = 0; i < 10; i++) {
    nodes.push(graph.addNode({ id: i }));
  }

  // Add some edges to create a connected graph
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
  const rng = eg.Rng.seedFrom(123n);

  // Create a FullSgd instance
  const sgd = new eg.FullSgd(graph, () => 100);

  // Run a complete layout process
  const scheduler = sgd.scheduler(10, 0.1);
  scheduler.run((eta) => {
    sgd.shuffle(rng);
    sgd.applyWithDrawingEuclidean2d(drawing, eta);
  });

  // Verify that all coordinates are finite numbers
  for (const u of graph.nodeIndices()) {
    assert(
      Number.isFinite(drawing.x(u)),
      "X coordinate should be a finite number"
    );
    assert(
      Number.isFinite(drawing.y(u)),
      "Y coordinate should be a finite number"
    );
  }

  // Verify that connected nodes are positioned relatively close to each other
  // by checking that the average distance between connected nodes is less than
  // the average distance between all node pairs
  let connectedPairsCount = 0;
  let connectedPairsDistance = 0;
  let allPairsCount = 0;
  let allPairsDistance = 0;

  // Calculate average distance between connected nodes
  for (const e of graph.edgeIndices()) {
    // Get the endpoints of the edge
    const endpoints = graph.edgeEndpoints(e);
    const u = endpoints[0];
    const v = endpoints[1];

    const dx = drawing.x(u) - drawing.x(v);
    const dy = drawing.y(u) - drawing.y(v);
    const distance = Math.sqrt(dx * dx + dy * dy);
    connectedPairsDistance += distance;
    connectedPairsCount++;
  }

  // Calculate average distance between all node pairs
  const nodeIndices = Array.from(graph.nodeIndices());
  for (let i = 0; i < nodeIndices.length; i++) {
    for (let j = i + 1; j < nodeIndices.length; j++) {
      const u = nodeIndices[i];
      const v = nodeIndices[j];
      const dx = drawing.x(u) - drawing.x(v);
      const dy = drawing.y(u) - drawing.y(v);
      const distance = Math.sqrt(dx * dx + dy * dy);
      allPairsDistance += distance;
      allPairsCount++;
    }
  }

  const avgConnectedDistance = connectedPairsDistance / connectedPairsCount;
  const avgAllDistance = allPairsDistance / allPairsCount;

  assert(
    avgConnectedDistance < avgAllDistance,
    "Connected nodes should be positioned closer to each other than the average distance between all nodes"
  );
};
