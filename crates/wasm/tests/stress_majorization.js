const assert = require("assert");
const eg = require("wasm-bindgen-test");

/**
 * Test basic instantiation of StressMajorization class
 */
exports.testStressMajorizationConstructor = function () {
  // Create a simple graph
  const graph = new eg.Graph();
  const node1 = graph.addNode({});
  const node2 = graph.addNode({});
  const edge = graph.addEdge(node1, node2, {});

  // Create a drawing
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

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
  const graph = new eg.Graph();
  const node1 = graph.addNode({});
  const node2 = graph.addNode({});
  const node3 = graph.addNode({});
  graph.addEdge(node1, node2, {});
  graph.addEdge(node2, node3, {});

  // Create a drawing with initial positions
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);
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
  const initialPositions = {};
  for (const u of graph.nodeIndices()) {
    initialPositions[u] = { x: drawing.x(u), y: drawing.y(u) };
  }

  // Apply a single iteration
  const stress = layout.apply(drawing);

  // Verify that the stress value is a finite number
  assert(Number.isFinite(stress), "Stress value should be a finite number");

  // Verify that at least one node position has changed
  let positionsChanged = false;
  for (const u of graph.nodeIndices()) {
    if (
      Math.abs(drawing.x(u) - initialPositions[u].x) > 1e-10 ||
      Math.abs(drawing.y(u) - initialPositions[u].y) > 1e-10
    ) {
      positionsChanged = true;
      break;
    }
  }
  assert(
    positionsChanged,
    "At least one node position should change after applying the algorithm"
  );

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
 * Test running the complete stress majorization algorithm
 */
exports.testStressMajorizationRun = function () {
  // Create a simple graph
  const graph = new eg.Graph();
  const node1 = graph.addNode({});
  const node2 = graph.addNode({});
  const node3 = graph.addNode({});
  const node4 = graph.addNode({});
  graph.addEdge(node1, node2, {});
  graph.addEdge(node2, node3, {});
  graph.addEdge(node3, node4, {});
  graph.addEdge(node4, node1, {});

  // Create a drawing with initial positions
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);
  // Place nodes in a line to create initial stress
  drawing.setX(0, 0.0);
  drawing.setY(0, 0.0);
  drawing.setX(1, 1.0);
  drawing.setY(1, 0.0);
  drawing.setX(2, 2.0);
  drawing.setY(2, 0.0);
  drawing.setX(3, 3.0);
  drawing.setY(3, 0.0);

  // Create a StressMajorization instance
  const layout = new eg.StressMajorization(graph, drawing, (e) => {
    return { distance: 1.0 };
  });

  // Record initial positions
  const initialPositions = {};
  for (const u of graph.nodeIndices()) {
    initialPositions[u] = { x: drawing.x(u), y: drawing.y(u) };
  }

  // Run the complete algorithm
  layout.run(drawing);

  // Verify that positions have changed
  let positionsChanged = false;
  for (const u of graph.nodeIndices()) {
    if (
      Math.abs(drawing.x(u) - initialPositions[u].x) > 1e-10 ||
      Math.abs(drawing.y(u) - initialPositions[u].y) > 1e-10
    ) {
      positionsChanged = true;
      break;
    }
  }
  assert(
    positionsChanged,
    "Node positions should change after running the algorithm"
  );

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
 * Test integration with other components and stress reduction
 */
exports.testStressMajorizationIntegration = function () {
  // Create a more complex graph
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

  // Create a drawing with initial positions
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Set specific positions instead of using random values
  // This is more reliable than using random values
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

  // Create a StressMajorization instance with custom distance function
  const layout = new eg.StressMajorization(graph, drawing, (e) => {
    const endpoints = graph.edgeEndpoints(e);
    const u = endpoints[0];
    const v = endpoints[1];
    // Use node IDs to determine distance
    const uId = graph.nodeWeight(u).id;
    const vId = graph.nodeWeight(v).id;
    return { distance: Math.abs(uId - vId) };
  });

  // Apply the layout algorithm 100 times instead of using run()
  // to avoid potential infinite loops
  for (let i = 0; i < 100; i++) {
    layout.apply(drawing);
  }

  // Calculate final stress
  const finalStress = eg.stress(graph, drawing);

  // Verify that stress has been reduced
  assert(
    finalStress < initialStress,
    "Stress should be reduced after running the algorithm"
  );

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
