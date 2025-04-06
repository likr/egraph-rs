const assert = require("assert");
const eg = require("wasm-bindgen-test");

/**
 * Test basic instantiation of DrawingEuclidean2d class
 */
exports.testDrawingEuclidean2dConstructor = function () {
  // Create a simple graph for the drawing
  const graph = new eg.Graph();
  const node1 = graph.addNode({ id: 1 });
  const node2 = graph.addNode({ id: 2 });
  graph.addEdge(node1, node2, {});

  // Create a drawing with initial placement
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Verify that the DrawingEuclidean2d instance exists
  assert(
    drawing instanceof eg.DrawingEuclidean2d,
    "Should create an instance of DrawingEuclidean2d"
  );

  // Verify initial state
  assert.strictEqual(
    drawing.len(),
    2,
    "Drawing should have the same number of nodes as the graph"
  );
  assert.strictEqual(drawing.is_empty(), false, "Drawing should not be empty");
};

/**
 * Test node coordinate operations (get/set x,y)
 */
exports.testNodeCoordinates = function () {
  // Create a simple graph for the drawing
  const graph = new eg.Graph();
  const node1 = graph.addNode({ id: 1 });
  const node2 = graph.addNode({ id: 2 });
  graph.addEdge(node1, node2, {});

  // Create a drawing with initial placement
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Verify initial coordinates are finite numbers
  assert(
    Number.isFinite(drawing.x(node1)),
    "Initial x coordinate should be a finite number"
  );
  assert(
    Number.isFinite(drawing.y(node1)),
    "Initial y coordinate should be a finite number"
  );
  assert(
    Number.isFinite(drawing.x(node2)),
    "Initial x coordinate should be a finite number"
  );
  assert(
    Number.isFinite(drawing.y(node2)),
    "Initial y coordinate should be a finite number"
  );

  // Test setting coordinates
  const newX1 = 10.5;
  const newY1 = 20.5;
  drawing.setX(node1, newX1);
  drawing.setY(node1, newY1);

  // Verify coordinates were set correctly
  assert.strictEqual(drawing.x(node1), newX1, "X coordinate should be updated");
  assert.strictEqual(drawing.y(node1), newY1, "Y coordinate should be updated");

  // Test setting coordinates for another node
  const newX2 = -5.5;
  const newY2 = -15.5;
  drawing.setX(node2, newX2);
  drawing.setY(node2, newY2);

  // Verify coordinates were set correctly
  assert.strictEqual(drawing.x(node2), newX2, "X coordinate should be updated");
  assert.strictEqual(drawing.y(node2), newY2, "Y coordinate should be updated");

  // Test getting coordinates for non-existent node
  // In the current implementation, x() and y() return undefined for non-existent nodes
  assert.strictEqual(
    drawing.x(999),
    undefined,
    "X coordinate for non-existent node should be undefined"
  );
  assert.strictEqual(
    drawing.y(999),
    undefined,
    "Y coordinate for non-existent node should be undefined"
  );
};

/**
 * Test drawing manipulation (centralize, clamp_region)
 */
exports.testDrawingManipulation = function () {
  // Create a simple graph for the drawing
  const graph = new eg.Graph();
  const node1 = graph.addNode({ id: 1 });
  const node2 = graph.addNode({ id: 2 });
  graph.addEdge(node1, node2, {});

  // Create a drawing with initial placement
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Set specific coordinates for testing
  drawing.setX(node1, 10);
  drawing.setY(node1, 20);
  drawing.setX(node2, 30);
  drawing.setY(node2, 40);

  // Test centralize
  drawing.centralize();

  // After centralization, the center of mass should be at (0,0)
  // For two nodes at (10,20) and (30,40), the center is (20,30)
  // After centralization, node1 should be at (-10,-10) and node2 at (10,10)
  const expectedX1 = -10;
  const expectedY1 = -10;
  const expectedX2 = 10;
  const expectedY2 = 10;

  // Allow for small floating-point differences
  assert(
    Math.abs(drawing.x(node1) - expectedX1) < 0.001,
    "Node1 X coordinate should be centralized"
  );
  assert(
    Math.abs(drawing.y(node1) - expectedY1) < 0.001,
    "Node1 Y coordinate should be centralized"
  );
  assert(
    Math.abs(drawing.x(node2) - expectedX2) < 0.001,
    "Node2 X coordinate should be centralized"
  );
  assert(
    Math.abs(drawing.y(node2) - expectedY2) < 0.001,
    "Node2 Y coordinate should be centralized"
  );

  // Test clamp_region
  // Set coordinates outside the clamping region
  drawing.setX(node1, -100);
  drawing.setY(node1, -200);
  drawing.setX(node2, 100);
  drawing.setY(node2, 200);

  // Clamp to region [-50, -50, 50, 50]
  drawing.clampRegion(-50, -50, 50, 50);

  // Verify coordinates are clamped
  assert.strictEqual(
    drawing.x(node1),
    -50,
    "X coordinate should be clamped to minimum"
  );
  assert.strictEqual(
    drawing.y(node1),
    -50,
    "Y coordinate should be clamped to minimum"
  );
  assert.strictEqual(
    drawing.x(node2),
    50,
    "X coordinate should be clamped to maximum"
  );
  assert.strictEqual(
    drawing.y(node2),
    50,
    "Y coordinate should be clamped to maximum"
  );
};

/**
 * Test edge segment representation
 */
exports.testEdgeSegments = function () {
  // Create a simple graph for the drawing
  const graph = new eg.Graph();
  const node1 = graph.addNode({ id: 1 });
  const node2 = graph.addNode({ id: 2 });
  graph.addEdge(node1, node2, {});

  // Create a drawing with initial placement
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Set specific coordinates for testing
  drawing.setX(node1, 0);
  drawing.setY(node1, 0);
  drawing.setX(node2, 10);
  drawing.setY(node2, 10);

  // Test edge segments
  const segments = drawing.edgeSegments(node1, node2);

  // Verify segments exist
  assert(segments, "Edge segments should exist");
  assert(
    segments.length > 0,
    "There should be at least one segment for the edge"
  );

  // For Euclidean 2D, there should be one straight line segment
  assert.strictEqual(
    segments.length,
    1,
    "There should be exactly one segment for a straight line in Euclidean 2D"
  );

  // Verify segment structure
  const segment = segments[0];
  assert(Array.isArray(segment), "Segment should be an array");
  assert.strictEqual(
    segment.length,
    2,
    "Segment should have two points (start and end)"
  );

  // Verify points structure
  const startPoint = segment[0];
  const endPoint = segment[1];
  assert(Array.isArray(startPoint), "Start point should be an array");
  assert(Array.isArray(endPoint), "End point should be an array");
  assert.strictEqual(
    startPoint.length,
    2,
    "Start point should have x,y coordinates"
  );
  assert.strictEqual(
    endPoint.length,
    2,
    "End point should have x,y coordinates"
  );

  // Verify coordinates match node positions
  assert.strictEqual(
    startPoint[0],
    drawing.x(node1),
    "Start point x should match node1 x"
  );
  assert.strictEqual(
    startPoint[1],
    drawing.y(node1),
    "Start point y should match node1 y"
  );
  assert.strictEqual(
    endPoint[0],
    drawing.x(node2),
    "End point x should match node2 x"
  );
  assert.strictEqual(
    endPoint[1],
    drawing.y(node2),
    "End point y should match node2 y"
  );

  // Test edge segments for non-existent edge
  // In the current implementation, edgeSegments() returns undefined for non-existent edges
  const nonExistentSegments = drawing.edgeSegments(node1, 999);
  assert.strictEqual(
    nonExistentSegments,
    undefined,
    "Edge segments for non-existent edge should be undefined"
  );
};

/**
 * Test integration with Graph class
 */
exports.testDrawingWithGraph = function () {
  // Create a more complex graph
  const graph = new eg.Graph();
  const nodes = [];
  for (let i = 0; i < 5; i++) {
    nodes.push(graph.addNode({ id: i }));
  }

  // Create a simple graph structure
  //    0
  //   / \
  //  1---2
  //  |   |
  //  3---4
  graph.addEdge(nodes[0], nodes[1], {});
  graph.addEdge(nodes[0], nodes[2], {});
  graph.addEdge(nodes[1], nodes[2], {});
  graph.addEdge(nodes[1], nodes[3], {});
  graph.addEdge(nodes[2], nodes[4], {});
  graph.addEdge(nodes[3], nodes[4], {});

  // Create a drawing with initial placement
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Verify that the drawing has the correct number of nodes
  assert.strictEqual(
    drawing.len(),
    graph.nodeCount(),
    "Drawing should have the same number of nodes as the graph"
  );

  // Verify all nodes have coordinates
  for (const nodeIndex of graph.nodeIndices()) {
    assert(
      Number.isFinite(drawing.x(nodeIndex)),
      `Node ${nodeIndex} should have a valid x coordinate`
    );
    assert(
      Number.isFinite(drawing.y(nodeIndex)),
      `Node ${nodeIndex} should have a valid y coordinate`
    );
  }

  // Test edge segments for all edges
  for (const edgeIndex of graph.edgeIndices()) {
    const endpoints = graph.edgeEndpoints(edgeIndex);
    const segments = drawing.edgeSegments(endpoints[0], endpoints[1]);
    assert(
      segments && segments.length > 0,
      `Edge ${edgeIndex} should have segments`
    );
  }

  // Test with a layout algorithm (if available)
  // Note: This is just a placeholder. In a real test, you would use an actual layout algorithm.
  // For example:
  // const layout = new eg.KamadaKawai(graph);
  // layout.run(drawing);
  //
  // Then verify that the drawing has been updated with new coordinates.
};
