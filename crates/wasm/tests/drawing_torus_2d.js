const assert = require("assert");
const eg = require("wasm-bindgen-test");

/**
 * Test basic instantiation of DrawingTorus2d class
 */
exports.testDrawingTorus2dConstructor = function () {
  // Create a simple graph for the drawing
  const graph = new eg.Graph();
  const node1 = graph.addNode({ id: 1 });
  const node2 = graph.addNode({ id: 2 });
  graph.addEdge(node1, node2, {});

  // Create a drawing with initial placement
  const drawing = eg.DrawingTorus2d.initialPlacement(graph);

  // Verify that the DrawingTorus2d instance exists
  assert(
    drawing instanceof eg.DrawingTorus2d,
    "Should create an instance of DrawingTorus2d"
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
  const drawing = eg.DrawingTorus2d.initialPlacement(graph);

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
  const newX1 = 0.5; // Within [0,1] range
  const newY1 = 0.25;
  drawing.setX(node1, newX1);
  drawing.setY(node1, newY1);

  // Verify coordinates were set correctly
  // Use approximate equality for coordinates as they might be normalized
  assert(
    Math.abs(drawing.x(node1) - newX1) < 0.001,
    "X coordinate should be updated"
  );
  assert(
    Math.abs(drawing.y(node1) - newY1) < 0.001,
    "Y coordinate should be updated"
  );

  // Test setting coordinates for another node
  const newX2 = 0.75;
  const newY2 = 0.75;
  drawing.setX(node2, newX2);
  drawing.setY(node2, newY2);

  // Verify coordinates were set correctly
  // Use approximate equality for coordinates as they might be normalized
  assert(
    Math.abs(drawing.x(node2) - newX2) < 0.001,
    "X coordinate should be updated"
  );
  assert(
    Math.abs(drawing.y(node2) - newY2) < 0.001,
    "Y coordinate should be updated"
  );

  // Test getting coordinates for non-existent node
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
  const drawing = eg.DrawingTorus2d.initialPlacement(graph);

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

  // Add a new node after creating the drawing
  const newNode = graph.addNode({ id: 5 });

  // Verify the drawing doesn't have the new node yet
  assert.strictEqual(
    drawing.x(newNode),
    undefined,
    "New node should not have coordinates in the drawing yet"
  );

  // Create a new drawing with the updated graph
  const updatedDrawing = eg.DrawingTorus2d.initialPlacement(graph);

  // Verify the new drawing includes the new node
  assert.strictEqual(
    updatedDrawing.len(),
    graph.nodeCount(),
    "Updated drawing should include the new node"
  );
  assert(
    Number.isFinite(updatedDrawing.x(newNode)),
    "New node should have a valid x coordinate in the updated drawing"
  );
};

/**
 * Test edge segment representation on a torus surface
 */
exports.testEdgeSegments = function () {
  // Create a simple graph for the drawing
  const graph = new eg.Graph();
  const node1 = graph.addNode({ id: 1 });
  const node2 = graph.addNode({ id: 2 });
  graph.addEdge(node1, node2, {});

  // Create a drawing with initial placement
  const drawing = eg.DrawingTorus2d.initialPlacement(graph);

  // Set specific coordinates for testing
  // Place nodes at positions that don't cross the boundary
  drawing.setX(node1, 0.25);
  drawing.setY(node1, 0.25);
  drawing.setX(node2, 0.75);
  drawing.setY(node2, 0.75);

  // Get edge segments
  const segments = drawing.edgeSegments(node1, node2);

  // Verify segments exist
  assert(segments, "Edge segments should exist");
  assert(
    segments.length > 0,
    "There should be at least one segment for the edge"
  );

  // For nodes that don't cross the boundary, there should be exactly one segment
  assert.strictEqual(
    segments.length,
    1,
    "There should be exactly one segment for nodes that don't cross the boundary"
  );

  // Verify the segment connects the two nodes
  const segment = segments[0];
  assert(Array.isArray(segment), "Segment should be an array");
  assert.strictEqual(segment.length, 2, "Segment should have two points");

  const p = segment[0];
  const q = segment[1];

  assert(Array.isArray(p), "First point should be an array");
  assert(Array.isArray(q), "Second point should be an array");
  assert.strictEqual(p.length, 2, "First point should have x,y coordinates");
  assert.strictEqual(q.length, 2, "Second point should have x,y coordinates");

  // Verify the segment endpoints match the node coordinates
  // Note: The segment might be in either direction
  const matchesDirectly =
    (Math.abs(p[0] - drawing.x(node1)) < 0.001 &&
      Math.abs(p[1] - drawing.y(node1)) < 0.001 &&
      Math.abs(q[0] - drawing.x(node2)) < 0.001 &&
      Math.abs(q[1] - drawing.y(node2)) < 0.001) ||
    (Math.abs(p[0] - drawing.x(node2)) < 0.001 &&
      Math.abs(p[1] - drawing.y(node2)) < 0.001 &&
      Math.abs(q[0] - drawing.x(node1)) < 0.001 &&
      Math.abs(q[1] - drawing.y(node1)) < 0.001);

  assert(
    matchesDirectly,
    "Segment endpoints should match the node coordinates"
  );

  // Now test with nodes that cross the boundary
  // Place nodes on opposite sides of the torus
  drawing.setX(node1, 0.05);
  drawing.setY(node1, 0.05);
  drawing.setX(node2, 0.95);
  drawing.setY(node2, 0.95);

  // Get edge segments
  const wrappingSegments = drawing.edgeSegments(node1, node2);

  // Verify segments exist
  assert(wrappingSegments, "Edge segments should exist for wrapping edge");
  assert(
    wrappingSegments.length > 0,
    "There should be at least one segment for the wrapping edge"
  );

  // For nodes that cross the boundary on a torus, there may be multiple segments
  // The exact number depends on the implementation, but we can verify they exist
  assert(
    wrappingSegments.length >= 1,
    "There should be at least one segment for the wrapping edge"
  );

  // Verify each segment is properly formatted
  for (const segment of wrappingSegments) {
    assert(Array.isArray(segment), "Each segment should be an array");
    assert.strictEqual(
      segment.length,
      2,
      "Each segment should have two points"
    );
    assert(
      Array.isArray(segment[0]),
      "First point of each segment should be an array"
    );
    assert(
      Array.isArray(segment[1]),
      "Second point of each segment should be an array"
    );
    assert.strictEqual(
      segment[0].length,
      2,
      "First point should have x,y coordinates"
    );
    assert.strictEqual(
      segment[1].length,
      2,
      "Second point should have x,y coordinates"
    );
  }
};

/**
 * Test torus wrapping behavior (coordinates wrapping around)
 */
exports.testTorusWrapping = function () {
  // Create a simple graph for the drawing
  const graph = new eg.Graph();
  const node1 = graph.addNode({ id: 1 });
  graph.addNode({ id: 2 });

  // Create a drawing with initial placement
  const drawing = eg.DrawingTorus2d.initialPlacement(graph);

  // Test wrapping behavior for x-coordinate
  // Set x-coordinate to a value > 1
  drawing.setX(node1, 1.25);
  // On a torus, 1.25 should wrap to 0.25
  assert.strictEqual(
    drawing.x(node1),
    0.25,
    "X-coordinate > 1 should wrap around to [0,1] range"
  );

  // Set x-coordinate to a negative value
  drawing.setX(node1, -0.25);
  // On a torus, -0.25 should wrap to 0.75
  assert.strictEqual(
    drawing.x(node1),
    0.75,
    "Negative x-coordinate should wrap around to [0,1] range"
  );

  // Test wrapping behavior for y-coordinate
  // Set y-coordinate to a value > 1
  drawing.setY(node1, 1.75);
  // On a torus, 1.75 should wrap to 0.75
  assert.strictEqual(
    drawing.y(node1),
    0.75,
    "Y-coordinate > 1 should wrap around to [0,1] range"
  );

  // Set y-coordinate to a negative value
  drawing.setY(node1, -0.5);
  // On a torus, -0.5 should wrap to 0.5
  assert.strictEqual(
    drawing.y(node1),
    0.5,
    "Negative y-coordinate should wrap around to [0,1] range"
  );

  // Test with multiple wraps
  drawing.setX(node1, 3.25);
  // 3.25 should wrap to 0.25 (3.25 % 1 = 0.25)
  assert.strictEqual(
    drawing.x(node1),
    0.25,
    "X-coordinate with multiple wraps should normalize correctly"
  );

  drawing.setY(node1, -2.75);
  // -2.75 should wrap to 0.25 (-2.75 % 1 = -0.75, then 1 - 0.75 = 0.25)
  assert.strictEqual(
    drawing.y(node1),
    0.25,
    "Negative y-coordinate with multiple wraps should normalize correctly"
  );
};

/**
 * Test coordinate validation and normalization
 */
exports.testCoordinateValidation = function () {
  // Create a simple graph for the drawing
  const graph = new eg.Graph();
  const node1 = graph.addNode({ id: 1 });
  const node2 = graph.addNode({ id: 2 });
  graph.addEdge(node1, node2, {});

  // Create a drawing with initial placement
  const drawing = eg.DrawingTorus2d.initialPlacement(graph);

  // Test valid values within range
  const validX = 0.5;
  const validY = 0.75;

  drawing.setX(node1, validX);
  drawing.setY(node1, validY);

  // Verify coordinates were set correctly
  // Use approximate equality for coordinates as they might be normalized
  assert(
    Math.abs(drawing.x(node1) - validX) < 0.001,
    "Valid x-coordinate should be set correctly"
  );
  assert(
    Math.abs(drawing.y(node1) - validY) < 0.001,
    "Valid y-coordinate should be set correctly"
  );

  // Test boundary values
  drawing.setX(node1, 0);
  drawing.setY(node1, 0);

  // Verify coordinates were set correctly
  assert.strictEqual(
    drawing.x(node1),
    0,
    "X-coordinate at lower boundary should be set correctly"
  );
  assert.strictEqual(
    drawing.y(node1),
    0,
    "Y-coordinate at lower boundary should be set correctly"
  );

  drawing.setX(node1, 1);
  drawing.setY(node1, 1);

  // Verify coordinates were set correctly
  // Note: On a torus, 1 might be normalized to 0 since they're equivalent
  const xResult = drawing.x(node1);
  const yResult = drawing.y(node1);
  assert(
    xResult === 1 || xResult === 0,
    "X-coordinate at upper boundary should be set correctly or normalized"
  );
  assert(
    yResult === 1 || yResult === 0,
    "Y-coordinate at upper boundary should be set correctly or normalized"
  );

  // Test setting coordinates for multiple nodes
  drawing.setX(node1, 0.125); // 1/8
  drawing.setY(node1, 0.25); // 1/4
  drawing.setX(node2, 0.375); // 3/8
  drawing.setY(node2, 0.5); // 1/2

  // Verify coordinates were set correctly
  // Use approximate equality for coordinates as they might be normalized
  assert(
    Math.abs(drawing.x(node1) - 0.125) < 0.001,
    "Node1 x-coordinate should be set correctly"
  );
  assert(
    Math.abs(drawing.y(node1) - 0.25) < 0.001,
    "Node1 y-coordinate should be set correctly"
  );
  assert(
    Math.abs(drawing.x(node2) - 0.375) < 0.001,
    "Node2 x-coordinate should be set correctly"
  );
  assert(
    Math.abs(drawing.y(node2) - 0.5) < 0.001,
    "Node2 y-coordinate should be set correctly"
  );

  // Test that setting one node's coordinates doesn't affect others
  drawing.setX(node1, 0.875); // 7/8

  // Verify only the intended node's coordinate was changed
  // Use approximate equality for coordinates as they might be normalized
  assert(
    Math.abs(drawing.x(node1) - 0.875) < 0.001,
    "Node1 x-coordinate should be updated"
  );
  assert(
    Math.abs(drawing.x(node2) - 0.375) < 0.001,
    "Node2 x-coordinate should remain unchanged"
  );
};

/**
 * Test integration with layout algorithms
 */
exports.testLayoutIntegration = function () {
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
  const drawing = eg.DrawingTorus2d.initialPlacement(graph);

  // Store initial coordinates
  const initialCoordinates = [];
  for (const nodeIndex of graph.nodeIndices()) {
    initialCoordinates.push({
      node: nodeIndex,
      x: drawing.x(nodeIndex),
      y: drawing.y(nodeIndex),
    });
  }

  // Apply a layout algorithm if available
  // For this test, we'll simulate a layout algorithm by manually updating coordinates
  // In a real scenario, you would use an actual layout algorithm like:
  // const layout = new eg.TorusLayout(graph);
  // layout.run(drawing);

  // Simulate layout algorithm by setting new coordinates
  for (const nodeIndex of graph.nodeIndices()) {
    // Set new coordinates based on node index, ensuring they stay within [0,1] range
    const newX = (nodeIndex * 0.2) % 1; // Distribute evenly in [0,1]
    const newY = (nodeIndex * 0.2 + 0.1) % 1; // Offset from x values

    drawing.setX(nodeIndex, newX);
    drawing.setY(nodeIndex, newY);
  }

  // Verify coordinates have changed
  let coordinatesChanged = false;
  for (const nodeIndex of graph.nodeIndices()) {
    const initial = initialCoordinates.find((c) => c.node === nodeIndex);
    if (
      Math.abs(drawing.x(nodeIndex) - initial.x) > 0.001 ||
      Math.abs(drawing.y(nodeIndex) - initial.y) > 0.001
    ) {
      coordinatesChanged = true;
      break;
    }
  }

  assert(
    coordinatesChanged,
    "At least one node's coordinates should change after layout application"
  );

  // Verify all nodes are still within the valid range [0,1]
  for (const nodeIndex of graph.nodeIndices()) {
    const x = drawing.x(nodeIndex);
    const y = drawing.y(nodeIndex);

    assert(
      x >= 0 && x <= 1,
      `Node ${nodeIndex} x-coordinate should be in range [0,1] after layout`
    );
    assert(
      y >= 0 && y <= 1,
      `Node ${nodeIndex} y-coordinate should be in range [0,1] after layout`
    );
  }
};
