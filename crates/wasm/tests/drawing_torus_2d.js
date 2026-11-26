const assert = require("assert");
const eg = require("wasm-bindgen-test");
const geometryTests = require("./util/drawing_torus_2d_geometry");

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

exports.testEdgeSegments = geometryTests.testEdgeSegments;
exports.testTorusWrapping = geometryTests.testTorusWrapping;
exports.testCoordinateValidation = geometryTests.testCoordinateValidation;

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
