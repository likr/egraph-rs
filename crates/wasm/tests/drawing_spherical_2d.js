const assert = require("assert");
const eg = require("wasm-bindgen-test");

/**
 * Test basic instantiation of DrawingSpherical2d class
 */
exports.testDrawingSpherical2dConstructor = function () {
  // Create a simple graph for the drawing
  const graph = new eg.Graph();
  const node1 = graph.addNode({ id: 1 });
  const node2 = graph.addNode({ id: 2 });
  graph.addEdge(node1, node2, {});

  // Create a drawing with initial placement
  const drawing = eg.DrawingSpherical2d.initialPlacement(graph);

  // Verify that the DrawingSpherical2d instance exists
  assert(
    drawing instanceof eg.DrawingSpherical2d,
    "Should create an instance of DrawingSpherical2d"
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
 * Test node coordinate operations (get/set longitude,latitude)
 */
exports.testNodeCoordinates = function () {
  // Create a simple graph for the drawing
  const graph = new eg.Graph();
  const node1 = graph.addNode({ id: 1 });
  const node2 = graph.addNode({ id: 2 });
  graph.addEdge(node1, node2, {});

  // Create a drawing with initial placement
  const drawing = eg.DrawingSpherical2d.initialPlacement(graph);

  // Verify initial coordinates are finite numbers
  assert(
    Number.isFinite(drawing.lon(node1)),
    "Initial longitude coordinate should be a finite number"
  );
  assert(
    Number.isFinite(drawing.lat(node1)),
    "Initial latitude coordinate should be a finite number"
  );
  assert(
    Number.isFinite(drawing.lon(node2)),
    "Initial longitude coordinate should be a finite number"
  );
  assert(
    Number.isFinite(drawing.lat(node2)),
    "Initial latitude coordinate should be a finite number"
  );

  // Test setting coordinates
  const newLon1 = 0.5; // Longitude in radians
  const newLat1 = 0.25; // Latitude in radians (0.25 = 1/4 can be represented exactly in binary)
  drawing.setLon(node1, newLon1);
  drawing.setLat(node1, newLat1);

  // Verify coordinates were set correctly
  assert.strictEqual(
    drawing.lon(node1),
    newLon1,
    "Longitude coordinate should be updated"
  );
  assert.strictEqual(
    drawing.lat(node1),
    newLat1,
    "Latitude coordinate should be updated"
  );

  // Test setting coordinates for another node
  const newLon2 = -0.5; // Longitude in radians
  const newLat2 = -0.25; // Latitude in radians (0.25 = 1/4 can be represented exactly in binary)
  drawing.setLon(node2, newLon2);
  drawing.setLat(node2, newLat2);

  // Verify coordinates were set correctly
  assert.strictEqual(
    drawing.lon(node2),
    newLon2,
    "Longitude coordinate should be updated"
  );
  assert.strictEqual(
    drawing.lat(node2),
    newLat2,
    "Latitude coordinate should be updated"
  );

  // Test getting coordinates for non-existent node
  assert.strictEqual(
    drawing.lon(999),
    undefined,
    "Longitude coordinate for non-existent node should be undefined"
  );
  assert.strictEqual(
    drawing.lat(999),
    undefined,
    "Latitude coordinate for non-existent node should be undefined"
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
  const drawing = eg.DrawingSpherical2d.initialPlacement(graph);

  // Verify that the drawing has the correct number of nodes
  assert.strictEqual(
    drawing.len(),
    graph.nodeCount(),
    "Drawing should have the same number of nodes as the graph"
  );

  // Verify all nodes have coordinates
  for (const nodeIndex of graph.nodeIndices()) {
    assert(
      Number.isFinite(drawing.lon(nodeIndex)),
      `Node ${nodeIndex} should have a valid longitude coordinate`
    );
    assert(
      Number.isFinite(drawing.lat(nodeIndex)),
      `Node ${nodeIndex} should have a valid latitude coordinate`
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
