const assert = require("assert");
const eg = require("wasm-bindgen-test");

/**
 * Test edge segment representation on a torus surface
 */
function testEdgeSegments() {
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
}

/**
 * Test torus wrapping behavior (coordinates wrapping around)
 */
function testTorusWrapping() {
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
}

/**
 * Test coordinate validation and normalization
 */
function testCoordinateValidation() {
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
}

module.exports = {
  testEdgeSegments,
  testTorusWrapping,
  testCoordinateValidation,
};
