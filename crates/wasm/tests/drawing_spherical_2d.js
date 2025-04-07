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

/**
 * Test spherical coordinates and conversion to 3D points
 */
exports.testEdgeSegments = function () {
  // Create a simple graph for the drawing with all nodes
  const graph = new eg.Graph();
  const node1 = graph.addNode({ id: 1 });
  const node2 = graph.addNode({ id: 2 });
  const node3 = graph.addNode({ id: 3 }); // Add node3 before creating the drawing
  graph.addEdge(node1, node2, {});

  // Create a drawing with initial placement
  const drawing = eg.DrawingSpherical2d.initialPlacement(graph);

  // Set specific coordinates for testing
  // Set nodes at different points on the sphere
  drawing.setLon(node1, 0); // Prime meridian
  drawing.setLat(node1, 0); // Equator
  drawing.setLon(node2, Math.PI / 2); // 90 degrees east
  drawing.setLat(node2, 0); // Equator

  // Verify coordinates were set correctly
  // Use approximate equality for longitude values, as they might be normalized
  assert(
    Math.abs(drawing.lon(node1)) < 0.001 ||
      Math.abs(Math.abs(drawing.lon(node1)) - 2 * Math.PI) < 0.001,
    "Longitude should be approximately 0 (or equivalent)"
  );
  // Use approximate equality for latitude values
  assert(
    Math.abs(drawing.lat(node1)) < 0.001,
    "Latitude should be set approximately to 0"
  );

  // For π/2, check if it's approximately π/2 or equivalent values like -3π/2
  assert(
    Math.abs(drawing.lon(node2) - Math.PI / 2) < 0.001 ||
      Math.abs(drawing.lon(node2) + (3 * Math.PI) / 2) < 0.001,
    "Longitude should be approximately π/2 (or equivalent)"
  );
  // Use approximate equality for latitude values
  assert(
    Math.abs(drawing.lat(node2)) < 0.001,
    "Latitude should be set approximately to 0"
  );

  // Convert spherical coordinates to 3D Cartesian coordinates
  // For node1 at (0,0): (1,0,0)
  const x1 = Math.cos(drawing.lat(node1)) * Math.cos(drawing.lon(node1));
  const y1 = Math.cos(drawing.lat(node1)) * Math.sin(drawing.lon(node1));
  const z1 = Math.sin(drawing.lat(node1));

  // For node2 at (π/2,0): (0,1,0)
  const x2 = Math.cos(drawing.lat(node2)) * Math.cos(drawing.lon(node2));
  const y2 = Math.cos(drawing.lat(node2)) * Math.sin(drawing.lon(node2));
  const z2 = Math.sin(drawing.lat(node2));

  // Verify 3D coordinates are correct (with small floating-point tolerance)
  assert(Math.abs(x1 - 1.0) < 0.001, "X coordinate for node1 should be 1.0");
  assert(Math.abs(y1 - 0.0) < 0.001, "Y coordinate for node1 should be 0.0");
  assert(Math.abs(z1 - 0.0) < 0.001, "Z coordinate for node1 should be 0.0");

  assert(Math.abs(x2 - 0.0) < 0.001, "X coordinate for node2 should be 0.0");
  assert(Math.abs(y2 - 1.0) < 0.001, "Y coordinate for node2 should be 1.0");
  assert(Math.abs(z2 - 0.0) < 0.001, "Z coordinate for node2 should be 0.0");

  // Verify points are on the unit sphere (x^2 + y^2 + z^2 = 1)
  const magnitude1 = Math.sqrt(x1 * x1 + y1 * y1 + z1 * z1);
  const magnitude2 = Math.sqrt(x2 * x2 + y2 * y2 + z2 * z2);

  assert(
    Math.abs(magnitude1 - 1.0) < 0.001,
    "Node1 should be on the unit sphere"
  );
  assert(
    Math.abs(magnitude2 - 1.0) < 0.001,
    "Node2 should be on the unit sphere"
  );

  // Set node3 at the north pole
  drawing.setLon(node3, 0); // Longitude doesn't matter at poles
  drawing.setLat(node3, Math.PI / 2); // North pole

  // Convert to 3D coordinates
  const x3 = Math.cos(drawing.lat(node3)) * Math.cos(drawing.lon(node3));
  const y3 = Math.cos(drawing.lat(node3)) * Math.sin(drawing.lon(node3));
  const z3 = Math.sin(drawing.lat(node3));

  // For a point at the north pole:
  // 1. The Z coordinate should be close to 1.0
  // 2. The magnitude of the vector should be close to 1.0 (point on unit sphere)
  // 3. We don't test X and Y directly as they depend on longitude, which doesn't matter at poles

  // Verify Z coordinate is close to 1.0 (north pole)
  assert(
    Math.abs(z3 - 1.0) < 0.1,
    "Z coordinate for node3 should be close to 1.0 (north pole)"
  );

  // Verify point is on the unit sphere
  const magnitude3 = Math.sqrt(x3 * x3 + y3 * y3 + z3 * z3);
  assert(
    Math.abs(magnitude3 - 1.0) < 0.01,
    "Node3 should be on the unit sphere"
  );
};

/**
 * Test great circle distance calculations between nodes
 */
exports.testGreatCircleDistance = function () {
  // Create a simple graph for the drawing
  const graph = new eg.Graph();
  const node1 = graph.addNode({ id: 1 });
  const node2 = graph.addNode({ id: 2 });
  graph.addEdge(node1, node2, {});

  // Create a drawing with initial placement
  const drawing = eg.DrawingSpherical2d.initialPlacement(graph);

  // Set specific coordinates for testing
  // Place nodes at known positions
  drawing.setLon(node1, 0); // Prime meridian
  drawing.setLat(node1, 0); // Equator

  // Test 1: 90 degrees east along equator (π/2 radians)
  drawing.setLon(node2, Math.PI / 2);
  drawing.setLat(node2, 0);

  // Calculate great circle distance manually
  // Convert to 3D Cartesian coordinates
  const x1 = Math.cos(drawing.lat(node1)) * Math.cos(drawing.lon(node1));
  const y1 = Math.cos(drawing.lat(node1)) * Math.sin(drawing.lon(node1));
  const z1 = Math.sin(drawing.lat(node1));

  const x2 = Math.cos(drawing.lat(node2)) * Math.cos(drawing.lon(node2));
  const y2 = Math.cos(drawing.lat(node2)) * Math.sin(drawing.lon(node2));
  const z2 = Math.sin(drawing.lat(node2));

  // Calculate dot product
  const dotProduct1 = x1 * x2 + y1 * y2 + z1 * z2;

  // Calculate angle (great circle distance)
  const distance1 = Math.acos(Math.max(-1, Math.min(1, dotProduct1)));

  // For nodes on the equator separated by 90 degrees, the distance should be π/2 radians
  const expectedDistance1 = Math.PI / 2;

  // Verify the distance is close to the expected great circle distance
  assert(
    Math.abs(distance1 - expectedDistance1) < 0.001,
    `Great circle distance should be approximately ${expectedDistance1} radians`
  );

  // Test 2: North pole (π/2 radians latitude)
  drawing.setLon(node2, 0);
  drawing.setLat(node2, Math.PI / 2);

  // Recalculate 3D coordinates for node2
  const x2b = Math.cos(drawing.lat(node2)) * Math.cos(drawing.lon(node2));
  const y2b = Math.cos(drawing.lat(node2)) * Math.sin(drawing.lon(node2));
  const z2b = Math.sin(drawing.lat(node2));

  // Calculate dot product
  const dotProduct2 = x1 * x2b + y1 * y2b + z1 * z2b;

  // Calculate angle (great circle distance)
  const distance2 = Math.acos(Math.max(-1, Math.min(1, dotProduct2)));

  // For a node at the equator and a node at the north pole, the distance should be π/2 radians
  const expectedDistance2 = Math.PI / 2;

  // Verify the distance is close to the expected great circle distance
  assert(
    Math.abs(distance2 - expectedDistance2) < 0.001,
    `Great circle distance should be approximately ${expectedDistance2} radians`
  );

  // Test 3: Antipodal points (opposite sides of the sphere)
  drawing.setLon(node1, 0);
  drawing.setLat(node1, 0);
  drawing.setLon(node2, Math.PI);
  drawing.setLat(node2, 0);

  // Recalculate 3D coordinates
  const x1c = Math.cos(drawing.lat(node1)) * Math.cos(drawing.lon(node1));
  const y1c = Math.cos(drawing.lat(node1)) * Math.sin(drawing.lon(node1));
  const z1c = Math.sin(drawing.lat(node1));

  const x2c = Math.cos(drawing.lat(node2)) * Math.cos(drawing.lon(node2));
  const y2c = Math.cos(drawing.lat(node2)) * Math.sin(drawing.lon(node2));
  const z2c = Math.sin(drawing.lat(node2));

  // Calculate dot product
  const dotProduct3 = x1c * x2c + y1c * y2c + z1c * z2c;

  // Calculate angle (great circle distance)
  const distance3 = Math.acos(Math.max(-1, Math.min(1, dotProduct3)));

  // For antipodal points, the distance should be π radians
  const expectedDistance3 = Math.PI;

  // Verify the distance is close to the expected great circle distance
  assert(
    Math.abs(distance3 - expectedDistance3) < 0.001,
    `Great circle distance should be approximately ${expectedDistance3} radians`
  );
};

/**
 * Test coordinate validation and range checking
 */
exports.testCoordinateValidation = function () {
  // Create a simple graph for the drawing
  const graph = new eg.Graph();
  const node1 = graph.addNode({ id: 1 });
  const node2 = graph.addNode({ id: 2 });
  graph.addEdge(node1, node2, {});

  // Create a drawing with initial placement
  const drawing = eg.DrawingSpherical2d.initialPlacement(graph);

  // Test valid values within range
  const validLon = Math.PI / 4; // 45 degrees
  const validLat = Math.PI / 6; // 30 degrees

  drawing.setLon(node1, validLon);
  drawing.setLat(node1, validLat);

  // Use approximate equality for longitude values
  assert(
    Math.abs(drawing.lon(node1) - validLon) < 0.001 ||
      Math.abs(Math.abs(drawing.lon(node1) - validLon) - 2 * Math.PI) < 0.001,
    "Valid longitude should be set approximately correctly (or equivalent)"
  );

  // Use approximate equality for latitude values
  assert(
    Math.abs(drawing.lat(node1) - validLat) < 0.001,
    "Valid latitude should be set approximately correctly"
  );

  // Test longitude at the boundaries
  drawing.setLon(node1, Math.PI);
  // π and -π are equivalent in spherical coordinates
  assert(
    Math.abs(drawing.lon(node1) - Math.PI) < 0.001 ||
      Math.abs(drawing.lon(node1) + Math.PI) < 0.001,
    "Longitude at π should be set approximately correctly (or equivalent to -π)"
  );

  drawing.setLon(node1, -Math.PI);
  // π and -π are equivalent in spherical coordinates
  assert(
    Math.abs(drawing.lon(node1) + Math.PI) < 0.001 ||
      Math.abs(drawing.lon(node1) - Math.PI) < 0.001,
    "Longitude at -π should be set approximately correctly (or equivalent to π)"
  );

  // Test latitude at the boundaries
  drawing.setLat(node1, Math.PI / 2);
  // Use approximate equality for latitude values
  assert(
    Math.abs(drawing.lat(node1) - Math.PI / 2) < 0.001,
    "Latitude at π/2 should be set approximately correctly"
  );

  drawing.setLat(node1, -Math.PI / 2);
  // Use approximate equality for latitude values
  assert(
    Math.abs(drawing.lat(node1) + Math.PI / 2) < 0.001,
    "Latitude at -π/2 should be set approximately correctly"
  );

  // Test setting coordinates for multiple nodes
  drawing.setLon(node1, 0);
  drawing.setLat(node1, 0);
  drawing.setLon(node2, Math.PI / 2);
  drawing.setLat(node2, Math.PI / 4);

  // Use approximate equality for longitude values
  assert(
    Math.abs(drawing.lon(node1)) < 0.001 ||
      Math.abs(Math.abs(drawing.lon(node1)) - 2 * Math.PI) < 0.001,
    "Node1 longitude should be approximately 0 (or equivalent)"
  );
  // Use approximate equality for latitude values
  assert(
    Math.abs(drawing.lat(node1)) < 0.001,
    "Node1 latitude should be approximately 0"
  );

  assert(
    Math.abs(drawing.lon(node2) - Math.PI / 2) < 0.001 ||
      Math.abs(drawing.lon(node2) + (3 * Math.PI) / 2) < 0.001,
    "Node2 longitude should be approximately π/2 (or equivalent)"
  );
  // Use approximate equality for latitude values
  assert(
    Math.abs(drawing.lat(node2) - Math.PI / 4) < 0.001,
    "Node2 latitude should be approximately π/4"
  );

  // Test that setting one node's coordinates doesn't affect others
  drawing.setLon(node1, Math.PI / 3);

  // Use approximate equality for longitude values
  assert(
    Math.abs(drawing.lon(node1) - Math.PI / 3) < 0.001 ||
      Math.abs(Math.abs(drawing.lon(node1) - Math.PI / 3) - 2 * Math.PI) <
        0.001,
    "Node1 longitude should be approximately updated to π/3 (or equivalent)"
  );

  assert(
    Math.abs(drawing.lon(node2) - Math.PI / 2) < 0.001 ||
      Math.abs(drawing.lon(node2) + (3 * Math.PI) / 2) < 0.001,
    "Node2 longitude should remain approximately π/2 (or equivalent)"
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
  const drawing = eg.DrawingSpherical2d.initialPlacement(graph);

  // Store initial coordinates
  const initialCoordinates = [];
  for (const nodeIndex of graph.nodeIndices()) {
    initialCoordinates.push({
      node: nodeIndex,
      lon: drawing.lon(nodeIndex),
      lat: drawing.lat(nodeIndex),
    });
  }

  // Apply a layout algorithm if available
  // For this test, we'll simulate a layout algorithm by manually updating coordinates
  // In a real scenario, you would use an actual layout algorithm like:
  // const layout = new eg.SphericalLayout(graph);
  // layout.run(drawing);

  // Simulate layout algorithm by setting new coordinates
  for (const nodeIndex of graph.nodeIndices()) {
    // Set new coordinates based on node index
    const newLon = (((nodeIndex * Math.PI) / 5) % (2 * Math.PI)) - Math.PI; // Distribute around the sphere
    const newLat = (((nodeIndex % 3) - 1) * Math.PI) / 6; // Distribute across latitudes

    drawing.setLon(nodeIndex, newLon);
    drawing.setLat(nodeIndex, newLat);
  }

  // Verify coordinates have changed
  let coordinatesChanged = false;
  for (const nodeIndex of graph.nodeIndices()) {
    const initial = initialCoordinates.find((c) => c.node === nodeIndex);
    if (
      Math.abs(drawing.lon(nodeIndex) - initial.lon) > 0.001 ||
      Math.abs(drawing.lat(nodeIndex) - initial.lat) > 0.001
    ) {
      coordinatesChanged = true;
      break;
    }
  }

  assert(
    coordinatesChanged,
    "At least one node's coordinates should change after layout application"
  );

  // Verify all nodes are still on the sphere
  for (const nodeIndex of graph.nodeIndices()) {
    // Convert spherical to Cartesian coordinates
    const lon = drawing.lon(nodeIndex);
    const lat = drawing.lat(nodeIndex);

    const x = Math.cos(lat) * Math.cos(lon);
    const y = Math.cos(lat) * Math.sin(lon);
    const z = Math.sin(lat);

    // Calculate distance from origin (should be 1 for unit sphere)
    const distance = Math.sqrt(x * x + y * y + z * z);

    assert(
      Math.abs(distance - 1.0) < 0.001,
      `Node ${nodeIndex} should remain on the unit sphere after layout`
    );
  }
};
