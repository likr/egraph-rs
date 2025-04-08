const assert = require("assert");
const eg = require("wasm-bindgen-test");

/**
 * Helper function to verify the structure of bundled edges
 * @param {Map} bundledEdges - The result of the fdeb function
 * @param {number[]} edgeIndices - Array of edge indices to check
 */
function verifyBundledEdgesStructure(bundledEdges, edgeIndices) {
  // Verify that the result is a Map object
  assert(bundledEdges instanceof Map, "FDEB result should be a Map object");

  // Check if the result is empty
  if (bundledEdges.size === 0) {
    // Empty result is acceptable
    return;
  }

  // Verify that the result contains entries for all edges
  for (const edgeIndex of edgeIndices) {
    assert(
      bundledEdges.has(edgeIndex),
      `FDEB result should contain an entry for edge ${edgeIndex}`
    );
    assert(
      Array.isArray(bundledEdges.get(edgeIndex)),
      `Edge ${edgeIndex} should have an array of points`
    );
  }
}

/**
 * Helper function to verify the coordinates of points in bundled edges
 * @param {Map} bundledEdges - The result of the fdeb function
 * @param {number[]} edgeIndices - Array of edge indices to check
 */
function verifyPointCoordinates(bundledEdges, edgeIndices) {
  if (bundledEdges.size === 0) {
    return;
  }

  for (const edgeIndex of edgeIndices) {
    const points = bundledEdges.get(edgeIndex);
    for (const point of points) {
      assert(
        Array.isArray(point) && point.length === 2,
        "Each point should be an array with x and y coordinates"
      );
      assert(
        typeof point[0] === "number" && typeof point[1] === "number",
        "Coordinates should be numbers"
      );
      assert(
        Number.isFinite(point[0]) && Number.isFinite(point[1]),
        "Coordinates should be finite numbers"
      );
    }
  }
}

/**
 * Helper function to store node positions before calling fdeb
 * @param {Object} drawing - The drawing object
 * @param {number} nodeCount - Number of nodes
 * @returns {Array} Array of node positions
 */
function storeNodePositions(drawing, nodeCount) {
  const positions = [];
  for (let i = 0; i < nodeCount; i++) {
    positions.push({
      x: drawing.x(i),
      y: drawing.y(i),
    });
  }
  return positions;
}

/**
 * Test basic functionality of the FDEB algorithm
 */
exports.testFdebBasic = function () {
  // Create a simple graph with two crossing edges
  const graph = new eg.Graph();
  const node1 = graph.addNode({});
  const node2 = graph.addNode({});
  const node3 = graph.addNode({});
  const node4 = graph.addNode({});

  const edge1 = graph.addEdge(node1, node3, {});
  const edge2 = graph.addEdge(node2, node4, {});

  // Create a drawing with positions that create crossing edges
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Set positions to create crossing edges
  // node1 (0,0) --- node3 (1,1)
  //     \             /
  //      \           /
  //       \         /
  //        \       /
  //         \     /
  // node2 (0,1) --- node4 (1,0)
  drawing.setX(0, 0.0);
  drawing.setY(0, 0.0);
  drawing.setX(1, 0.0);
  drawing.setY(1, 1.0);
  drawing.setX(2, 1.0);
  drawing.setY(2, 1.0);
  drawing.setX(3, 1.0);
  drawing.setY(3, 0.0);

  // Apply FDEB algorithm
  const bundledEdges = eg.fdeb(graph, drawing);

  // Verify the structure and coordinates
  verifyBundledEdgesStructure(bundledEdges, [0, 1]);
  verifyPointCoordinates(bundledEdges, [0, 1]);
};

/**
 * Test FDEB with a more complex graph
 */
exports.testFdebWithComplexGraph = function () {
  // Create a more complex graph with multiple edges
  const graph = new eg.Graph();
  const nodes = [];

  // Create a grid of nodes
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      nodes.push(graph.addNode({ x: i, y: j }));
    }
  }

  // Add edges between nodes
  const edges = [];
  // Connect nodes in a grid pattern
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 3; j++) {
      // Horizontal edges
      edges.push(graph.addEdge(nodes[i * 4 + j], nodes[i * 4 + j + 1], {}));
      // Vertical edges
      edges.push(graph.addEdge(nodes[j * 4 + i], nodes[(j + 1) * 4 + i], {}));
    }
  }

  // Add some diagonal edges that will cross others
  edges.push(graph.addEdge(nodes[0], nodes[15], {})); // Diagonal from top-left to bottom-right
  edges.push(graph.addEdge(nodes[3], nodes[12], {})); // Diagonal from top-right to bottom-left
  edges.push(graph.addEdge(nodes[5], nodes[10], {})); // Another diagonal

  // Create a drawing with positions matching the grid
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Set positions based on the grid
  for (let i = 0; i < 16; i++) {
    const x = i % 4;
    const y = Math.floor(i / 4);
    drawing.setX(i, x);
    drawing.setY(i, y);
  }

  // Apply FDEB algorithm
  const bundledEdges = eg.fdeb(graph, drawing);

  // Verify the structure
  const edgeIndices = Array.from({ length: edges.length }, (_, i) => i);
  verifyBundledEdgesStructure(bundledEdges, edgeIndices);

  // Verify that the diagonal edges have more points than the grid edges
  // Diagonal edges are more likely to be bundled and have more complex paths
  const diagonalEdges = [
    edges[edges.length - 3],
    edges[edges.length - 2],
    edges[edges.length - 1],
  ];
  const gridEdges = edges.slice(0, edges.length - 3);

  // Count the total number of points for each edge type
  let diagonalPointCount = 0;
  let gridPointCount = 0;

  for (let i = 0; i < diagonalEdges.length; i++) {
    const edgeIndex = edges.length - 3 + i;
    diagonalPointCount += bundledEdges.get(edgeIndex).length;
  }

  // Sample a few grid edges for comparison
  const sampleGridEdges = gridEdges.slice(0, 3);
  for (let i = 0; i < sampleGridEdges.length; i++) {
    gridPointCount += bundledEdges.get(i).length;
  }

  // Calculate average points per edge
  const avgDiagonalPoints = diagonalPointCount / diagonalEdges.length;
  const avgGridPoints = gridPointCount / sampleGridEdges.length;

  // Diagonal edges should have at least as many points as grid edges
  // This is a heuristic test, not a strict requirement
  assert(
    avgDiagonalPoints >= avgGridPoints,
    "Diagonal edges should have at least as many points as grid edges on average"
  );
};

/**
 * Test the structure of the FDEB result
 */
exports.testFdebResultStructure = function () {
  // Create a simple graph
  const graph = new eg.Graph();
  const node1 = graph.addNode({});
  const node2 = graph.addNode({});
  const node3 = graph.addNode({});

  const edge1 = graph.addEdge(node1, node2, {});
  const edge2 = graph.addEdge(node2, node3, {});
  const edge3 = graph.addEdge(node3, node1, {});

  // Create a drawing
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Set positions to form a triangle
  drawing.setX(0, 0.0);
  drawing.setY(0, 0.0);
  drawing.setX(1, 1.0);
  drawing.setY(1, 0.0);
  drawing.setX(2, 0.5);
  drawing.setY(2, 1.0);

  // Store node positions before calling fdeb
  const nodePositions = storeNodePositions(drawing, 3);

  // Apply FDEB algorithm
  const bundledEdges = eg.fdeb(graph, drawing);

  // Verify the structure
  verifyBundledEdgesStructure(bundledEdges, [0, 1, 2]);
  verifyPointCoordinates(bundledEdges, [0, 1, 2]);

  // Verify that each edge has at least 2 points (start and end)
  for (let i = 0; i < 3; i++) {
    const points = bundledEdges.get(i);
    assert(points.length >= 2, "Each edge should have at least 2 points");
  }

  // Verify that endpoints are near the source and target nodes
  for (let i = 0; i < 3; i++) {
    const points = bundledEdges.get(i);
    const endpoints = graph.edgeEndpoints(i);
    const sourceNode = endpoints[0];
    const targetNode = endpoints[1];

    // Use stored node positions
    const sourceX = nodePositions[sourceNode].x;
    const sourceY = nodePositions[sourceNode].y;
    const targetX = nodePositions[targetNode].x;
    const targetY = nodePositions[targetNode].y;

    const firstPoint = points[0];
    const lastPoint = points[points.length - 1];

    // Calculate distances from endpoints to source and target nodes
    const distToSource = Math.sqrt(
      Math.pow(firstPoint[0] - sourceX, 2) +
        Math.pow(firstPoint[1] - sourceY, 2)
    );

    const distToTarget = Math.sqrt(
      Math.pow(lastPoint[0] - targetX, 2) + Math.pow(lastPoint[1] - targetY, 2)
    );

    // The distances should be small (endpoints should be near the nodes)
    const threshold = 0.1;
    assert(
      distToSource < threshold,
      "First point should be near the source node"
    );
    assert(
      distToTarget < threshold,
      "Last point should be near the target node"
    );
  }
};

/**
 * Test integration of FDEB with other components
 */
exports.testFdebIntegration = function () {
  // Create a graph with multiple edges
  const graph = new eg.Graph();
  const nodes = [];

  // Create nodes
  for (let i = 0; i < 6; i++) {
    nodes.push(graph.addNode({ id: i }));
  }

  // Add edges to create a complete graph (all nodes connected to all others)
  const edges = [];
  for (let i = 0; i < 6; i++) {
    for (let j = i + 1; j < 6; j++) {
      edges.push(graph.addEdge(nodes[i], nodes[j], { weight: i + j }));
    }
  }

  // Create a drawing with initial positions
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Set positions in a circle
  for (let i = 0; i < 6; i++) {
    const angle = (i / 6) * 2 * Math.PI;
    drawing.setX(i, Math.cos(angle));
    drawing.setY(i, Math.sin(angle));
  }

  // Apply a layout algorithm to optimize the positions
  const layout = new eg.StressMajorization(graph, drawing, (e) => {
    return { distance: 1.0 };
  });

  // Apply the layout algorithm multiple times
  for (let i = 0; i < 20; i++) {
    layout.apply(drawing);
  }

  // Store node positions before calling fdeb
  const nodePositions = storeNodePositions(drawing, nodes.length);

  // Apply FDEB algorithm to the optimized layout
  const bundledEdges = eg.fdeb(graph, drawing);

  // Verify the structure
  const edgeIndices = Array.from({ length: edges.length }, (_, i) => i);
  verifyBundledEdgesStructure(bundledEdges, edgeIndices);
  verifyPointCoordinates(bundledEdges, edgeIndices);

  // Calculate the total length of all bundled edges
  let totalLength = 0;
  for (let i = 0; i < edges.length; i++) {
    const points = bundledEdges.get(i);
    for (let j = 1; j < points.length; j++) {
      const dx = points[j][0] - points[j - 1][0];
      const dy = points[j][1] - points[j - 1][1];
      totalLength += Math.sqrt(dx * dx + dy * dy);
    }
  }

  // Verify that the total length is a finite number and positive
  assert(
    Number.isFinite(totalLength),
    "Total length of bundled edges should be a finite number"
  );
  assert(totalLength > 0, "Total length of bundled edges should be positive");

  // Calculate the total length of straight edges for comparison
  let straightLength = 0;
  for (const edge of edges) {
    const endpoints = graph.edgeEndpoints(edge);
    const sourceNode = endpoints[0];
    const targetNode = endpoints[1];

    const dx = nodePositions[targetNode].x - nodePositions[sourceNode].x;
    const dy = nodePositions[targetNode].y - nodePositions[sourceNode].y;
    straightLength += Math.sqrt(dx * dx + dy * dy);
  }

  // Bundled edges are typically longer than straight edges
  assert(
    totalLength >= straightLength,
    "Total length of bundled edges should be at least as long as straight edges"
  );
};
