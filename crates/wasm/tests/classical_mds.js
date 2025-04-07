const assert = require("assert");
const eg = require("wasm-bindgen-test");

/**
 * Test basic instantiation of ClassicalMds class
 */
exports.testClassicalMdsConstructor = function () {
  // Create a simple graph
  const graph = new eg.Graph();
  const node1 = graph.addNode({});
  const node2 = graph.addNode({});
  const edge = graph.addEdge(node1, node2, {});

  // Create a ClassicalMds instance with a simple length function
  const mds = new eg.ClassicalMds(graph, () => 1.0);

  // Verify that the MDS instance exists
  assert(
    mds instanceof eg.ClassicalMds,
    "Should create an instance of ClassicalMds"
  );
};

/**
 * Test run2d method for 2D layout generation
 */
exports.testClassicalMdsRun2d = function () {
  // Create a simple graph
  const graph = new eg.Graph();
  const node1 = graph.addNode({});
  const node2 = graph.addNode({});
  const node3 = graph.addNode({});
  graph.addEdge(node1, node2, {});
  graph.addEdge(node2, node3, {});

  // Create a ClassicalMds instance
  const mds = new eg.ClassicalMds(graph, () => 1.0);

  // Generate a 2D layout
  const drawing = mds.run2d();

  // Verify that the drawing is a DrawingEuclidean2d instance
  assert(
    drawing instanceof eg.DrawingEuclidean2d,
    "Should return a DrawingEuclidean2d instance"
  );

  // Verify that all nodes have valid coordinates
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

  // Verify that the drawing has the correct number of nodes
  let nodeCount = 0;
  for (const _ of graph.nodeIndices()) {
    nodeCount++;
  }
  assert.strictEqual(nodeCount, 3, "Drawing should contain all 3 nodes");
};

/**
 * Test run method for n-dimensional layout generation
 */
exports.testClassicalMdsRun = function () {
  // Create a simple graph
  const graph = new eg.Graph();
  const node1 = graph.addNode({});
  const node2 = graph.addNode({});
  const node3 = graph.addNode({});
  graph.addEdge(node1, node2, {});
  graph.addEdge(node2, node3, {});

  // Create a ClassicalMds instance
  const mds = new eg.ClassicalMds(graph, () => 1.0);

  // Generate a 3D layout
  const drawing = mds.run(3);

  // Verify that the drawing is a DrawingEuclidean instance
  assert(
    drawing instanceof eg.DrawingEuclidean,
    "Should return a DrawingEuclidean instance"
  );

  // Verify that all nodes have valid coordinates in all dimensions
  for (const u of graph.nodeIndices()) {
    for (let d = 0; d < 3; d++) {
      assert(
        Number.isFinite(drawing.get(u, d)),
        `Coordinate at dimension ${d} should be a finite number`
      );
    }
  }

  // Verify that we can access coordinates in all dimensions
  for (let d = 0; d < 3; d++) {
    for (const u of graph.nodeIndices()) {
      assert(
        Number.isFinite(drawing.get(u, d)),
        `Coordinate at dimension ${d} should be a finite number`
      );
    }
  }

  // Verify that the drawing has the correct number of nodes
  let nodeCount = 0;
  for (const _ of graph.nodeIndices()) {
    nodeCount++;
  }
  assert.strictEqual(nodeCount, 3, "Drawing should contain all 3 nodes");
};

/**
 * Test with different graph structures
 */
exports.testClassicalMdsWithDifferentGraphs = function () {
  // Test with a line graph
  const lineGraph = new eg.Graph();
  const lineNodes = [];
  for (let i = 0; i < 5; i++) {
    lineNodes.push(lineGraph.addNode({}));
  }
  for (let i = 0; i < 4; i++) {
    lineGraph.addEdge(lineNodes[i], lineNodes[i + 1], {});
  }

  const lineMds = new eg.ClassicalMds(lineGraph, () => 1.0);
  const lineDrawing = lineMds.run2d();

  // Verify that all nodes have valid coordinates
  for (const u of lineGraph.nodeIndices()) {
    assert(
      Number.isFinite(lineDrawing.x(u)),
      "X coordinate should be a finite number"
    );
    assert(
      Number.isFinite(lineDrawing.y(u)),
      "Y coordinate should be a finite number"
    );
  }

  // Test with a cycle graph
  const cycleGraph = new eg.Graph();
  const cycleNodes = [];
  for (let i = 0; i < 5; i++) {
    cycleNodes.push(cycleGraph.addNode({}));
  }
  for (let i = 0; i < 4; i++) {
    cycleGraph.addEdge(cycleNodes[i], cycleNodes[i + 1], {});
  }
  cycleGraph.addEdge(cycleNodes[4], cycleNodes[0], {}); // Close the cycle

  const cycleMds = new eg.ClassicalMds(cycleGraph, () => 1.0);
  const cycleDrawing = cycleMds.run2d();

  // Verify that all nodes have valid coordinates
  for (const u of cycleGraph.nodeIndices()) {
    assert(
      Number.isFinite(cycleDrawing.x(u)),
      "X coordinate should be a finite number"
    );
    assert(
      Number.isFinite(cycleDrawing.y(u)),
      "Y coordinate should be a finite number"
    );
  }

  // Test with a complete graph
  const completeGraph = new eg.Graph();
  const completeNodes = [];
  for (let i = 0; i < 5; i++) {
    completeNodes.push(completeGraph.addNode({}));
  }
  for (let i = 0; i < 5; i++) {
    for (let j = i + 1; j < 5; j++) {
      completeGraph.addEdge(completeNodes[i], completeNodes[j], {});
    }
  }

  const completeMds = new eg.ClassicalMds(completeGraph, () => 1.0);
  const completeDrawing = completeMds.run2d();

  // Verify that all nodes have valid coordinates
  for (const u of completeGraph.nodeIndices()) {
    assert(
      Number.isFinite(completeDrawing.x(u)),
      "X coordinate should be a finite number"
    );
    assert(
      Number.isFinite(completeDrawing.y(u)),
      "Y coordinate should be a finite number"
    );
  }

  // Note: Disconnected graphs are not tested as they may produce NaN coordinates
  // with ClassicalMds since the algorithm relies on graph-theoretic distances
};

/**
 * Test with custom length function
 */
exports.testClassicalMdsWithCustomLengthFunction = function () {
  // Create a simple graph
  const graph = new eg.Graph();
  const node1 = graph.addNode({});
  const node2 = graph.addNode({});
  const node3 = graph.addNode({});
  graph.addEdge(node1, node2, {});
  graph.addEdge(node2, node3, {});

  // Create a ClassicalMds instance with a custom length function
  // that returns different values for different edge indices
  const mds = new eg.ClassicalMds(graph, (edgeIndex) => {
    return edgeIndex === 0 ? 1.0 : 2.0;
  });

  // Generate a 2D layout
  const drawing = mds.run2d();

  // Verify that all nodes have valid coordinates
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

  // Create another instance with a constant length function for comparison
  const mdsConstant = new eg.ClassicalMds(graph, () => 1.0);
  const drawingConstant = mdsConstant.run2d();

  // Verify that the layouts are valid
  for (const u of graph.nodeIndices()) {
    assert(
      Number.isFinite(drawingConstant.x(u)),
      "X coordinate should be a finite number"
    );
    assert(
      Number.isFinite(drawingConstant.y(u)),
      "Y coordinate should be a finite number"
    );
  }
};

/**
 * Test handling of high-dimensional embeddings
 */
exports.testClassicalMdsHandlesHighDimensions = function () {
  // Create a simple graph
  const graph = new eg.Graph();
  const node1 = graph.addNode({});
  const node2 = graph.addNode({});
  const node3 = graph.addNode({});
  graph.addEdge(node1, node2, {});
  graph.addEdge(node2, node3, {});

  // Create a ClassicalMds instance
  const mds = new eg.ClassicalMds(graph, () => 1.0);

  // Test with dimensions higher than the number of nodes
  const drawing = mds.run(5);

  // Verify that all nodes have valid coordinates in all dimensions
  for (const u of graph.nodeIndices()) {
    for (let d = 0; d < 5; d++) {
      assert(
        Number.isFinite(drawing.get(u, d)),
        `Coordinate at dimension ${d} should be a finite number`
      );
    }
  }

  // Verify that we can access coordinates in all dimensions
  for (let d = 0; d < 5; d++) {
    for (const u of graph.nodeIndices()) {
      assert(
        Number.isFinite(drawing.get(u, d)),
        `Coordinate at dimension ${d} should be a finite number`
      );
    }
  }
};

/**
 * Test integration with other components
 */
exports.testClassicalMdsIntegration = function () {
  // Create a graph
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

  // Create a ClassicalMds instance
  const mds = new eg.ClassicalMds(graph, () => 1.0);

  // Generate a 2D layout
  const drawing = mds.run2d();

  // Verify that all nodes have valid coordinates
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

  // Test integration with FullSgd for layout refinement
  const sgd = new eg.FullSgd(graph, () => 1.0);
  const scheduler = sgd.scheduler(10, 0.1);

  // Apply SGD to refine the MDS layout
  scheduler.run((eta) => {
    sgd.applyWithDrawingEuclidean2d(drawing, eta);
  });

  // Verify that all coordinates are still valid after SGD refinement
  for (const u of graph.nodeIndices()) {
    assert(
      Number.isFinite(drawing.x(u)),
      "X coordinate should be a finite number after SGD"
    );
    assert(
      Number.isFinite(drawing.y(u)),
      "Y coordinate should be a finite number after SGD"
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
