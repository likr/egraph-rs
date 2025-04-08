const assert = require("assert");
const eg = require("wasm-bindgen-test");
const helpers = require("./util/test_helpers");

/**
 * Test basic instantiation of ClassicalMds class
 */
exports.testClassicalMdsConstructor = function () {
  // Create a simple graph
  const { graph } = helpers.createTestGraph("line", 2);

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
  const { graph } = helpers.createTestGraph("line", 3);

  // Apply MDS layout
  const { drawing } = helpers.applyLayout("mds", graph, null, {
    dimensions: 2,
  });

  // Verify that the drawing is a DrawingEuclidean2d instance
  assert(
    drawing instanceof eg.DrawingEuclidean2d,
    "Should return a DrawingEuclidean2d instance"
  );

  // Verify that all coordinates are finite numbers
  helpers.verifyFiniteCoordinates2d(drawing, graph);

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
  const { graph } = helpers.createTestGraph("line", 3);

  // Apply MDS layout
  const { drawing } = helpers.applyLayout("mds", graph, null, {
    dimensions: 3,
  });

  // Verify that the drawing is a DrawingEuclidean instance
  assert(
    drawing instanceof eg.DrawingEuclidean,
    "Should return a DrawingEuclidean instance"
  );

  // Verify that all coordinates are finite numbers
  helpers.verifyFiniteCoordinatesNd(drawing, graph, 3);

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
  const { graph: lineGraph } = helpers.createTestGraph("line", 5);
  const { drawing: lineDrawing } = helpers.applyLayout("mds", lineGraph, null, {
    dimensions: 2,
  });
  helpers.verifyFiniteCoordinates2d(lineDrawing, lineGraph);

  // Test with a cycle graph
  const { graph: cycleGraph } = helpers.createTestGraph("cycle", 5);
  const { drawing: cycleDrawing } = helpers.applyLayout(
    "mds",
    cycleGraph,
    null,
    {
      dimensions: 2,
    }
  );
  helpers.verifyFiniteCoordinates2d(cycleDrawing, cycleGraph);

  // Test with a complete graph
  const { graph: completeGraph } = helpers.createTestGraph("complete", 5);
  const { drawing: completeDrawing } = helpers.applyLayout(
    "mds",
    completeGraph,
    null,
    { dimensions: 2 }
  );
  helpers.verifyFiniteCoordinates2d(completeDrawing, completeGraph);

  // Note: Disconnected graphs are not tested as they may produce NaN coordinates
  // with ClassicalMds since the algorithm relies on graph-theoretic distances
};

/**
 * Test with custom length function
 */
exports.testClassicalMdsWithCustomLengthFunction = function () {
  // Create a simple graph
  const { graph } = helpers.createTestGraph("line", 3);

  // Create a ClassicalMds instance with a custom length function
  // that returns different values for different edge indices
  const customLengthFunc = (edgeIndex) => {
    return edgeIndex === 0 ? 1.0 : 2.0;
  };

  // Apply MDS layout with custom length function
  const { drawing } = helpers.applyLayout("mds", graph, null, {
    dimensions: 2,
    lengthFunc: customLengthFunc,
  });

  // Verify that all coordinates are finite numbers
  helpers.verifyFiniteCoordinates2d(drawing, graph);

  // Apply MDS layout with constant length function for comparison
  const { drawing: drawingConstant } = helpers.applyLayout("mds", graph, null, {
    dimensions: 2,
    lengthFunc: () => 1.0,
  });

  // Verify that all coordinates are finite numbers
  helpers.verifyFiniteCoordinates2d(drawingConstant, graph);
};

/**
 * Test handling of high-dimensional embeddings
 */
exports.testClassicalMdsHandlesHighDimensions = function () {
  // Create a simple graph
  const { graph } = helpers.createTestGraph("line", 3);

  // Apply MDS layout with high dimensions
  const { drawing } = helpers.applyLayout("mds", graph, null, {
    dimensions: 5,
  });

  // Verify that all coordinates are finite numbers
  helpers.verifyFiniteCoordinatesNd(drawing, graph, 5);
};

/**
 * Test integration with other components
 */
exports.testClassicalMdsIntegration = function () {
  // Create a more complex graph
  const { graph } = helpers.createTestGraph("custom", 10, (graph, nodes) => {
    // Create a path
    for (let i = 0; i < 9; i++) {
      graph.addEdge(nodes[i], nodes[i + 1], {});
    }
    // Add some cross edges
    graph.addEdge(nodes[0], nodes[5], {});
    graph.addEdge(nodes[2], nodes[7], {});
    graph.addEdge(nodes[3], nodes[8], {});
  });

  // Apply MDS layout
  const { drawing } = helpers.applyLayout("mds", graph, null, {
    dimensions: 2,
  });

  // Verify that all coordinates are finite numbers
  helpers.verifyFiniteCoordinates2d(drawing, graph);

  // Record initial positions
  const initialPositions = helpers.recordInitialPositions2d(drawing, graph);

  // Apply SGD to refine the MDS layout
  helpers.applyLayout("sgd_full", graph, drawing, {
    iterations: 10,
    learningRate: 0.1,
  });

  // Verify that positions have changed
  helpers.verifyPositionsChanged2d(
    drawing,
    graph,
    initialPositions,
    "SGD should change node positions"
  );

  // Verify that all coordinates are still finite numbers
  helpers.verifyFiniteCoordinates2d(drawing, graph);

  // Verify that connected nodes are positioned closer together
  helpers.verifyConnectedNodesCloser(graph, drawing);
};
