const assert = require("assert");
const eg = require("wasm-bindgen-test");
const helpers = require("./util/test_helpers");

/**
 * Test basic instantiation of KamadaKawai class
 */
exports.testKamadaKawaiConstructor = function () {
  // Create a simple graph
  const { graph } = helpers.createLineGraph(2);

  // Create a KamadaKawai instance with a simple distance function
  const layout = new eg.KamadaKawai(graph, () => ({ distance: 1.0 }));

  // Verify that the layout instance exists
  assert(
    layout instanceof eg.KamadaKawai,
    "Should create an instance of KamadaKawai"
  );
};

/**
 * Test epsilon parameter getter and setter
 */
exports.testKamadaKawaiEpsilon = function () {
  // Create a simple graph
  const { graph } = helpers.createLineGraph(2);

  // Create a KamadaKawai instance
  const layout = new eg.KamadaKawai(graph, () => ({ distance: 1.0 }));

  // Test default epsilon value
  assert(typeof layout.eps === "number", "Epsilon should be a number");
  assert(layout.eps > 0, "Epsilon should be positive");

  // Test setting epsilon
  const newEps = 0.005;
  layout.eps = newEps;

  // Check if the epsilon value was updated
  // Note: Due to potential floating-point precision issues,
  // we'll check if the value is close to what we set
  assert(
    Math.abs(layout.eps - newEps) < 1e-6,
    "Epsilon should be updated to approximately the new value"
  );
};

/**
 * Test node selection functionality
 */
exports.testKamadaKawaiSelectNode = function () {
  // Create a simple graph
  const { graph } = helpers.createLineGraph(3);

  // Create a KamadaKawai instance
  const layout = new eg.KamadaKawai(graph, () => ({ distance: 1.0 }));

  // Create a drawing
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Test node selection
  const selectedNode = layout.selectNode(drawing);

  // Verify that a valid node is selected
  assert(
    selectedNode === null ||
      (typeof selectedNode === "number" &&
        selectedNode >= 0 &&
        selectedNode < 3),
    "Selected node should be null or a valid node index"
  );
};

/**
 * Test applying the algorithm to a single node
 */
exports.testKamadaKawaiApplyToNode = function () {
  // Create a simple graph
  const { graph } = helpers.createLineGraph(3);

  // Create a KamadaKawai instance
  const layout = new eg.KamadaKawai(graph, () => ({ distance: 1.0 }));

  // Create a drawing
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Record initial position of node 1
  const initialX = drawing.x(1);
  const initialY = drawing.y(1);

  // Apply the algorithm to node 1
  layout.applyToNode(1, drawing);

  // Verify that the position of node 1 has changed
  assert(
    drawing.x(1) !== initialX || drawing.y(1) !== initialY,
    "Node position should change after applying the algorithm"
  );

  // Verify that the coordinates are finite numbers
  assert(
    Number.isFinite(drawing.x(1)),
    "X coordinate should be a finite number"
  );
  assert(
    Number.isFinite(drawing.y(1)),
    "Y coordinate should be a finite number"
  );
};

/**
 * Test running the complete algorithm
 */
exports.testKamadaKawaiRun = function () {
  // Create a simple graph
  const { graph } = helpers.createLineGraph(3);

  // Create a drawing
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Record initial positions
  const initialPositions = helpers.recordInitialPositions2d(drawing, graph);

  // Apply Kamada-Kawai layout
  const layout = new eg.KamadaKawai(graph, () => ({ distance: 1.0 }));
  layout.run(drawing);

  // Verify that positions have changed
  helpers.verifyPositionsChanged2d(
    drawing,
    graph,
    initialPositions,
    "Node positions should change after running the algorithm"
  );

  // Verify that all coordinates are finite numbers
  helpers.verifyFiniteCoordinates2d(drawing, graph);
};

/**
 * Test integration with other components
 */
exports.testKamadaKawaiIntegration = function () {
  // Create a more complex graph
  const graph = new eg.Graph();
  const nodes = [];

  // Create nodes
  for (let i = 0; i < 10; i++) {
    nodes.push(graph.addNode({ id: i }));
  }

  // Create a path
  for (let i = 0; i < 9; i++) {
    graph.addEdge(nodes[i], nodes[i + 1], {});
  }

  // Add some cross edges
  graph.addEdge(nodes[0], nodes[5], {});
  graph.addEdge(nodes[2], nodes[7], {});
  graph.addEdge(nodes[3], nodes[8], {});

  // Create a drawing
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Custom distance function that uses node IDs
  const customDistanceFunc = (e) => {
    const endpoints = graph.edgeEndpoints(e);
    const u = endpoints[0];
    const v = endpoints[1];
    // Use node IDs to determine distance
    const uId = graph.nodeWeight(u).id;
    const vId = graph.nodeWeight(v).id;
    return { distance: Math.abs(uId - vId) };
  };

  // Apply Kamada-Kawai layout
  const layout = new eg.KamadaKawai(graph, customDistanceFunc);
  layout.eps = 0.01; // Set a larger epsilon for faster convergence in tests
  layout.run(drawing);

  // Verify layout quality
  helpers.verifyLayoutQuality(graph, drawing);

  // Verify that connected nodes are positioned closer together
  helpers.verifyConnectedNodesCloser(graph, drawing);
};
