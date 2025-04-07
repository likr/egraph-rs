const assert = require("assert");
const eg = require("wasm-bindgen-test");

/**
 * Test basic instantiation of KamadaKawai class
 */
exports.testKamadaKawaiConstructor = function () {
  // Create a simple graph
  const graph = new eg.Graph();
  const node1 = graph.addNode({});
  const node2 = graph.addNode({});
  const edge = graph.addEdge(node1, node2, {});

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
  const graph = new eg.Graph();
  const node1 = graph.addNode({});
  const node2 = graph.addNode({});
  graph.addEdge(node1, node2, {});

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
  const graph = new eg.Graph();
  const node1 = graph.addNode({});
  const node2 = graph.addNode({});
  const node3 = graph.addNode({});
  graph.addEdge(node1, node2, {});
  graph.addEdge(node2, node3, {});

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
  const graph = new eg.Graph();
  const node1 = graph.addNode({});
  const node2 = graph.addNode({});
  const node3 = graph.addNode({});
  graph.addEdge(node1, node2, {});
  graph.addEdge(node2, node3, {});

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
  const graph = new eg.Graph();
  const node1 = graph.addNode({});
  const node2 = graph.addNode({});
  const node3 = graph.addNode({});
  graph.addEdge(node1, node2, {});
  graph.addEdge(node2, node3, {});

  // Create a KamadaKawai instance
  const layout = new eg.KamadaKawai(graph, () => ({ distance: 1.0 }));

  // Create a drawing
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Record initial positions
  const initialPositions = {};
  for (const u of graph.nodeIndices()) {
    initialPositions[u] = { x: drawing.x(u), y: drawing.y(u) };
  }

  // Run the complete algorithm
  layout.run(drawing);

  // Verify that positions have changed
  let positionsChanged = false;
  for (const u of graph.nodeIndices()) {
    if (
      Math.abs(drawing.x(u) - initialPositions[u].x) > 1e-10 ||
      Math.abs(drawing.y(u) - initialPositions[u].y) > 1e-10
    ) {
      positionsChanged = true;
      break;
    }
  }
  assert(
    positionsChanged,
    "Node positions should change after running the algorithm"
  );

  // Verify that all coordinates are finite numbers
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
};

/**
 * Test integration with other components
 */
exports.testKamadaKawaiIntegration = function () {
  // Create a more complex graph
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

  // Create a drawing
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Create a KamadaKawai instance with custom distance function
  const layout = new eg.KamadaKawai(graph, (e) => {
    const endpoints = graph.edgeEndpoints(e);
    const u = endpoints[0];
    const v = endpoints[1];
    // Use node IDs to determine distance
    const uId = graph.nodeWeight(u).id;
    const vId = graph.nodeWeight(v).id;
    return { distance: Math.abs(uId - vId) };
  });

  // Set a larger epsilon for faster convergence in tests
  layout.eps = 0.01;

  // Run the layout algorithm
  layout.run(drawing);

  // Verify that all coordinates are finite numbers
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
