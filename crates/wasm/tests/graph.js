const assert = require("assert");
const eg = require("wasm-bindgen-test");

/**
 * Test basic instantiation of Graph class
 */
exports.testGraphConstructor = function () {
  const graph = new eg.Graph();
  // Verify that the Graph instance exists
  assert(graph instanceof eg.Graph, "Should create an instance of Graph");

  // Verify initial state
  assert.strictEqual(graph.nodeCount(), 0, "New graph should have 0 nodes");
  assert.strictEqual(graph.edgeCount(), 0, "New graph should have 0 edges");

  // Verify node and edge indices are empty
  assert.strictEqual(
    graph.nodeIndices().length,
    0,
    "New graph should have empty nodeIndices"
  );
  assert.strictEqual(
    graph.edgeIndices().length,
    0,
    "New graph should have empty edgeIndices"
  );
};

/**
 * Test node operations (add, remove, get)
 */
exports.testNodeOperations = function () {
  const graph = new eg.Graph();

  // Test adding nodes with different data types
  const node1 = graph.addNode({ id: 1, label: "Node 1" });
  const node2 = graph.addNode("string node");
  const node3 = graph.addNode(42);

  // Verify node count
  assert.strictEqual(graph.nodeCount(), 3, "Graph should have 3 nodes");

  // Verify node indices
  const nodeIndices = graph.nodeIndices();
  assert.strictEqual(
    nodeIndices.length,
    3,
    "nodeIndices should have 3 elements"
  );
  assert(nodeIndices.includes(node1), "nodeIndices should include node1");
  assert(nodeIndices.includes(node2), "nodeIndices should include node2");
  assert(nodeIndices.includes(node3), "nodeIndices should include node3");

  // Verify node weights
  const node1Weight = graph.nodeWeight(node1);
  assert.deepStrictEqual(
    node1Weight,
    { id: 1, label: "Node 1" },
    "Node 1 weight should match"
  );

  const node2Weight = graph.nodeWeight(node2);
  assert.strictEqual(node2Weight, "string node", "Node 2 weight should match");

  const node3Weight = graph.nodeWeight(node3);
  assert.strictEqual(node3Weight, 42, "Node 3 weight should match");

  // Test removing a node
  const removedNode = graph.removeNode(node2);
  assert.strictEqual(
    removedNode,
    "string node",
    "Removed node value should match"
  );
  assert.strictEqual(
    graph.nodeCount(),
    2,
    "Graph should have 2 nodes after removal"
  );

  // Test error on invalid node index
  try {
    graph.nodeWeight(999);
    assert.fail("Should throw error for invalid node index");
  } catch (error) {
    assert(error, "Should throw error for invalid node index");
  }
};

/**
 * Test edge operations (add, remove, get)
 */
exports.testEdgeOperations = function () {
  const graph = new eg.Graph();

  // Add nodes
  const node1 = graph.addNode({ id: 1 });
  const node2 = graph.addNode({ id: 2 });
  const node3 = graph.addNode({ id: 3 });

  // Test adding edges with different data types
  const edge1 = graph.addEdge(node1, node2, { weight: 1.5 });
  const edge2 = graph.addEdge(node2, node3, "connection");
  const edge3 = graph.addEdge(node3, node1, 42);

  // Verify edge count
  assert.strictEqual(graph.edgeCount(), 3, "Graph should have 3 edges");

  // Verify edge indices
  const edgeIndices = graph.edgeIndices();
  assert.strictEqual(
    edgeIndices.length,
    3,
    "edgeIndices should have 3 elements"
  );
  assert(edgeIndices.includes(edge1), "edgeIndices should include edge1");
  assert(edgeIndices.includes(edge2), "edgeIndices should include edge2");
  assert(edgeIndices.includes(edge3), "edgeIndices should include edge3");

  // Verify edge weights
  const edge1Weight = graph.edgeWeight(edge1);
  assert.deepStrictEqual(
    edge1Weight,
    { weight: 1.5 },
    "Edge 1 weight should match"
  );

  const edge2Weight = graph.edgeWeight(edge2);
  assert.strictEqual(edge2Weight, "connection", "Edge 2 weight should match");

  const edge3Weight = graph.edgeWeight(edge3);
  assert.strictEqual(edge3Weight, 42, "Edge 3 weight should match");

  // Test edge endpoints
  const endpoints1 = graph.edgeEndpoints(edge1);
  assert.strictEqual(
    endpoints1.length,
    2,
    "Edge endpoints should have 2 elements"
  );
  assert.strictEqual(endpoints1[0], node1, "First endpoint should be node1");
  assert.strictEqual(endpoints1[1], node2, "Second endpoint should be node2");

  // Test contains edge
  assert(
    graph.containsEdge(node1, node2),
    "Graph should contain edge between node1 and node2"
  );
  assert(
    !graph.containsEdge(node1, node1),
    "Graph should not contain self-loop"
  );

  // Test find edge
  const foundEdge = graph.findEdge(node1, node2);
  assert.strictEqual(
    foundEdge,
    edge1,
    "findEdge should return correct edge index"
  );

  // Test removing an edge
  const removedEdge = graph.removeEdge(edge2);
  assert.strictEqual(
    removedEdge,
    "connection",
    "Removed edge value should match"
  );
  assert.strictEqual(
    graph.edgeCount(),
    2,
    "Graph should have 2 edges after removal"
  );

  // Test error on invalid edge index
  try {
    graph.edgeWeight(999);
    assert.fail("Should throw error for invalid edge index");
  } catch (error) {
    assert(error, "Should throw error for invalid edge index");
  }

  // Test error on finding non-existent edge
  try {
    graph.findEdge(node2, node3); // We removed this edge
    assert.fail("Should throw error for non-existent edge");
  } catch (error) {
    assert(error, "Should throw error for non-existent edge");
  }
};

/**
 * Test graph traversal and iteration
 */
exports.testGraphTraversal = function () {
  const graph = new eg.Graph();

  // Create a simple graph
  //    0
  //   / \
  //  1---2
  //  |   |
  //  3---4
  const nodes = [];
  for (let i = 0; i < 5; i++) {
    nodes.push(graph.addNode({ id: i }));
  }

  graph.addEdge(nodes[0], nodes[1], {});
  graph.addEdge(nodes[0], nodes[2], {});
  graph.addEdge(nodes[1], nodes[2], {});
  graph.addEdge(nodes[1], nodes[3], {});
  graph.addEdge(nodes[2], nodes[4], {});
  graph.addEdge(nodes[3], nodes[4], {});

  // Test neighbors
  const neighbors0 = graph.neighbors(nodes[0]);
  assert.strictEqual(neighbors0.length, 2, "Node 0 should have 2 neighbors");
  assert(
    neighbors0.includes(nodes[1]),
    "Node 0 neighbors should include node 1"
  );
  assert(
    neighbors0.includes(nodes[2]),
    "Node 0 neighbors should include node 2"
  );

  const neighbors1 = graph.neighbors(nodes[1]);
  assert.strictEqual(neighbors1.length, 3, "Node 1 should have 3 neighbors");
  assert(
    neighbors1.includes(nodes[0]),
    "Node 1 neighbors should include node 0"
  );
  assert(
    neighbors1.includes(nodes[2]),
    "Node 1 neighbors should include node 2"
  );
  assert(
    neighbors1.includes(nodes[3]),
    "Node 1 neighbors should include node 3"
  );

  // Test neighbors undirected (should be the same as neighbors for undirected graph)
  const neighborsUndirected1 = graph.neighborsUndirected(nodes[1]);
  assert.strictEqual(
    neighborsUndirected1.length,
    3,
    "Node 1 should have 3 undirected neighbors"
  );
  assert(
    neighborsUndirected1.includes(nodes[0]),
    "Node 1 undirected neighbors should include node 0"
  );
  assert(
    neighborsUndirected1.includes(nodes[2]),
    "Node 1 undirected neighbors should include node 2"
  );
  assert(
    neighborsUndirected1.includes(nodes[3]),
    "Node 1 undirected neighbors should include node 3"
  );

  // Test neighbors directed
  const outNeighbors1 = graph.neighborsDirected(nodes[1], 0); // Outgoing
  assert.strictEqual(
    outNeighbors1.length,
    3,
    "Node 1 should have 3 outgoing neighbors"
  );

  const inNeighbors1 = graph.neighborsDirected(nodes[1], 1); // Incoming
  assert.strictEqual(
    inNeighbors1.length,
    3,
    "Node 1 should have 3 incoming neighbors"
  );

  // Test edges
  const edges1 = graph.edges(nodes[1]);
  assert.strictEqual(edges1.length, 3, "Node 1 should have 3 edges");

  // Test externals (nodes with no outgoing/incoming edges)
  // In this graph, there are no externals since all nodes have both incoming and outgoing edges
  const outExternals = graph.externals(0);
  assert.strictEqual(
    outExternals.length,
    0,
    "Graph should have 0 nodes with no outgoing edges"
  );

  const inExternals = graph.externals(1);
  assert.strictEqual(
    inExternals.length,
    0,
    "Graph should have 0 nodes with no incoming edges"
  );
};

/**
 * Test integration with drawing component
 */
exports.testGraphWithDrawing = function () {
  // Create a simple graph
  const graph = new eg.Graph();
  const node1 = graph.addNode({ id: 1 });
  const node2 = graph.addNode({ id: 2 });
  const node3 = graph.addNode({ id: 3 });

  graph.addEdge(node1, node2, {});
  graph.addEdge(node2, node3, {});
  graph.addEdge(node3, node1, {});

  // Create a drawing
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Verify that the drawing has the correct number of nodes
  for (const nodeIndex of graph.nodeIndices()) {
    // Check that coordinates are finite numbers
    assert(
      Number.isFinite(drawing.x(nodeIndex)),
      "X coordinate should be a finite number"
    );
    assert(
      Number.isFinite(drawing.y(nodeIndex)),
      "Y coordinate should be a finite number"
    );
  }

  // Test map function
  const mappedGraph = graph.map(
    (nodeIndex, nodeValue) => ({ ...nodeValue, mapped: true }),
    (edgeIndex, edgeValue) => ({ ...edgeValue, mapped: true })
  );

  // Verify mapped node values
  for (const nodeIndex of mappedGraph.nodeIndices()) {
    const nodeValue = mappedGraph.nodeWeight(nodeIndex);
    assert(nodeValue.mapped, "Mapped node should have mapped property");
    assert(nodeValue.id, "Mapped node should preserve original properties");
  }

  // Test filter_map function
  const filteredGraph = graph.filterMap(
    (nodeIndex, nodeValue) =>
      nodeValue.id !== 2 ? { ...nodeValue, filtered: true } : null,
    (edgeIndex, edgeValue) => ({ ...edgeValue, filtered: true })
  );

  // Verify filtered graph
  assert.strictEqual(
    filteredGraph.nodeCount(),
    2,
    "Filtered graph should have 2 nodes"
  );
  assert.strictEqual(
    filteredGraph.edgeCount(),
    1,
    "Filtered graph should have 1 edge"
  );

  // Verify filtered node values
  for (const nodeIndex of filteredGraph.nodeIndices()) {
    const nodeValue = filteredGraph.nodeWeight(nodeIndex);
    assert(nodeValue.filtered, "Filtered node should have filtered property");
    assert(nodeValue.id !== 2, "Node with id 2 should be filtered out");
  }
};
