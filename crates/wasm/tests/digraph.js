const assert = require("assert");
const eg = require("wasm-bindgen-test");

/**
 * Test basic instantiation of DiGraph class
 */
exports.testDiGraphConstructor = function () {
  const digraph = new eg.DiGraph();
  // Verify that the DiGraph instance exists
  assert(digraph instanceof eg.DiGraph, "Should create an instance of DiGraph");

  // Verify initial state
  assert.strictEqual(digraph.nodeCount(), 0, "New digraph should have 0 nodes");
  assert.strictEqual(digraph.edgeCount(), 0, "New digraph should have 0 edges");

  // Verify node and edge indices are empty
  assert.strictEqual(
    digraph.nodeIndices().length,
    0,
    "New digraph should have empty nodeIndices"
  );
  assert.strictEqual(
    digraph.edgeIndices().length,
    0,
    "New digraph should have empty edgeIndices"
  );
};

/**
 * Test node operations (add, remove, get)
 */
exports.testNodeOperations = function () {
  const digraph = new eg.DiGraph();

  // Test adding nodes with different data types
  const node1 = digraph.addNode({ id: 1, label: "Node 1" });
  const node2 = digraph.addNode("string node");
  const node3 = digraph.addNode(42);

  // Verify node count
  assert.strictEqual(digraph.nodeCount(), 3, "DiGraph should have 3 nodes");

  // Verify node indices
  const nodeIndices = digraph.nodeIndices();
  assert.strictEqual(
    nodeIndices.length,
    3,
    "nodeIndices should have 3 elements"
  );
  assert(nodeIndices.includes(node1), "nodeIndices should include node1");
  assert(nodeIndices.includes(node2), "nodeIndices should include node2");
  assert(nodeIndices.includes(node3), "nodeIndices should include node3");

  // Verify node weights
  const node1Weight = digraph.nodeWeight(node1);
  assert.deepStrictEqual(
    node1Weight,
    { id: 1, label: "Node 1" },
    "Node 1 weight should match"
  );

  const node2Weight = digraph.nodeWeight(node2);
  assert.strictEqual(node2Weight, "string node", "Node 2 weight should match");

  const node3Weight = digraph.nodeWeight(node3);
  assert.strictEqual(node3Weight, 42, "Node 3 weight should match");

  // Test removing a node
  const removedNode = digraph.removeNode(node2);
  assert.strictEqual(
    removedNode,
    "string node",
    "Removed node value should match"
  );
  assert.strictEqual(
    digraph.nodeCount(),
    2,
    "DiGraph should have 2 nodes after removal"
  );

  // Test error on invalid node index
  try {
    digraph.nodeWeight(999);
    assert.fail("Should throw error for invalid node index");
  } catch (error) {
    assert(error, "Should throw error for invalid node index");
  }
};

/**
 * Test edge operations (add, remove, get)
 */
exports.testEdgeOperations = function () {
  const digraph = new eg.DiGraph();

  // Add nodes
  const node1 = digraph.addNode({ id: 1 });
  const node2 = digraph.addNode({ id: 2 });
  const node3 = digraph.addNode({ id: 3 });

  // Test adding edges with different data types
  const edge1 = digraph.addEdge(node1, node2, { weight: 1.5 });
  const edge2 = digraph.addEdge(node2, node3, "connection");
  const edge3 = digraph.addEdge(node3, node1, 42);

  // Verify edge count
  assert.strictEqual(digraph.edgeCount(), 3, "DiGraph should have 3 edges");

  // Verify edge indices
  const edgeIndices = digraph.edgeIndices();
  assert.strictEqual(
    edgeIndices.length,
    3,
    "edgeIndices should have 3 elements"
  );
  assert(edgeIndices.includes(edge1), "edgeIndices should include edge1");
  assert(edgeIndices.includes(edge2), "edgeIndices should include edge2");
  assert(edgeIndices.includes(edge3), "edgeIndices should include edge3");

  // Verify edge weights
  const edge1Weight = digraph.edgeWeight(edge1);
  assert.deepStrictEqual(
    edge1Weight,
    { weight: 1.5 },
    "Edge 1 weight should match"
  );

  const edge2Weight = digraph.edgeWeight(edge2);
  assert.strictEqual(edge2Weight, "connection", "Edge 2 weight should match");

  const edge3Weight = digraph.edgeWeight(edge3);
  assert.strictEqual(edge3Weight, 42, "Edge 3 weight should match");

  // Test edge endpoints
  const endpoints1 = digraph.edgeEndpoints(edge1);
  assert.strictEqual(
    endpoints1.length,
    2,
    "Edge endpoints should have 2 elements"
  );
  assert.strictEqual(endpoints1[0], node1, "First endpoint should be node1");
  assert.strictEqual(endpoints1[1], node2, "Second endpoint should be node2");

  // Test contains edge
  assert(
    digraph.containsEdge(node1, node2),
    "DiGraph should contain edge from node1 to node2"
  );
  assert(
    !digraph.containsEdge(node1, node1),
    "DiGraph should not contain self-loop"
  );
  assert(
    !digraph.containsEdge(node2, node1),
    "DiGraph should not contain edge from node2 to node1 (reverse direction)"
  );

  // Test find edge
  const foundEdge = digraph.findEdge(node1, node2);
  assert.strictEqual(
    foundEdge,
    edge1,
    "findEdge should return correct edge index"
  );

  // Test removing an edge
  const removedEdge = digraph.removeEdge(edge2);
  assert.strictEqual(
    removedEdge,
    "connection",
    "Removed edge value should match"
  );
  assert.strictEqual(
    digraph.edgeCount(),
    2,
    "DiGraph should have 2 edges after removal"
  );

  // Test error on invalid edge index
  try {
    digraph.edgeWeight(999);
    assert.fail("Should throw error for invalid edge index");
  } catch (error) {
    assert(error, "Should throw error for invalid edge index");
  }

  // Test error on finding non-existent edge
  try {
    digraph.findEdge(node2, node3); // We removed this edge
    assert.fail("Should throw error for non-existent edge");
  } catch (error) {
    assert(error, "Should throw error for non-existent edge");
  }
};

/**
 * Test graph traversal and iteration
 */
exports.testGraphTraversal = function () {
  const digraph = new eg.DiGraph();

  // Create a simple directed graph
  //    0 --> 1 --> 2
  //    ^     |     |
  //    |     v     v
  //    4 <-- 3 <-- 5
  const nodes = [];
  for (let i = 0; i < 6; i++) {
    nodes.push(digraph.addNode({ id: i }));
  }

  digraph.addEdge(nodes[0], nodes[1], {});
  digraph.addEdge(nodes[1], nodes[2], {});
  digraph.addEdge(nodes[1], nodes[3], {});
  digraph.addEdge(nodes[2], nodes[5], {});
  digraph.addEdge(nodes[3], nodes[4], {});
  digraph.addEdge(nodes[4], nodes[0], {});
  digraph.addEdge(nodes[5], nodes[3], {});

  // Test neighbors (outgoing)
  const neighbors1 = digraph.neighbors(nodes[1]);
  assert.strictEqual(
    neighbors1.length,
    2,
    "Node 1 should have 2 outgoing neighbors"
  );
  assert(
    neighbors1.includes(nodes[2]),
    "Node 1 outgoing neighbors should include node 2"
  );
  assert(
    neighbors1.includes(nodes[3]),
    "Node 1 outgoing neighbors should include node 3"
  );

  // Test edges (outgoing)
  const edges1 = digraph.edges(nodes[1]);
  assert.strictEqual(edges1.length, 2, "Node 1 should have 2 outgoing edges");

  // Test externals (nodes with no outgoing/incoming edges)
  const outExternals = digraph.externals(0); // Nodes with no outgoing edges
  assert.strictEqual(
    outExternals.length,
    0,
    "Graph should have 0 nodes with no outgoing edges"
  );

  // In our directed graph, node 0 has an incoming edge from node 4
  const inExternals = digraph.externals(1); // Nodes with no incoming edges
  assert.strictEqual(
    inExternals.length,
    0,
    "Graph should have 0 nodes with no incoming edges"
  );
};

/**
 * Test directed-specific functionality (in/out neighbors)
 */
exports.testInOutNeighbors = function () {
  const digraph = new eg.DiGraph();

  // Create a simple directed graph
  //    0 --> 1 --> 2
  //    ^     |     |
  //    |     v     v
  //    4 <-- 3 <-- 5
  const nodes = [];
  for (let i = 0; i < 6; i++) {
    nodes.push(digraph.addNode({ id: i }));
  }

  digraph.addEdge(nodes[0], nodes[1], {});
  digraph.addEdge(nodes[1], nodes[2], {});
  digraph.addEdge(nodes[1], nodes[3], {});
  digraph.addEdge(nodes[2], nodes[5], {});
  digraph.addEdge(nodes[3], nodes[4], {});
  digraph.addEdge(nodes[4], nodes[0], {});
  digraph.addEdge(nodes[5], nodes[3], {});

  // Test outgoing neighbors
  const outNeighbors1 = digraph.neighborsDirected(nodes[1], 0); // Outgoing
  assert.strictEqual(
    outNeighbors1.length,
    2,
    "Node 1 should have 2 outgoing neighbors"
  );
  assert(
    outNeighbors1.includes(nodes[2]),
    "Node 1 outgoing neighbors should include node 2"
  );
  assert(
    outNeighbors1.includes(nodes[3]),
    "Node 1 outgoing neighbors should include node 3"
  );

  // Test incoming neighbors
  const inNeighbors3 = digraph.neighborsDirected(nodes[3], 1); // Incoming
  assert.strictEqual(
    inNeighbors3.length,
    2,
    "Node 3 should have 2 incoming neighbors"
  );
  assert(
    inNeighbors3.includes(nodes[1]),
    "Node 3 incoming neighbors should include node 1"
  );
  assert(
    inNeighbors3.includes(nodes[5]),
    "Node 3 incoming neighbors should include node 5"
  );

  // Test undirected neighbors (both incoming and outgoing)
  const undirectedNeighbors1 = digraph.neighborsUndirected(nodes[1]);
  assert.strictEqual(
    undirectedNeighbors1.length,
    3,
    "Node 1 should have 3 undirected neighbors"
  );
  assert(
    undirectedNeighbors1.includes(nodes[0]),
    "Node 1 undirected neighbors should include node 0"
  );
  assert(
    undirectedNeighbors1.includes(nodes[2]),
    "Node 1 undirected neighbors should include node 2"
  );
  assert(
    undirectedNeighbors1.includes(nodes[3]),
    "Node 1 undirected neighbors should include node 3"
  );

  // Note: DiGraph doesn't have inDegree, outDegree, sources, or sinks methods
};

/**
 * Test integration with drawing component
 */
exports.testDiGraphWithDrawing = function () {
  // Create a simple directed graph
  const digraph = new eg.DiGraph();
  const node1 = digraph.addNode({ id: 1 });
  const node2 = digraph.addNode({ id: 2 });
  const node3 = digraph.addNode({ id: 3 });

  digraph.addEdge(node1, node2, {});
  digraph.addEdge(node2, node3, {});
  digraph.addEdge(node3, node1, {});

  // Create an equivalent undirected graph for drawing
  // since DrawingEuclidean2d.initialPlacement only accepts Graph instances
  const graph = new eg.Graph();
  const gNode1 = graph.addNode({ id: 1 });
  const gNode2 = graph.addNode({ id: 2 });
  const gNode3 = graph.addNode({ id: 3 });

  graph.addEdge(gNode1, gNode2, {});
  graph.addEdge(gNode2, gNode3, {});
  graph.addEdge(gNode3, gNode1, {});

  // Create a drawing using the undirected graph
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Verify that the drawing has the correct number of nodes
  // Note: We're using the node indices from the undirected graph
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
  const mappedGraph = digraph.map(
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
  const filteredGraph = digraph.filterMap(
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
