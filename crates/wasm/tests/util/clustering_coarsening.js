const assert = require("assert");
const eg = require("wasm-bindgen-test");

/**
 * Helper function to verify the structure of a coarsened graph
 * @param {Object} graph - The original graph
 * @param {Object} coarsenedGraph - The coarsened graph
 * @param {Object} groupMap - Mapping from group IDs to node indices in the coarsened graph
 * @param {Function} groupFn - The grouping function used for coarsening
 */
function verifyCoarsenedGraphStructure(
  graph,
  coarsenedGraph,
  groupMap,
  groupFn
) {
  // Verify that the coarsened graph is a valid graph
  assert(
    coarsenedGraph instanceof eg.Graph,
    "Coarsened graph should be a Graph instance"
  );

  // Verify that the number of nodes in the coarsened graph matches the number of unique groups
  const uniqueGroups = new Set();
  for (const u of graph.nodeIndices()) {
    uniqueGroups.add(groupFn(u));
  }
  assert.strictEqual(
    coarsenedGraph.nodeCount(),
    uniqueGroups.size,
    "Number of nodes in coarsened graph should match number of unique groups"
  );

  // Convert uniqueGroups Set to an array of strings for comparison
  const uniqueGroupsArray = Array.from(uniqueGroups).map(String);

  // Verify that the group map contains entries for all unique groups
  // Note: The group map keys are strings, so we need to convert our uniqueGroups to strings
  const groupMapKeys = Object.keys(groupMap);

  // Check that all groups in the map are valid
  for (const groupId of groupMapKeys) {
    assert(
      uniqueGroupsArray.includes(groupId),
      `Group map contains invalid group ID: ${groupId}`
    );
  }

  // Check that all unique groups are in the map
  for (const groupId of uniqueGroupsArray) {
    assert(
      groupMapKeys.includes(groupId),
      `Group map is missing entry for group ID: ${groupId}`
    );
  }

  // Verify that all group IDs in the map correspond to valid nodes in the coarsened graph
  for (const groupId in groupMap) {
    const nodeIndex = groupMap[groupId];
    assert(
      nodeIndex >= 0 && nodeIndex < coarsenedGraph.nodeCount(),
      `Node index ${nodeIndex} for group ${groupId} should be valid in the coarsened graph`
    );
  }
}

/**
 * Test basic graph coarsening functionality
 */
function testBasicCoarsening() {
  // Create a simple graph with 4 nodes and 4 edges
  const graph = new eg.Graph();
  const node0 = graph.addNode({ id: 0 });
  const node1 = graph.addNode({ id: 1 });
  const node2 = graph.addNode({ id: 2 });
  const node3 = graph.addNode({ id: 3 });

  // Create a path graph: node0 -- node1 -- node2 -- node3
  const edge0 = graph.addEdge(node0, node1, { weight: 1 });
  const edge1 = graph.addEdge(node1, node2, { weight: 2 });
  const edge2 = graph.addEdge(node2, node3, { weight: 3 });
  const edge3 = graph.addEdge(node3, node0, { weight: 4 }); // Make it a cycle

  // Store node and edge data to avoid accessing the graph in callbacks
  const nodeData = {};
  for (const u of graph.nodeIndices()) {
    nodeData[u] = graph.nodeWeight(u);
  }

  const edgeData = {};
  for (const e of graph.edgeIndices()) {
    edgeData[e] = graph.edgeWeight(e);
  }

  // Define a simple grouping function: nodes 0,1 in group 0, nodes 2,3 in group 1
  const groupFn = (u) => Math.floor(u / 2);

  // Define node and edge merging functions
  const shrinkNodeFn = (nodeIds) => {
    return {
      ids: Array.from(nodeIds),
      count: nodeIds.length,
    };
  };

  const shrinkEdgeFn = (edgeIds) => {
    return {
      ids: Array.from(edgeIds),
      count: edgeIds.length,
    };
  };

  // Apply coarsening
  const result = eg.coarsen(graph, groupFn, shrinkNodeFn, shrinkEdgeFn);

  // Extract coarsened graph and group map
  const coarsenedGraph = result[0];
  const groupMap = result[1];

  // Debug: Log the group map and unique groups
  console.log("Group Map:", groupMap);
  console.log("Coarsened Graph:", coarsenedGraph);

  const uniqueGroups = new Set();
  for (const u of graph.nodeIndices()) {
    uniqueGroups.add(groupFn(u));
  }
  console.log("Unique Groups:", Array.from(uniqueGroups));

  // Convert the group map to a plain object if it's a Map
  const groupMapObj = {};
  if (groupMap instanceof Map) {
    for (const [key, value] of groupMap.entries()) {
      groupMapObj[key] = value;
    }
  } else {
    Object.assign(groupMapObj, groupMap);
  }

  console.log("Group Map as Object:", groupMapObj);

  verifyCoarsenedGraphStructure(graph, coarsenedGraph, groupMapObj, groupFn);

  // Verify that the coarsened graph has the expected number of nodes
  assert.strictEqual(
    coarsenedGraph.nodeCount(),
    2,
    "Coarsened graph should have 2 nodes"
  );

  // Verify that there are edges between the coarsened nodes
  assert(
    coarsenedGraph.containsEdge(groupMapObj["0"], groupMapObj["1"]) ||
      coarsenedGraph.containsEdge(groupMapObj["1"], groupMapObj["0"]),
    "Coarsened graph should have an edge between group 0 and group 1"
  );

  // Verify that the node weights contain the expected information
  for (const u of coarsenedGraph.nodeIndices()) {
    const weight = coarsenedGraph.nodeWeight(u);
    assert(Array.isArray(weight.ids), "Node weight should have an 'ids' array");
    assert.strictEqual(
      weight.count,
      weight.ids.length,
      "Node weight 'count' should match the length of 'ids'"
    );
    assert.strictEqual(weight.count, 2, "Each group should contain 2 nodes");
  }

  // Verify that the edge weights contain the expected information
  for (const e of coarsenedGraph.edgeIndices()) {
    const weight = coarsenedGraph.edgeWeight(e);
    assert(Array.isArray(weight.ids), "Edge weight should have an 'ids' array");
    assert.strictEqual(
      weight.count,
      weight.ids.length,
      "Edge weight 'count' should match the length of 'ids'"
    );
  }
}

/**
 * Test coarsening with a more complex graph
 */
function testComplexGraphCoarsening() {
  // Create a more complex graph with 9 nodes in a 3x3 grid
  const graph = new eg.Graph();
  const nodes = [];

  // Create nodes
  for (let i = 0; i < 9; i++) {
    nodes.push(graph.addNode({ id: i, row: Math.floor(i / 3), col: i % 3 }));
  }

  // Connect nodes in a grid pattern
  const edges = [];
  // Horizontal connections
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 2; col++) {
      const index = row * 3 + col;
      edges.push(
        graph.addEdge(nodes[index], nodes[index + 1], { type: "horizontal" })
      );
    }
  }

  // Vertical connections
  for (let row = 0; row < 2; row++) {
    for (let col = 0; col < 3; col++) {
      const index = row * 3 + col;
      edges.push(
        graph.addEdge(nodes[index], nodes[index + 3], { type: "vertical" })
      );
    }
  }

  // Store node and edge data to avoid accessing the graph in callbacks
  const nodeData = {};
  for (const u of graph.nodeIndices()) {
    nodeData[u] = graph.nodeWeight(u);
  }

  const edgeData = {};
  for (const e of graph.edgeIndices()) {
    edgeData[e] = graph.edgeWeight(e);
  }

  // Define a grouping function based on the quadrant (2x2 grid)
  const groupFn = (u) => {
    const data = nodeData[u];
    // Group by quadrant: top-left, top-right, bottom-left, bottom-right
    return (data.row < 1.5 ? 0 : 1) + (data.col < 1.5 ? 0 : 2);
  };

  // Define node and edge merging functions
  const shrinkNodeFn = (nodeIds) => {
    return {
      ids: Array.from(nodeIds),
      count: nodeIds.length,
    };
  };

  const shrinkEdgeFn = (edgeIds) => {
    // Count edge types
    const types = {};
    for (const e of edgeIds) {
      const type = edgeData[e].type;
      types[type] = (types[type] || 0) + 1;
    }
    return {
      ids: Array.from(edgeIds),
      count: edgeIds.length,
      types,
    };
  };

  // Apply coarsening
  const result = eg.coarsen(graph, groupFn, shrinkNodeFn, shrinkEdgeFn);

  // Extract coarsened graph and group map
  const coarsenedGraph = result[0];
  const groupMap = result[1];

  // Debug: Log the group map and unique groups
  console.log("Complex Graph - Group Map:", groupMap);
  console.log("Complex Graph - Coarsened Graph:", coarsenedGraph);

  const uniqueGroups = new Set();
  for (const u of graph.nodeIndices()) {
    uniqueGroups.add(groupFn(u));
  }
  console.log("Complex Graph - Unique Groups:", Array.from(uniqueGroups));

  // Convert the group map to a plain object if it's a Map
  const groupMapObj = {};
  if (groupMap instanceof Map) {
    for (const [key, value] of groupMap.entries()) {
      groupMapObj[key] = value;
    }
  } else {
    Object.assign(groupMapObj, groupMap);
  }

  console.log("Complex Graph - Group Map as Object:", groupMapObj);

  verifyCoarsenedGraphStructure(graph, coarsenedGraph, groupMapObj, groupFn);

  // Verify that there are exactly 4 nodes in the coarsened graph (4 quadrants)
  assert.strictEqual(
    coarsenedGraph.nodeCount(),
    4,
    "Coarsened graph should have 4 nodes"
  );

  // Verify that the coarsened graph has the expected edges
  // Each quadrant should be connected to its adjacent quadrants
  assert(
    coarsenedGraph.containsEdge(groupMapObj["0"], groupMapObj["1"]) ||
      coarsenedGraph.containsEdge(groupMapObj["1"], groupMapObj["0"]),
    "Coarsened graph should have an edge between quadrants 0 and 1"
  );
  assert(
    coarsenedGraph.containsEdge(groupMapObj["0"], groupMapObj["2"]) ||
      coarsenedGraph.containsEdge(groupMapObj["2"], groupMapObj["0"]),
    "Coarsened graph should have an edge between quadrants 0 and 2"
  );
  assert(
    coarsenedGraph.containsEdge(groupMapObj["1"], groupMapObj["3"]) ||
      coarsenedGraph.containsEdge(groupMapObj["3"], groupMapObj["1"]),
    "Coarsened graph should have an edge between quadrants 1 and 3"
  );
  assert(
    coarsenedGraph.containsEdge(groupMapObj["2"], groupMapObj["3"]) ||
      coarsenedGraph.containsEdge(groupMapObj["3"], groupMapObj["2"]),
    "Coarsened graph should have an edge between quadrants 2 and 3"
  );

  // Verify that the node weights contain the expected information
  for (const u of coarsenedGraph.nodeIndices()) {
    const weight = coarsenedGraph.nodeWeight(u);
    assert(Array.isArray(weight.ids), "Node weight should have an 'ids' array");
    assert.strictEqual(
      weight.count,
      weight.ids.length,
      "Node weight 'count' should match the length of 'ids'"
    );
  }

  // Verify that the edge weights contain the expected information
  for (const e of coarsenedGraph.edgeIndices()) {
    const weight = coarsenedGraph.edgeWeight(e);
    assert(Array.isArray(weight.ids), "Edge weight should have an 'ids' array");
    assert.strictEqual(
      weight.count,
      weight.ids.length,
      "Edge weight 'count' should match the length of 'ids'"
    );
    assert(
      typeof weight.types === "object",
      "Edge weight should have a 'types' object"
    );
    // The sum of type counts should equal the total count
    const typeSum = Object.values(weight.types).reduce(
      (sum, count) => sum + count,
      0
    );
    assert.strictEqual(
      typeSum,
      weight.count,
      "Sum of type counts should equal the total count"
    );
  }
}

module.exports = {
  verifyCoarsenedGraphStructure,
  testBasicCoarsening,
  testComplexGraphCoarsening,
};
