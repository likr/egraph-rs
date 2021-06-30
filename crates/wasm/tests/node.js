const assert = require("assert");
const wasm = require("wasm-bindgen-test");

exports.testConstructGraph = function (data) {
  const { Graph } = wasm;
  const graph = new Graph();
  const indices = new Map();
  for (const node of data.nodes) {
    indices.set(node.id, graph.addNode(node));
  }
  for (const link of data.links) {
    const { source, target } = link;
    graph.addEdge(indices.get(source), indices.get(target), link);
  }
  assert.strictEqual(graph.nodeCount(), data.nodes.length);
  assert.strictEqual(graph.edgeCount(), data.links.length);
};
