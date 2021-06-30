const assert = require("assert");
const wasm = require("wasm-bindgen-test");

function constructGraph(data) {
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
  return graph;
}

function runSimulation(graph, forces) {
  const { Simulation, initialPlacement } = wasm;
  const initialCoordinates = initialPlacement(graph);
  const simulation = new Simulation(graph, (u) => initialCoordinates[u]);
  return simulation.run(forces);
}

function checkSimulation(graph, forces) {
  const coordinates = runSimulation(graph, forces);
  for (const u of graph.nodeIndices()) {
    assert(Number.isFinite(coordinates[u][0]));
    assert(Number.isFinite(coordinates[u][1]));
  }
}

exports.testConstructGraph = function (data) {
  const graph = constructGraph(data);
  assert.strictEqual(graph.nodeCount(), data.nodes.length);
  assert.strictEqual(graph.edgeCount(), data.links.length);
};

exports.testSimulation = function (data) {
  const { Simulation, initialPlacement } = wasm;
  const graph = constructGraph(data);
  const initialCoordinates = initialPlacement(graph);
  const simulation = new Simulation(graph, (u) => initialCoordinates[u]);
  const coordinates = simulation.run([]);
  for (const u of graph.nodeIndices()) {
    assert.strictEqual(initialCoordinates[u][0], coordinates[u][0]);
    assert.strictEqual(initialCoordinates[u][1], coordinates[u][1]);
  }
};

exports.testCenterForce = function (data) {
  const { CenterForce } = wasm;
  const graph = constructGraph(data);
  checkSimulation(graph, [new CenterForce()]);
};

exports.testCollideForce = function (data) {
  const { CollideForce } = wasm;
  const graph = constructGraph(data);
  checkSimulation(graph, [
    new CollideForce(graph, () => ({ radius: 10 }), {
      strength: 0.1,
      iterations: 1,
    }),
  ]);
};

exports.testLinkForce = function (data) {
  const { LinkForce } = wasm;
  const graph = constructGraph(data);
  checkSimulation(graph, [new LinkForce(graph)]);
  checkSimulation(graph, [new LinkForce(graph, () => ({}))]);
  checkSimulation(graph, [new LinkForce(graph, () => ({ strength: 0.5 }))]);
  checkSimulation(graph, [new LinkForce(graph, () => ({ distance: 30 }))]);
  checkSimulation(graph, [
    new LinkForce(graph, () => ({ distance: 30, strength: 0.5 })),
  ]);
};

exports.testManyBodyForce = function (data) {
  const { ManyBodyForce } = wasm;
  const graph = constructGraph(data);
  checkSimulation(graph, [new ManyBodyForce(graph)]);
  checkSimulation(graph, [new ManyBodyForce(graph, () => ({}))]);
  checkSimulation(graph, [new ManyBodyForce(graph, () => ({ strength: 30 }))]);
};

exports.testPositionForce = function (data) {
  const { PositionForce } = wasm;
  const graph = constructGraph(data);
  checkSimulation(graph, [
    new PositionForce(graph, () => ({ strength: 0.1, x: 0, y: 0 })),
  ]);
};

exports.testRadialForce = function (data) {
  const { RadialForce } = wasm;
  const graph = constructGraph(data);
  checkSimulation(graph, [
    new RadialForce(graph, () => ({ strength: 0.1, radius: 100, x: 0, y: 0 })),
  ]);
};
