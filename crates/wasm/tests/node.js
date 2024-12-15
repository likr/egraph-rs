const assert = require("assert");
const eg = require("wasm-bindgen-test");

function constructGraph(data) {
  const { Graph } = eg;
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

function checkResult(graph, coordinates) {
  for (const u of graph.nodeIndices()) {
    assert(Number.isFinite(coordinates.x(u)));
    assert(Number.isFinite(coordinates.y(u)));
  }
}

exports.testConstructGraph = function (data) {
  const graph = constructGraph(data);
  assert.strictEqual(graph.nodeCount(), data.nodes.length);
  assert.strictEqual(graph.edgeCount(), data.links.length);
};

exports.testKamadaKawai = function (data) {
  const graph = constructGraph(data);
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);
  const kamadaKawai = new eg.KamadaKawai(graph, () => ({ distance: 1 }));
  kamadaKawai.run(drawing);
  checkResult(graph, drawing);
};

exports.testStressMajorization = function (data) {
  const graph = constructGraph(data);
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);
  const stressMajorization = new eg.StressMajorization(graph, drawing, () => ({
    distance: 100,
  }));
  stressMajorization.run(drawing);
  checkResult(graph, drawing);
};

exports.testClassicalMds = function (data) {
  const graph = constructGraph(data);
  const drawing = new eg.ClassicalMds(graph, () => 100).run2d();
  checkResult(graph, drawing);
};

exports.testPivotMds = function (data) {
  const graph = constructGraph(data);
  const drawing = new eg.PivotMds(
    graph,
    () => 100,
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  ).run2d();
  checkResult(graph, drawing);
};

exports.testFullSgd = function (data) {
  const rng = eg.Rng.seedFrom(0n);
  const graph = constructGraph(data);
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);
  const sgd = new eg.FullSgd(graph, () => 100);
  const scheduler = sgd.scheduler(15, 0.1);
  scheduler.run((eta) => {
    sgd.shuffle(rng);
    sgd.applyWithDrawingEuclidean2d(drawing, eta);
  });
  checkResult(graph, drawing);
};

exports.testSparseSgd = function (data) {
  const rng = eg.Rng.seedFrom(0n);
  const graph = constructGraph(data);
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);
  const sgd = new eg.SparseSgd(graph, () => 100, 50, rng);
  const scheduler = sgd.scheduler(15, 0.1);
  scheduler.run((eta) => {
    sgd.shuffle(rng);
    sgd.applyWithDrawingEuclidean2d(drawing, eta);
  });
  checkResult(graph, drawing);
};

exports.testCrossingNumber = function (data) {
  const graph = constructGraph(data);
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);
  eg.crossingNumber(graph, drawing);
};

exports.testNeighborhoodPreservation = function (data) {
  const graph = constructGraph(data);
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);
  eg.neighborhoodPreservation(graph, drawing);
};

exports.testStress = function (data) {
  const graph = constructGraph(data);
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);
  eg.stress(graph, drawing);
};
