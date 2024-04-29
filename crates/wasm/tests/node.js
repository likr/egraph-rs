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

function checkSimulation(graph, forces) {
  const { Coordinates, Simulation } = eg;
  const coordinates = Coordinates.initialPlacement(graph);
  const simulation = new Simulation();
  simulation.run((alpha) => {
    for (const force of forces) {
      force.apply(coordinates, alpha);
    }
  });
  checkResult(graph, coordinates);
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

exports.testCoordinates = function (data) {
  const { Coordinates } = eg;
  const graph = constructGraph(data);
  const coordinates = Coordinates.initialPlacement(graph);
  assert.strictEqual(coordinates.len(), data.nodes.length);
  assert(Number.isFinite(coordinates.x(0)));
  coordinates.setX(0, 42);
  assert.strictEqual(coordinates.x(0), 42);
  assert(Number.isFinite(coordinates.y(0)));
  coordinates.setY(0, 42);
  assert.strictEqual(coordinates.y(0), 42);
};

exports.testSimulation = function (data) {
  const { Simulation } = eg;
  const repeat = 300;
  const simulation = new Simulation();
  simulation.iterations = repeat;
  let count = 0;
  simulation.run((alpha) => {
    assert(Number.isFinite(alpha));
    count += 1;
  });
  assert.strictEqual(repeat, count);
};

exports.testForceDirectedLayout = function (data) {
  const { Coordinates, Simulation, ManyBodyForce, LinkForce } = eg;
  const graph = constructGraph(data);
  const coordinates = Coordinates.initialPlacement(graph);
  const forces = [
    new ManyBodyForce(graph, () => ({ strength: -30 })),
    new LinkForce(graph, () => ({ distance: 30 })),
  ];
  const simulation = new Simulation();
  simulation.iterations = 1;
  simulation.run((alpha) => {
    for (const force of forces) {
      force.apply(coordinates, alpha);
    }
  });
  for (const u of graph.nodeIndices()) {
    assert(Number.isFinite(coordinates.x(u)));
    assert(Number.isFinite(coordinates.y(u)));
  }
};

exports.testHyperbolicForceDirectedLayout = function (data) {
  const { Coordinates, Simulation, ManyBodyForce, LinkForce } = eg;
  const graph = constructGraph(data);
  const coordinates = Coordinates.initialPlacement(graph);
  const tangentSpace = Coordinates.initialPlacement(graph);
  const forces = [
    new ManyBodyForce(graph, () => ({ strength: -0.5 })),
    new LinkForce(graph, () => ({ distance: 0.5 })),
  ];
  const simulation = new Simulation();
  simulation.run((alpha) => {
    applyInHyperbolicSpace(coordinates, tangentSpace, (u) => {
      for (const force of forces) {
        force.applyToNode(u, tangentSpace, alpha);
      }
    });
  });
  for (const u of graph.nodeIndices()) {
    assert(Number.isFinite(coordinates.x(u)));
    assert(Number.isFinite(coordinates.y(u)));
  }
};

exports.testCollideForce = function (data) {
  const { CollideForce } = eg;
  const graph = constructGraph(data);
  checkSimulation(graph, [
    new CollideForce(graph, () => ({ radius: 10 }), {
      strength: 0.1,
      iterations: 1,
    }),
  ]);
};

exports.testLinkForce = function (data) {
  const { LinkForce } = eg;
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
  const { ManyBodyForce } = eg;
  const graph = constructGraph(data);
  checkSimulation(graph, [new ManyBodyForce(graph)]);
  checkSimulation(graph, [new ManyBodyForce(graph, () => ({}))]);
  checkSimulation(graph, [new ManyBodyForce(graph, () => ({ strength: 30 }))]);
};

exports.testPositionForce = function (data) {
  const { PositionForce } = eg;
  const graph = constructGraph(data);
  checkSimulation(graph, [
    new PositionForce(graph, () => ({ strength: 0.1, x: 0, y: 0 })),
  ]);
};

exports.testRadialForce = function (data) {
  const { RadialForce } = eg;
  const graph = constructGraph(data);
  checkSimulation(graph, [
    new RadialForce(graph, () => ({ strength: 0.1, radius: 100, x: 0, y: 0 })),
  ]);
};

exports.testGroupLinkForce = function (data) {
  const { GroupLinkForce } = eg;
  const graph = constructGraph(data);
  checkSimulation(graph, [
    new GroupLinkForce(
      graph,
      (u) => ({ group: data.nodes[u].group }),
      () => ({ distance: 30, strength: 0.1 }),
      () => ({ distance: 30, strength: 0.01 })
    ),
  ]);
};

exports.testGroupManyBodyForce = function (data) {
  const { GroupManyBodyForce } = eg;
  const graph = constructGraph(data);
  checkSimulation(graph, [
    new GroupManyBodyForce(graph, (u) => ({
      group: data.nodes[u].group,
      strength: 0.1,
    })),
  ]);
};

exports.testGroupPositionForce = function (data) {
  const { GroupPositionForce } = eg;
  const graph = constructGraph(data);
  const groups = {
    0: [-41.81871, -26.057499],
    1: [-27.468754, 19.578482],
    2: [-16.478506, -16.130516],
    3: [2.144027, -3.559982],
    4: [-72.14328, -40.380527],
    5: [32.31394, 14.893769],
    6: [-1.9056562, 15.8394985],
    7: [-8.233656, -45.34843],
    8: [15.518936, -31.066544],
    9: [-45.534084, 4.6855717],
  };
  checkSimulation(graph, [
    new GroupPositionForce(
      graph,
      (u) => ({ group: data.nodes[u].group, strength: 0.1 }),
      (g) => ({ x: groups[g][0], y: groups[g][1] })
    ),
  ]);
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
  scheduler.step((eta) => {
    sgd.shuffle(rng);
    sgd.step(eta);
  });
  checkResult(graph, drawing);
};

exports.testSparseSgd = function (data) {
  const rng = eg.Rng.seedFrom(0n);
  const graph = constructGraph(data);
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);
  const sgd = new eg.SparseSgd(graph, () => 100, 50, rng);
  const scheduler = sgd.scheduler(15, 0.1);
  scheduler.step((eta) => {
    sgd.shuffle(rng);
    sgd.step(eta);
  });
  checkResult(graph, drawing);
};

exports.testCoarsen = function (data) {
  const graph = constructGraph(data);
  const [coarsenedGraph, _] = eg.coarsen(
    graph,
    (u) => data.nodes[u].group,
    (children) => ({ children }),
    (children) => ({ children })
  );

  assert.strictEqual(coarsenedGraph.nodeCount(), 10);
  assert.strictEqual(coarsenedGraph.edgeCount(), 17);

  let totalNodes = 0;
  for (const g of coarsenedGraph.nodeIndices()) {
    totalNodes += coarsenedGraph.nodeWeight(g).children.length;
  }
  assert.strictEqual(graph.nodeCount(), totalNodes);
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
