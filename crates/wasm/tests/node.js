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

function checkSimulation(graph, forces) {
  const { Coordinates, Simulation } = wasm;
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
  const { Coordinates } = wasm;
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
  const { Simulation } = wasm;
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
  const { Coordinates, Simulation, ManyBodyForce, LinkForce } = wasm;
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
  const { Coordinates, Simulation, ManyBodyForce, LinkForce } = wasm;
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

exports.testKamadaKawai = function (data) {
  const { Coordinates, KamadaKawai } = wasm;
  const graph = constructGraph(data);
  const coordinates = Coordinates.initialPlacement(graph);
  const kamadaKawai = new KamadaKawai(graph, () => ({ distance: 1 }));
  kamadaKawai.run(coordinates);
  checkResult(graph, coordinates);
};

exports.testStressMajorization = function (data) {
  const { Coordinates, StressMajorization } = wasm;
  const graph = constructGraph(data);
  const coordinates = Coordinates.initialPlacement(graph);
  const stressMajorization = new StressMajorization(graph, coordinates, () => ({
    distance: 100,
  }));
  stressMajorization.run(coordinates);
  checkResult(graph, coordinates);
};

exports.testCoarsen = function (data) {
  const { coarsen } = wasm;
  const graph = constructGraph(data);
  const coarsenedGraph = coarsen(
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
