const assert = require("assert");
const eg = require("wasm-bindgen-test");

/**
 * Test basic instantiation of Rng class
 */
exports.testRngConstructor = function () {
  const rng = new eg.Rng();
  // Verify that the RNG instance exists
  assert(rng instanceof eg.Rng, "Should create an instance of Rng");
};

/**
 * Test seeded random number generation
 * This test verifies that the same seed produces the same sequence of random numbers
 */
exports.testRngSeedFrom = function () {
  // Create two RNGs with the same seed
  const seed = 42n;
  const rng1 = eg.Rng.seedFrom(seed);
  const rng2 = eg.Rng.seedFrom(seed);

  // Test with SparseSgd which uses RNG internally
  const graph = new eg.Graph();
  const node1 = graph.addNode({});
  const node2 = graph.addNode({});
  const node3 = graph.addNode({});
  graph.addEdge(node1, node2, {});
  graph.addEdge(node2, node3, {});

  // Create two SparseSgd instances with the same parameters but different RNG instances
  const sgd1 = new eg.SparseSgd(graph, () => 100, 10, rng1);
  const sgd2 = new eg.SparseSgd(graph, () => 100, 10, rng2);

  // Shuffle both and verify they produce the same result
  sgd1.shuffle(rng1);
  sgd2.shuffle(rng2);

  // Since we can't directly compare the internal state, we'll use the SGD to test
  // that the random sequences are the same by applying them to identical drawings
  const drawing1 = eg.DrawingEuclidean2d.initialPlacement(graph);
  const drawing2 = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Apply the same operations with both RNGs
  sgd1.applyWithDrawingEuclidean2d(drawing1, 0.1);
  sgd2.applyWithDrawingEuclidean2d(drawing2, 0.1);

  // Verify that the results are identical
  for (const u of graph.nodeIndices()) {
    assert.strictEqual(
      drawing1.x(u),
      drawing2.x(u),
      "X coordinates should be identical with same seed"
    );
    assert.strictEqual(
      drawing1.y(u),
      drawing2.y(u),
      "Y coordinates should be identical with same seed"
    );
  }
};

/**
 * Test that different seeds produce different sequences
 */
exports.testRngDifferentSeeds = function () {
  // Create two RNGs with different seeds
  const rng1 = eg.Rng.seedFrom(42n);
  const rng2 = eg.Rng.seedFrom(43n);

  // Test with SparseSgd which uses RNG internally
  const graph = new eg.Graph();
  const node1 = graph.addNode({});
  const node2 = graph.addNode({});
  const node3 = graph.addNode({});
  graph.addEdge(node1, node2, {});
  graph.addEdge(node2, node3, {});

  // Create two SparseSgd instances with the same parameters but different RNG instances
  const sgd1 = new eg.SparseSgd(graph, () => 100, 10, rng1);
  const sgd2 = new eg.SparseSgd(graph, () => 100, 10, rng2);

  // Shuffle both
  sgd1.shuffle(rng1);
  sgd2.shuffle(rng2);

  // Apply to identical drawings
  const drawing1 = eg.DrawingEuclidean2d.initialPlacement(graph);
  const drawing2 = eg.DrawingEuclidean2d.initialPlacement(graph);

  sgd1.applyWithDrawingEuclidean2d(drawing1, 0.1);
  sgd2.applyWithDrawingEuclidean2d(drawing2, 0.1);

  // Check if at least one coordinate is different
  // (There's a tiny probability they could be the same by chance, but it's extremely unlikely)
  let hasDifference = false;
  for (const u of graph.nodeIndices()) {
    if (drawing1.x(u) !== drawing2.x(u) || drawing1.y(u) !== drawing2.y(u)) {
      hasDifference = true;
      break;
    }
  }

  assert(
    hasDifference,
    "Different seeds should produce different random sequences"
  );
};

/**
 * Test integration with SGD layout algorithm
 */
exports.testRngWithSgdLayout = function () {
  // Create a seeded RNG for reproducible results
  const rng = eg.Rng.seedFrom(123n);

  // Create a simple graph
  const graph = new eg.Graph();
  const nodes = [];
  for (let i = 0; i < 10; i++) {
    nodes.push(graph.addNode({ id: i }));
  }

  // Add some edges
  for (let i = 0; i < 9; i++) {
    graph.addEdge(nodes[i], nodes[i + 1], {});
  }

  // Create a drawing
  const drawing = eg.DrawingEuclidean2d.initialPlacement(graph);

  // Create an SGD layout with our RNG
  const sgd = new eg.SparseSgd(graph, () => 100, 5, rng);

  // Run a few iterations
  const scheduler = sgd.scheduler(5, 0.1);
  scheduler.run((eta) => {
    sgd.shuffle(rng);
    sgd.applyWithDrawingEuclidean2d(drawing, eta);
  });

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
