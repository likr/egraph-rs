const assert = require("assert");
const eg = require("wasm-bindgen-test");

/**
 * Creates a test graph with specified structure
 * @param {string} type - Graph type: 'line', 'cycle', 'complete', 'simple', or 'custom'
 * @param {number} size - Number of nodes
 * @param {Function} edgeCreator - Custom function to create edges (for 'custom' type)
 * @returns {Object} Object containing graph and nodes array
 */
function createTestGraph(type = "simple", size = 3, edgeCreator = null) {
  const graph = new eg.Graph();
  const nodes = [];

  // Create nodes
  for (let i = 0; i < size; i++) {
    nodes.push(graph.addNode({ id: i }));
  }

  // Create edges based on graph type
  if (type === "line") {
    for (let i = 0; i < size - 1; i++) {
      graph.addEdge(nodes[i], nodes[i + 1], {});
    }
  } else if (type === "cycle") {
    for (let i = 0; i < size; i++) {
      graph.addEdge(nodes[i], nodes[(i + 1) % size], {});
    }
  } else if (type === "complete") {
    for (let i = 0; i < size; i++) {
      for (let j = i + 1; j < size; j++) {
        graph.addEdge(nodes[i], nodes[j], {});
      }
    }
  } else if (type === "simple") {
    // Simple triangle graph
    if (size >= 3) {
      graph.addEdge(nodes[0], nodes[1], {});
      graph.addEdge(nodes[1], nodes[2], {});
      graph.addEdge(nodes[2], nodes[0], {});
    }
  } else if (type === "custom" && typeof edgeCreator === "function") {
    edgeCreator(graph, nodes);
  }

  return { graph, nodes };
}

/**
 * Creates a directed test graph with specified structure
 * @param {string} type - Graph type: 'line', 'cycle', 'complete', 'simple', or 'custom'
 * @param {number} size - Number of nodes
 * @param {Function} edgeCreator - Custom function to create edges (for 'custom' type)
 * @returns {Object} Object containing graph and nodes array
 */
function createTestDiGraph(type = "simple", size = 3, edgeCreator = null) {
  const graph = new eg.DiGraph();
  const nodes = [];

  // Create nodes
  for (let i = 0; i < size; i++) {
    nodes.push(graph.addNode({ id: i }));
  }

  // Create edges based on graph type
  if (type === "line") {
    for (let i = 0; i < size - 1; i++) {
      graph.addEdge(nodes[i], nodes[i + 1], {});
    }
  } else if (type === "cycle") {
    for (let i = 0; i < size; i++) {
      graph.addEdge(nodes[i], nodes[(i + 1) % size], {});
    }
  } else if (type === "complete") {
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        if (i !== j) {
          graph.addEdge(nodes[i], nodes[j], {});
        }
      }
    }
  } else if (type === "simple") {
    // Simple directed triangle graph
    if (size >= 3) {
      graph.addEdge(nodes[0], nodes[1], {});
      graph.addEdge(nodes[1], nodes[2], {});
      graph.addEdge(nodes[2], nodes[0], {});
    }
  } else if (type === "custom" && typeof edgeCreator === "function") {
    edgeCreator(graph, nodes);
  }

  return { graph, nodes };
}

/**
 * Records initial positions of nodes in a 2D drawing
 * @param {Object} drawing - Drawing object (Euclidean2d, Torus2d, etc.)
 * @param {Object} graph - Graph object
 * @returns {Object} Object mapping node indices to position objects
 */
function recordInitialPositions2d(drawing, graph) {
  const initialPositions = {};
  for (const u of graph.nodeIndices()) {
    initialPositions[u] = { x: drawing.x(u), y: drawing.y(u) };
  }
  return initialPositions;
}

/**
 * Records initial positions of nodes in a spherical drawing
 * @param {Object} drawing - DrawingSpherical2d object
 * @param {Object} graph - Graph object
 * @returns {Object} Object mapping node indices to position objects
 */
function recordInitialSphericalPositions(drawing, graph) {
  const initialPositions = {};
  for (const u of graph.nodeIndices()) {
    initialPositions[u] = { lon: drawing.lon(u), lat: drawing.lat(u) };
  }
  return initialPositions;
}

/**
 * Records initial positions of nodes in an n-dimensional drawing
 * @param {Object} drawing - DrawingEuclidean object
 * @param {Object} graph - Graph object
 * @param {number} dimensions - Number of dimensions
 * @returns {Object} Object mapping node indices to position arrays
 */
function recordInitialPositionsNd(drawing, graph, dimensions) {
  const initialPositions = {};
  for (const u of graph.nodeIndices()) {
    initialPositions[u] = Array.from({ length: dimensions }, (_, i) =>
      drawing.get(u, i)
    );
  }
  return initialPositions;
}

/**
 * Verifies that positions have changed from initial positions in a 2D drawing
 * @param {Object} drawing - Drawing object (Euclidean2d, Torus2d, etc.)
 * @param {Object} graph - Graph object
 * @param {Object} initialPositions - Initial positions object
 * @param {string} message - Assertion message
 */
function verifyPositionsChanged2d(
  drawing,
  graph,
  initialPositions,
  message = "Positions should change"
) {
  let positionsChanged = false;
  for (const u of graph.nodeIndices()) {
    if (
      drawing.x(u) !== initialPositions[u].x ||
      drawing.y(u) !== initialPositions[u].y
    ) {
      positionsChanged = true;
      break;
    }
  }
  assert(positionsChanged, message);
}

/**
 * Verifies that positions have changed from initial positions in a spherical drawing
 * @param {Object} drawing - DrawingSpherical2d object
 * @param {Object} graph - Graph object
 * @param {Object} initialPositions - Initial positions object
 * @param {string} message - Assertion message
 */
function verifySphericalPositionsChanged(
  drawing,
  graph,
  initialPositions,
  message = "Positions should change"
) {
  let positionsChanged = false;
  for (const u of graph.nodeIndices()) {
    if (
      drawing.lon(u) !== initialPositions[u].lon ||
      drawing.lat(u) !== initialPositions[u].lat
    ) {
      positionsChanged = true;
      break;
    }
  }
  assert(positionsChanged, message);
}

/**
 * Verifies that positions have changed from initial positions in an n-dimensional drawing
 * @param {Object} drawing - DrawingEuclidean object
 * @param {Object} graph - Graph object
 * @param {Object} initialPositions - Initial positions object
 * @param {number} dimensions - Number of dimensions
 * @param {string} message - Assertion message
 */
function verifyPositionsChangedNd(
  drawing,
  graph,
  initialPositions,
  dimensions,
  message = "Positions should change"
) {
  let positionsChanged = false;
  for (const u of graph.nodeIndices()) {
    for (let d = 0; d < dimensions; d++) {
      if (drawing.get(u, d) !== initialPositions[u][d]) {
        positionsChanged = true;
        break;
      }
    }
    if (positionsChanged) break;
  }
  assert(positionsChanged, message);
}

/**
 * Verifies that all coordinates in a 2D drawing are finite numbers
 * @param {Object} drawing - Drawing object (Euclidean2d, Torus2d, etc.)
 * @param {Object} graph - Graph object
 */
function verifyFiniteCoordinates2d(drawing, graph) {
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
}

/**
 * Verifies that all coordinates in a spherical drawing are finite numbers
 * @param {Object} drawing - DrawingSpherical2d object
 * @param {Object} graph - Graph object
 */
function verifyFiniteSphericalCoordinates(drawing, graph) {
  for (const u of graph.nodeIndices()) {
    assert(
      Number.isFinite(drawing.lon(u)),
      "Longitude should be a finite number"
    );
    assert(
      Number.isFinite(drawing.lat(u)),
      "Latitude should be a finite number"
    );
  }
}

/**
 * Verifies that all coordinates in an n-dimensional drawing are finite numbers
 * @param {Object} drawing - DrawingEuclidean object
 * @param {Object} graph - Graph object
 * @param {number} dimensions - Number of dimensions
 */
function verifyFiniteCoordinatesNd(drawing, graph, dimensions) {
  for (const u of graph.nodeIndices()) {
    for (let d = 0; d < dimensions; d++) {
      assert(
        Number.isFinite(drawing.get(u, d)),
        `Coordinate at dimension ${d} should be a finite number`
      );
    }
  }
}

/**
 * Verifies that all coordinates in a torus drawing are within valid range (0 to 1)
 * @param {Object} drawing - DrawingTorus2d object
 * @param {Object} graph - Graph object
 */
function verifyTorusCoordinateRange(drawing, graph) {
  for (const u of graph.nodeIndices()) {
    assert(
      drawing.x(u) >= 0 && drawing.x(u) <= 1,
      "X coordinate should be within torus range"
    );
    assert(
      drawing.y(u) >= 0 && drawing.y(u) <= 1,
      "Y coordinate should be within torus range"
    );
  }
}

/**
 * Verifies that all coordinates in a hyperbolic drawing are within the Poincaré disc
 * @param {Object} drawing - DrawingHyperbolic2d object
 * @param {Object} graph - Graph object
 */
function verifyHyperbolicCoordinateRange(drawing, graph) {
  for (const u of graph.nodeIndices()) {
    const distance = Math.sqrt(
      drawing.x(u) * drawing.x(u) + drawing.y(u) * drawing.y(u)
    );
    assert(
      distance < 1.0001, // Allow for small floating-point errors
      "Node should be within the Poincaré disc"
    );
  }
}

/**
 * Verifies that all latitudes in a spherical drawing are within valid range (-π/2 to π/2)
 * @param {Object} drawing - DrawingSpherical2d object
 * @param {Object} graph - Graph object
 */
function verifySphericalCoordinateRange(drawing, graph) {
  for (const u of graph.nodeIndices()) {
    assert(
      drawing.lat(u) >= -Math.PI / 2 && drawing.lat(u) <= Math.PI / 2,
      "Latitude should be within valid range"
    );
  }
}

/**
 * Creates a seeded RNG for reproducible tests
 * @param {BigInt} seed - Seed value
 * @returns {Object} RNG instance
 */
function createSeededRng(seed = 42n) {
  return eg.Rng.seedFrom(seed);
}

/**
 * Verifies that connected nodes are positioned closer to each other than the average distance
 * @param {Object} graph - Graph object
 * @param {Object} drawing - Drawing object (Euclidean2d)
 */
function verifyConnectedNodesCloser(graph, drawing) {
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
}

/**
 * Runs a scheduler with a specified number of iterations
 * @param {Object} scheduler - Scheduler object
 * @param {Function} callback - Callback function to run on each iteration
 */
function runScheduler(scheduler, callback) {
  scheduler.run(callback);
}

/**
 * Runs a scheduler step by step
 * @param {Object} scheduler - Scheduler object
 * @param {Function} callback - Callback function to run on each step
 */
function runSchedulerStepByStep(scheduler, callback) {
  while (!scheduler.isFinished()) {
    scheduler.step(callback);
  }
}

/**
 * Creates a star graph with a central node connected to all other nodes
 * @param {number} size - Number of nodes (including the central node)
 * @returns {Object} Object containing graph and nodes array
 */
function createStarGraph(size = 5) {
  const graph = new eg.Graph();
  const nodes = [];

  // Create nodes
  for (let i = 0; i < size; i++) {
    nodes.push(graph.addNode({ id: i }));
  }

  // Create edges from central node (0) to all other nodes
  for (let i = 1; i < size; i++) {
    graph.addEdge(nodes[0], nodes[i], {});
  }

  return { graph, nodes };
}

/**
 * Creates a grid graph with the specified width and height
 * @param {number} width - Number of nodes in the horizontal direction
 * @param {number} height - Number of nodes in the vertical direction
 * @returns {Object} Object containing graph and nodes array (as a 2D array)
 */
function createGridGraph(width = 3, height = 3) {
  const graph = new eg.Graph();
  const nodes = [];

  // Create nodes
  for (let y = 0; y < height; y++) {
    const row = [];
    for (let x = 0; x < width; x++) {
      row.push(graph.addNode({ id: y * width + x, x, y }));
    }
    nodes.push(row);
  }

  // Create horizontal edges
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width - 1; x++) {
      graph.addEdge(nodes[y][x], nodes[y][x + 1], {});
    }
  }

  // Create vertical edges
  for (let y = 0; y < height - 1; y++) {
    for (let x = 0; x < width; x++) {
      graph.addEdge(nodes[y][x], nodes[y + 1][x], {});
    }
  }

  return { graph, nodes };
}

/**
 * Creates a drawing based on the graph and drawing type
 * @param {Object} graph - Graph object
 * @param {string} drawingType - Type of drawing: 'euclidean2d', 'spherical2d', 'hyperbolic2d', 'torus2d', or 'euclidean'
 * @param {number} dimensions - Number of dimensions for euclidean drawing (default: 2)
 * @returns {Object} Drawing object
 */
function createDrawing(graph, drawingType = "euclidean2d", dimensions = 2) {
  switch (drawingType.toLowerCase()) {
    case "euclidean2d":
      return eg.DrawingEuclidean2d.initialPlacement(graph);
    case "spherical2d":
      return eg.DrawingSpherical2d.initialPlacement(graph);
    case "hyperbolic2d":
      return eg.DrawingHyperbolic2d.initialPlacement(graph);
    case "torus2d":
      return eg.DrawingTorus2d.initialPlacement(graph);
    case "euclidean":
      // For n-dimensional Euclidean drawing, we need to use ClassicalMds
      const mds = new eg.ClassicalMds(graph, () => 1.0);
      return mds.run(dimensions);
    default:
      throw new Error(`Unknown drawing type: ${drawingType}`);
  }
}

/**
 * Applies a layout algorithm to a drawing
 * @param {string} layoutType - Type of layout algorithm: 'mds', 'kamada_kawai', 'stress_majorization', 'sgd_full', 'sgd_sparse'
 * @param {Object} graph - Graph object
 * @param {Object} drawing - Drawing object
 * @param {Object} options - Options for the layout algorithm
 * @returns {Object} The layout algorithm instance
 */
function applyLayout(layoutType, graph, drawing, options = {}) {
  let layout;

  switch (layoutType.toLowerCase()) {
    case "mds":
      layout = new eg.ClassicalMds(graph, options.lengthFunc || (() => 1.0));
      if (options.dimensions === 2 || !options.dimensions) {
        drawing = layout.run2d();
      } else {
        drawing = layout.run(options.dimensions);
      }
      break;

    case "kamada_kawai":
      layout = new eg.KamadaKawai(
        graph,
        options.distanceFunc || (() => ({ distance: 1.0 }))
      );
      if (options.epsilon) {
        layout.eps = options.epsilon;
      }
      layout.run(drawing);
      break;

    case "stress_majorization":
      layout = new eg.StressMajorization(
        graph,
        drawing,
        options.distanceFunc || (() => ({ distance: 1.0 }))
      );

      // Apply the algorithm multiple times instead of using run()
      // to avoid potential infinite loops
      const iterations = options.iterations || 50;
      for (let i = 0; i < iterations; i++) {
        layout.apply(drawing);
      }
      break;

    case "sgd_full":
      layout = new eg.FullSgd(graph, options.lengthFunc || (() => 1.0));

      const scheduler = layout.scheduler(
        options.iterations || 100,
        options.learningRate || 0.1
      );

      scheduler.run((eta) => {
        if (options.rng) {
          layout.shuffle(options.rng);
        }

        if (drawing instanceof eg.DrawingEuclidean2d) {
          layout.applyWithDrawingEuclidean2d(drawing, eta);
        } else if (drawing instanceof eg.DrawingSpherical2d) {
          layout.applyWithDrawingSpherical2d(drawing, eta);
        } else if (drawing instanceof eg.DrawingHyperbolic2d) {
          layout.applyWithDrawingHyperbolic2d(drawing, eta);
        } else if (drawing instanceof eg.DrawingTorus2d) {
          layout.applyWithDrawingTorus2d(drawing, eta);
        } else {
          layout.applyWithDrawingEuclidean(drawing, eta);
        }
      });
      break;

    case "sgd_sparse":
      const rng = options.rng || new eg.Rng();
      const pivots = options.pivots || 2;

      layout = new eg.SparseSgd(
        graph,
        options.lengthFunc || (() => 1.0),
        pivots,
        rng
      );

      const sparseScheduler = layout.scheduler(
        options.iterations || 100,
        options.learningRate || 0.1
      );

      sparseScheduler.run((eta) => {
        if (options.rng) {
          layout.shuffle(options.rng);
        }

        if (drawing instanceof eg.DrawingEuclidean2d) {
          layout.applyWithDrawingEuclidean2d(drawing, eta);
        } else if (drawing instanceof eg.DrawingSpherical2d) {
          layout.applyWithDrawingSpherical2d(drawing, eta);
        } else if (drawing instanceof eg.DrawingHyperbolic2d) {
          layout.applyWithDrawingHyperbolic2d(drawing, eta);
        } else if (drawing instanceof eg.DrawingTorus2d) {
          layout.applyWithDrawingTorus2d(drawing, eta);
        } else {
          layout.applyWithDrawingEuclidean(drawing, eta);
        }
      });
      break;

    default:
      throw new Error(`Unknown layout type: ${layoutType}`);
  }

  return { layout, drawing };
}

/**
 * Verifies layout quality using various metrics
 * @param {Object} graph - Graph object
 * @param {Object} drawing - Drawing object
 * @param {Object} options - Options for verification
 */
function verifyLayoutQuality(graph, drawing, options = {}) {
  // Verify that all coordinates are finite numbers
  if (
    drawing instanceof eg.DrawingEuclidean2d ||
    drawing instanceof eg.DrawingTorus2d ||
    drawing instanceof eg.DrawingHyperbolic2d
  ) {
    verifyFiniteCoordinates2d(drawing, graph);
  } else if (drawing instanceof eg.DrawingSpherical2d) {
    verifyFiniteSphericalCoordinates(drawing, graph);
  } else if (drawing instanceof eg.DrawingEuclidean) {
    const dimensions = options.dimensions || 3;
    verifyFiniteCoordinatesNd(drawing, graph, dimensions);
  }

  // Verify that connected nodes are positioned closer together
  if (
    drawing instanceof eg.DrawingEuclidean2d &&
    options.verifyConnectedNodesCloser !== false
  ) {
    verifyConnectedNodesCloser(graph, drawing);
  }

  // Verify specific drawing constraints
  if (drawing instanceof eg.DrawingTorus2d) {
    verifyTorusCoordinateRange(drawing, graph);
  } else if (drawing instanceof eg.DrawingHyperbolic2d) {
    verifyHyperbolicCoordinateRange(drawing, graph);
  } else if (drawing instanceof eg.DrawingSpherical2d) {
    verifySphericalCoordinateRange(drawing, graph);
  }

  // Calculate stress if requested
  if (options.calculateStress) {
    const stress = eg.stress(graph, drawing);
    assert(Number.isFinite(stress), "Stress should be a finite number");
    return { stress };
  }

  return {};
}

/**
 * Verifies that layout quality has improved
 * @param {Object} graph - Graph object
 * @param {Object} beforeDrawing - Drawing before layout application
 * @param {Object} afterDrawing - Drawing after layout application
 * @param {string} metric - Metric to use for comparison: 'stress', 'crossing_number', 'neighborhood_preservation'
 * @returns {Object} Object containing before and after metric values
 */
function verifyLayoutImprovement(
  graph,
  beforeDrawing,
  afterDrawing,
  metric = "stress"
) {
  let beforeValue, afterValue;

  switch (metric.toLowerCase()) {
    case "stress":
      beforeValue = eg.stress(graph, beforeDrawing);
      afterValue = eg.stress(graph, afterDrawing);
      assert(
        afterValue <= beforeValue,
        `Stress should be reduced or equal after layout (before: ${beforeValue}, after: ${afterValue})`
      );
      break;

    case "crossing_number":
      if (beforeDrawing instanceof eg.DrawingTorus2d) {
        beforeValue = eg.crossingNumberWithDrawingTorus2d(graph, beforeDrawing);
        afterValue = eg.crossingNumberWithDrawingTorus2d(graph, afterDrawing);
      } else {
        beforeValue = eg.crossingNumber(graph, beforeDrawing);
        afterValue = eg.crossingNumber(graph, afterDrawing);
      }
      // We don't assert improvement here as some layouts might increase crossings
      // while optimizing for other metrics
      break;

    case "neighborhood_preservation":
      beforeValue = eg.neighborhoodPreservation(graph, beforeDrawing);
      afterValue = eg.neighborhoodPreservation(graph, afterDrawing);
      assert(
        afterValue >= beforeValue,
        `Neighborhood preservation should be improved or equal after layout (before: ${beforeValue}, after: ${afterValue})`
      );
      break;

    default:
      throw new Error(`Unknown metric: ${metric}`);
  }

  return { beforeValue, afterValue };
}

/**
 * Verifies that node positions match expected positions within tolerance
 * @param {Object} drawing - Drawing object
 * @param {Object} expectedPositions - Expected positions object mapping node indices to position objects
 * @param {number} tolerance - Tolerance for position comparison
 */
function verifyNodePositions(drawing, expectedPositions, tolerance = 0.001) {
  for (const [nodeIndexStr, position] of Object.entries(expectedPositions)) {
    // Convert string key back to number
    const nodeIndex = Number(nodeIndexStr);

    if (
      drawing instanceof eg.DrawingEuclidean2d ||
      drawing instanceof eg.DrawingTorus2d ||
      drawing instanceof eg.DrawingHyperbolic2d
    ) {
      assert(
        Math.abs(drawing.x(nodeIndex) - position.x) < tolerance,
        `Node ${nodeIndex} X coordinate should match expected value`
      );
      assert(
        Math.abs(drawing.y(nodeIndex) - position.y) < tolerance,
        `Node ${nodeIndex} Y coordinate should match expected value`
      );
    } else if (drawing instanceof eg.DrawingSpherical2d) {
      assert(
        Math.abs(drawing.lon(nodeIndex) - position.lon) < tolerance,
        `Node ${nodeIndex} longitude should match expected value`
      );
      assert(
        Math.abs(drawing.lat(nodeIndex) - position.lat) < tolerance,
        `Node ${nodeIndex} latitude should match expected value`
      );
    } else if (drawing instanceof eg.DrawingEuclidean) {
      for (let d = 0; d < position.length; d++) {
        assert(
          Math.abs(drawing.get(nodeIndex, d) - position[d]) < tolerance,
          `Node ${nodeIndex} coordinate at dimension ${d} should match expected value`
        );
      }
    }
  }
}

module.exports = {
  createTestGraph,
  createTestDiGraph,
  createStarGraph,
  createGridGraph,
  createDrawing,
  applyLayout,
  verifyLayoutQuality,
  verifyLayoutImprovement,
  verifyNodePositions,
  recordInitialPositions2d,
  recordInitialSphericalPositions,
  recordInitialPositionsNd,
  verifyPositionsChanged2d,
  verifySphericalPositionsChanged,
  verifyPositionsChangedNd,
  verifyFiniteCoordinates2d,
  verifyFiniteSphericalCoordinates,
  verifyFiniteCoordinatesNd,
  verifyTorusCoordinateRange,
  verifyHyperbolicCoordinateRange,
  verifySphericalCoordinateRange,
  createSeededRng,
  verifyConnectedNodesCloser,
  runScheduler,
  runSchedulerStepByStep,
};
