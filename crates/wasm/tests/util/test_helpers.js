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

module.exports = {
  createTestGraph,
  createTestDiGraph,
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
