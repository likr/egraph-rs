const assert = require("assert");
const eg = require("wasm-bindgen-test");
const drawingHelpers = require("./drawing_helpers");

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
    drawingHelpers.verifyFiniteCoordinates2d(drawing, graph);
  } else if (drawing instanceof eg.DrawingSpherical2d) {
    drawingHelpers.verifyFiniteSphericalCoordinates(drawing, graph);
  } else if (drawing instanceof eg.DrawingEuclidean) {
    const dimensions = options.dimensions || 3;
    drawingHelpers.verifyFiniteCoordinatesNd(drawing, graph, dimensions);
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
    drawingHelpers.verifyTorusCoordinateRange(drawing, graph);
  } else if (drawing instanceof eg.DrawingHyperbolic2d) {
    drawingHelpers.verifyHyperbolicCoordinateRange(drawing, graph);
  } else if (drawing instanceof eg.DrawingSpherical2d) {
    drawingHelpers.verifySphericalCoordinateRange(drawing, graph);
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
  verifyConnectedNodesCloser,
  verifyLayoutQuality,
  verifyLayoutImprovement,
  verifyNodePositions,
};
