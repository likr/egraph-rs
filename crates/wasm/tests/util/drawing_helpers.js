const assert = require("assert");

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

module.exports = {
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
};
