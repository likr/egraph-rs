const eg = require("wasm-bindgen-test");

/**
 * Creates a line (path) graph where each node connects to the next
 * @param {number} size - Number of nodes
 * @returns {Object} Object containing graph and nodes array
 */
function createLineGraph(size = 3) {
  const graph = new eg.Graph();
  const nodes = [];

  // Create nodes
  for (let i = 0; i < size; i++) {
    nodes.push(graph.addNode({ id: i }));
  }

  // Create edges
  for (let i = 0; i < size - 1; i++) {
    graph.addEdge(nodes[i], nodes[i + 1], {});
  }

  return { graph, nodes };
}

/**
 * Creates a directed line (path) graph where each node connects to the next
 * @param {number} size - Number of nodes
 * @returns {Object} Object containing graph and nodes array
 */
function createLineDiGraph(size = 3) {
  const graph = new eg.DiGraph();
  const nodes = [];

  // Create nodes
  for (let i = 0; i < size; i++) {
    nodes.push(graph.addNode({ id: i }));
  }

  // Create edges
  for (let i = 0; i < size - 1; i++) {
    graph.addEdge(nodes[i], nodes[i + 1], {});
  }

  return { graph, nodes };
}

/**
 * Creates a cycle graph where nodes form a circular path
 * @param {number} size - Number of nodes
 * @returns {Object} Object containing graph and nodes array
 */
function createCycleGraph(size = 3) {
  const graph = new eg.Graph();
  const nodes = [];

  // Create nodes
  for (let i = 0; i < size; i++) {
    nodes.push(graph.addNode({ id: i }));
  }

  // Create edges
  for (let i = 0; i < size; i++) {
    graph.addEdge(nodes[i], nodes[(i + 1) % size], {});
  }

  return { graph, nodes };
}

/**
 * Creates a directed cycle graph where nodes form a circular path
 * @param {number} size - Number of nodes
 * @returns {Object} Object containing graph and nodes array
 */
function createCycleDiGraph(size = 3) {
  const graph = new eg.DiGraph();
  const nodes = [];

  // Create nodes
  for (let i = 0; i < size; i++) {
    nodes.push(graph.addNode({ id: i }));
  }

  // Create edges
  for (let i = 0; i < size; i++) {
    graph.addEdge(nodes[i], nodes[(i + 1) % size], {});
  }

  return { graph, nodes };
}

/**
 * Creates a complete graph where every node connects to every other node
 * @param {number} size - Number of nodes
 * @returns {Object} Object containing graph and nodes array
 */
function createCompleteGraph(size = 3) {
  const graph = new eg.Graph();
  const nodes = [];

  // Create nodes
  for (let i = 0; i < size; i++) {
    nodes.push(graph.addNode({ id: i }));
  }

  // Create edges
  for (let i = 0; i < size; i++) {
    for (let j = i + 1; j < size; j++) {
      graph.addEdge(nodes[i], nodes[j], {});
    }
  }

  return { graph, nodes };
}

/**
 * Creates a directed complete graph where every node connects to every other node
 * @param {number} size - Number of nodes
 * @returns {Object} Object containing graph and nodes array
 */
function createCompleteDiGraph(size = 3) {
  const graph = new eg.DiGraph();
  const nodes = [];

  // Create nodes
  for (let i = 0; i < size; i++) {
    nodes.push(graph.addNode({ id: i }));
  }

  // Create edges
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      if (i !== j) {
        graph.addEdge(nodes[i], nodes[j], {});
      }
    }
  }

  return { graph, nodes };
}

/**
 * Creates a triangle graph (3 nodes in a cycle)
 * @returns {Object} Object containing graph and nodes array
 */
function createTriangleGraph() {
  const graph = new eg.Graph();
  const nodes = [];

  // Create nodes
  for (let i = 0; i < 3; i++) {
    nodes.push(graph.addNode({ id: i }));
  }

  // Create edges
  graph.addEdge(nodes[0], nodes[1], {});
  graph.addEdge(nodes[1], nodes[2], {});
  graph.addEdge(nodes[2], nodes[0], {});

  return { graph, nodes };
}

/**
 * Creates a directed triangle graph (3 nodes in a cycle)
 * @returns {Object} Object containing graph and nodes array
 */
function createTriangleDiGraph() {
  const graph = new eg.DiGraph();
  const nodes = [];

  // Create nodes
  for (let i = 0; i < 3; i++) {
    nodes.push(graph.addNode({ id: i }));
  }

  // Create edges
  graph.addEdge(nodes[0], nodes[1], {});
  graph.addEdge(nodes[1], nodes[2], {});
  graph.addEdge(nodes[2], nodes[0], {});

  return { graph, nodes };
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
 * Creates a directed star graph with a central node connected to all other nodes
 * @param {number} size - Number of nodes (including the central node)
 * @param {boolean} outward - If true, edges go from center to periphery; if false, from periphery to center
 * @returns {Object} Object containing graph and nodes array
 */
function createStarDiGraph(size = 5, outward = true) {
  const graph = new eg.DiGraph();
  const nodes = [];

  // Create nodes
  for (let i = 0; i < size; i++) {
    nodes.push(graph.addNode({ id: i }));
  }

  // Create edges between central node (0) and all other nodes
  for (let i = 1; i < size; i++) {
    if (outward) {
      // Edges from center to periphery
      graph.addEdge(nodes[0], nodes[i], {});
    } else {
      // Edges from periphery to center
      graph.addEdge(nodes[i], nodes[0], {});
    }
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
 * Creates a directed grid graph with the specified width and height
 * @param {number} width - Number of nodes in the horizontal direction
 * @param {number} height - Number of nodes in the vertical direction
 * @returns {Object} Object containing graph and nodes array (as a 2D array)
 */
function createGridDiGraph(width = 3, height = 3) {
  const graph = new eg.DiGraph();
  const nodes = [];

  // Create nodes
  for (let y = 0; y < height; y++) {
    const row = [];
    for (let x = 0; x < width; x++) {
      row.push(graph.addNode({ id: y * width + x, x, y }));
    }
    nodes.push(row);
  }

  // Create horizontal edges (left to right)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width - 1; x++) {
      graph.addEdge(nodes[y][x], nodes[y][x + 1], {});
    }
  }

  // Create vertical edges (top to bottom)
  for (let y = 0; y < height - 1; y++) {
    for (let x = 0; x < width; x++) {
      graph.addEdge(nodes[y][x], nodes[y + 1][x], {});
    }
  }

  return { graph, nodes };
}

module.exports = {
  createLineGraph,
  createLineDiGraph,
  createCycleGraph,
  createCycleDiGraph,
  createCompleteGraph,
  createCompleteDiGraph,
  createTriangleGraph,
  createTriangleDiGraph,
  createStarGraph,
  createStarDiGraph,
  createGridGraph,
  createGridDiGraph,
  createSeededRng,
  runScheduler,
  runSchedulerStepByStep,
};
