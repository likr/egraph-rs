import * as d3 from "d3";
import { CsrGraph, loadDataset } from "egraph";

const palette = d3.scaleOrdinal(d3.schemeCategory10);

export async function loadMiserablesGraph() {
  const response = await fetch("/data/miserables.json");
  const data = await response.json();
  const idToIndex = new Map();
  const nodes = data.nodes.map((node, i) => {
    idToIndex.set(node.id, i);
    return {
      index: i,
      id: node.id,
      name: node.name ?? String(node.id),
      group: node.group ?? 0,
      color: palette(node.group ?? 0),
    };
  });

  const edgePairs = [];
  const links = [];
  for (const link of data.links) {
    const u = idToIndex.get(link.source);
    const v = idToIndex.get(link.target);
    if (u !== undefined && v !== undefined) {
      edgePairs.push(u, v);
      links.push({
        source: u,
        target: v,
        value: link.value ?? 1,
        strokeWidth: Math.max(0.8, Math.sqrt(link.value ?? 1) * 0.7),
      });
    }
  }

  const edges = new Uint32Array(edgePairs);
  const graph = CsrGraph.fromEdgeList(nodes.length, edges, null, false, true);
  return { graph, edges, nodes, links, numNodes: nodes.length };
}

export function loadEmbeddedGraph(name) {
  const raw = loadDataset(name);
  const lcc = raw.largestConnectedComponent();
  const graph = lcc.graph();
  lcc.free();
  raw.free();

  const numNodes = graph.numNodes();
  const indptr = graph.indptr();
  const indices = graph.indices();
  const edgePairs = [];
  const links = [];
  for (let u = 0; u < numNodes; u++) {
    for (let k = indptr[u]; k < indptr[u + 1]; k++) {
      const v = indices[k];
      if (u < v) {
        edgePairs.push(u, v);
        links.push({ source: u, target: v, value: 1, strokeWidth: 0.7 });
      }
    }
  }
  const nodes = Array.from({ length: numNodes }, (_, i) => ({
    index: i,
    id: i,
    name: String(i),
    group: i % 10,
    color: palette(i % 10),
  }));
  return {
    graph,
    edges: new Uint32Array(edgePairs),
    nodes,
    links,
    numNodes,
  };
}

export function createManifoldRunner(solver, isFinishedFn = () => solver.isFinished) {
  return {
    get isFinished() {
      return isFinishedFn();
    },
    get iteration() {
      return solver.iteration;
    },
    get totalIterations() {
      return solver.totalIterations;
    },
    get status() {
      return solver.status;
    },
    step(layout) {
      return solver.stepManifold(layout);
    },
    runSteps(layout, k) {
      return solver.runStepsManifold(layout, k);
    },
  };
}
