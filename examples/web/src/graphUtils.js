import * as d3 from "d3";
import { CsrGraph } from "egraph";

const palette = d3.scaleOrdinal(d3.schemeCategory10);

const DATASET_LOADERS = {
  lesmis: () => import("../dataset/les_miserables.json"),
  karate: () => import("../dataset/karate_club.json"),
  "1138_bus": () => import("../dataset/1138_bus.json"),
  "3_elt": () => import("../dataset/3elt.json"),
};

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
  const graph = CsrGraph.fromEdgeList(nodes.length, edges, null, {
    directed: false,
    sanitize: true,
  });
  return { graph, edges, nodes, links, numNodes: nodes.length };
}

export async function loadEmbeddedGraph(name) {
  const loader = DATASET_LOADERS[name];
  if (!loader) {
    throw new Error(`Unknown dataset: ${name}`);
  }
  const mod = await loader();
  const data = mod.default ?? mod;
  const idToIndex = new Map();
  data.nodes.forEach((node, i) => {
    idToIndex.set(node.id, i);
  });

  const rawEdgePairs = [];
  for (const link of data.links) {
    const u = idToIndex.get(link.source);
    const v = idToIndex.get(link.target);
    if (u !== undefined && v !== undefined) {
      rawEdgePairs.push(u, v);
    }
  }

  const raw = CsrGraph.fromEdgeList(
    data.nodes.length,
    new Uint32Array(rawEdgePairs),
    null,
    { directed: false, sanitize: true },
  );
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
