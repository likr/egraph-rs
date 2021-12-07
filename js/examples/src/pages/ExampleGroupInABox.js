import { useEffect, useRef } from "react";
import * as d3 from "d3";
import {
  Coordinates,
  Graph,
  Simulation,
  coarsen,
  ManyBodyForce,
  LinkForce,
  PositionForce,
  GroupManyBodyForce,
  GroupLinkForce,
  GroupPositionForce,
} from "egraph";
import { Wrapper } from "../wrapper";

function createGraph(data) {
  const graph = new Graph();
  const indices = new Map();
  for (const node of data.nodes) {
    indices.set(node.id, graph.addNode(node));
  }
  for (const link of data.links) {
    link.strokeWidth = Math.sqrt(link.value);
    const { source, target } = link;
    graph.addEdge(indices.get(source), indices.get(target), link);
  }
  return graph;
}

function groupLayout(graph) {
  const groups = new Set();
  const nodeGroups = {};
  for (const u of graph.nodeIndices()) {
    const { group } = graph.nodeWeight(u);
    groups.add(group);
    nodeGroups[u] = group;
  }
  const [coarsenedGraph, groupIds] = coarsen(
    graph,
    (u) => nodeGroups[u],
    (children) => ({ children }),
    (children) => ({ children })
  );

  const coordinates = Coordinates.initialPlacement(coarsenedGraph);
  const simulation = new Simulation();
  const forces = [
    new ManyBodyForce(coarsenedGraph, () => ({ strength: -3000 })),
    new LinkForce(coarsenedGraph, () => ({ distance: 100 })),
    new PositionForce(coarsenedGraph, () => ({ x: 0, y: 0, strength: 0.1 })),
  ];
  simulation.run((alpha) => {
    for (const force of forces) {
      force.apply(coordinates, alpha);
    }
    coordinates.updatePosition(0.6);
  });

  const result = {};
  for (const g of groups) {
    const u = groupIds[g];
    const group = coarsenedGraph.nodeWeight(u);
    result[g] = {
      id: g,
      x: coordinates.x(u),
      y: coordinates.y(u),
      width: 20 * Math.sqrt(group.children.length),
      height: 20 * Math.sqrt(group.children.length),
      type: "circle",
    };
  }
  return result;
}

function layout(data) {
  const graph = createGraph(data);
  const groups = groupLayout(graph);

  const coordinates = Coordinates.initialPlacement(graph);
  const simulation = new Simulation();
  const forces = [
    new GroupManyBodyForce(graph, (u) => {
      const node = graph.nodeWeight(u);
      return { group: node.group, strength: -30 };
    }),
    new GroupLinkForce(
      graph,
      (u) => {
        const node = graph.nodeWeight(u);
        return { group: node.group };
      },
      (e) => {
        return {
          distance: 30,
          strength: 0.01,
        };
      },
      (e) => {
        return {
          distance: 30,
          strength: 0.001,
        };
      }
    ),
    new GroupPositionForce(
      graph,
      (u) => {
        const node = graph.nodeWeight(u);
        return { group: node.group, strength: 0.8 };
      },
      (g) => {
        return groups[g];
      }
    ),
  ];
  simulation.run((alpha) => {
    for (const force of forces) {
      force.apply(coordinates, alpha);
    }
    coordinates.updatePosition(0.6);
  });
  for (const u of graph.nodeIndices()) {
    const node = graph.nodeWeight(u);
    node.x = coordinates.x(u);
    node.y = coordinates.y(u);
  }
  data.groups = Object.values(groups);
  return data;
}

export function ExampleGroupInABox() {
  const rendererRef = useRef();

  useEffect(() => {
    window
      .fetch("/data/miserables.json")
      .then((response) => response.json())
      .then((data) => {
        const result = layout(data);
        const color = d3.scaleOrdinal(d3.schemeCategory10);
        for (const node of result.nodes) {
          node.fillColor = color(node.group);
        }
        rendererRef.current.load(result);
        rendererRef.current.center();
      });
  });

  return (
    <div>
      <div>
        <Wrapper
          onResize={(width, height) => {
            rendererRef.current.width = width;
            rendererRef.current.height = height;
          }}
        >
          <eg-renderer
            ref={rendererRef}
            transition-duration="1000"
            default-node-width="10"
            default-node-height="10"
            default-node-stroke-color="#fff"
            default-node-stroke-width="1.5"
            default-link-stroke-color="#999"
            default-link-stroke-opacity="0.6"
            no-auto-update
            no-auto-centering
          />
        </Wrapper>
      </div>
    </div>
  );
}
