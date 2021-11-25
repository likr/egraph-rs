import { useEffect, useRef } from "react";
import * as d3 from "d3";
import {
  Coordinates,
  Graph,
  Simulation,
  coarsen,
  ManyBodyForce,
  LinkForce,
} from "egraph";
import { Wrapper } from "../wrapper";

function layout(data) {
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

  const groups = new Map(
    graph.nodeIndices().map((u) => [u, graph.nodeWeight(u).group])
  );
  const coarsenedGraph = coarsen(
    graph,
    (u) => groups.get(u),
    (children) => ({ children }),
    (children) => ({ children })
  );

  const coordinates = Coordinates.initialPlacement(coarsenedGraph);
  const simulation = new Simulation();
  const forces = [
    new ManyBodyForce(coarsenedGraph),
    new LinkForce(coarsenedGraph),
  ];
  simulation.run((alpha) => {
    for (const force of forces) {
      force.apply(coordinates, alpha);
    }
    coordinates.updatePosition(0.6);
  });
  for (const u of coarsenedGraph.nodeIndices()) {
    const node = coarsenedGraph.nodeWeight(u);
    node.x = coordinates.x(u);
    node.y = coordinates.y(u);
  }
  console.log(JSON.stringify(coordinates.toJSON()));
  return {
    nodes: coarsenedGraph
      .nodeIndices()
      .map((u) => ({ id: u, ...coarsenedGraph.nodeWeight(u) })),
    links: coarsenedGraph.edgeIndices().map((e) => {
      const [u, v] = coarsenedGraph.edgeEndpoints(e);
      return {
        source: u,
        target: v,
        ...coarsenedGraph.edgeWeight(e),
      };
    }),
  };
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
          node.fillColor = color(node.id);
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
