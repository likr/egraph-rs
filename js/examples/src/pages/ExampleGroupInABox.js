import { useEffect, useRef } from "react";
import * as d3 from "d3";
import { Coordinates, Graph, Simulation } from "egraph";
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

  const coordinates = Coordinates.initialPlacement(graph);
  const simulation = new Simulation();
  const forces = [];
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
}

export function ExampleGroupInABox() {
  const rendererRef = useRef();

  useEffect(() => {
    window
      .fetch("/data/miserables.json")
      .then((response) => response.json())
      .then((data) => {
        const color = d3.scaleOrdinal(d3.schemeCategory10);
        for (const node of data.nodes) {
          node.fillColor = color(node.group);
        }
        layout(data);
        rendererRef.current.load(data);
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
            node-label-property="name"
            no-auto-update
            no-auto-centering
          />
        </Wrapper>
      </div>
    </div>
  );
}
