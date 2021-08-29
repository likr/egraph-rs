import { useEffect, useRef } from "react";
import * as d3 from "d3";
import {
  Coordinates,
  Graph,
  Simulation,
  FruchtermanReingoldForce,
  PositionForce,
} from "egraph";
import { Wrapper } from "../wrapper";

export function ExampleFruchtermanReingold() {
  const rendererRef = useRef();

  useEffect(() => {
    (async () => {
      const response = await fetch("/data/miserables.json");
      const data = await response.json();
      const color = d3.scaleOrdinal(d3.schemeCategory10);
      const graph = new Graph();
      const indices = new Map();
      for (const node of data.nodes) {
        node.fillColor = color(node.group);
        indices.set(node.id, graph.addNode(node));
      }
      for (const link of data.links) {
        link.strokeWidth = Math.sqrt(link.value);
        const { source, target } = link;
        graph.addEdge(indices.get(source), indices.get(target), link);
      }

      const coordinates = Coordinates.initialPlacement(graph);
      const simulation = new Simulation();
      const forces = [
        new FruchtermanReingoldForce(graph, 30, 1),
        new PositionForce(graph, () => ({ x: 0, y: 0 })),
      ];
      simulation.runStep(300, (alpha) => {
        console.log(alpha, coordinates.x(0), coordinates.vx(0));
        for (const force of forces) {
          force.apply(coordinates, alpha);
        }
        coordinates.updatePosition(0.6);
        coordinates.updatePosition(0.0);
        coordinates.clampRegion(-500, -500, 500, 500);
      });
      for (const u of graph.nodeIndices()) {
        const node = graph.nodeWeight(u);
        node.x = coordinates.x(u);
        node.y = coordinates.y(u);
        console.log(node.x, node.y);
      }

      rendererRef.current.load(data);
      rendererRef.current.center();
    })();
  }, []);

  return (
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
        default-node-type="circle"
        default-link-stroke-color="#999"
        default-link-stroke-opacity="0.6"
        node-id-property="id"
        node-label-property="name"
        no-auto-centering
      />
    </Wrapper>
  );
}
