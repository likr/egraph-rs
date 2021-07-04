import React, { useEffect, useRef } from "react";
import * as d3 from "d3";
import { Graph, initialPlacement, stressMajorization } from "egraph";
import { Wrapper } from "../wrapper";

export function ExampleStressMajorization() {
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

      const initialCoordinates = initialPlacement(graph);
      const coordinates = stressMajorization(graph, initialCoordinates, () => ({
        distance: 100,
      }));
      for (const u of graph.nodeIndices()) {
        const node = graph.nodeWeight(u);
        const [x, y] = coordinates[u];
        node.x = x;
        node.y = y;
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
        node-label-property="name"
        no-auto-centering
      />
    </Wrapper>
  );
}
