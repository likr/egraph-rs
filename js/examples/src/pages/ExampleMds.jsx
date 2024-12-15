import React, { useEffect, useRef } from "react";
import * as d3 from "d3";
import { Graph, ClassicalMds } from "egraph/dist/web/egraph_wasm";
import { Wrapper } from "../wrapper";

export function ExampleMds() {
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

      const drawing = new ClassicalMds(graph, () => 100).run2d();
      for (const u of graph.nodeIndices()) {
        const node = graph.nodeWeight(u);
        node.x = drawing.x(u);
        node.y = drawing.y(u);
      }

      rendererRef.current.load(data);
      rendererRef.current.focus(0, 0);
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
