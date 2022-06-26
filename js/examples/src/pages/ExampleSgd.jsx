import React, { useEffect, useRef } from "react";
import * as d3 from "d3";
import {
  Graph,
  Coordinates,
  Simulation,
  Sgd,
} from "egraph/dist/web/egraph_wasm";
import data from "egraph-dataset/1138_bus.json";
import { Wrapper } from "../wrapper";

export function ExampleSgd() {
  const rendererRef = useRef();

  useEffect(() => {
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
    const sgd = new Sgd(graph, () => 30);

    const draw = () => {
      if (!rendererRef.current) {
        return;
      }
      sgd.apply(coordinates);
      for (const u of graph.nodeIndices()) {
        const node = graph.nodeWeight(u);
        node.x = coordinates.x(u);
        node.y = coordinates.y(u);
      }
      rendererRef.current.update();
      window.requestAnimationFrame(draw);
    };

    rendererRef.current.load(data);
    draw();
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
