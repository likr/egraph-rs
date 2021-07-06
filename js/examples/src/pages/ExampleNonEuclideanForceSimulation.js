import React, { useEffect, useRef } from "react";
import * as d3 from "d3";
import { Graph, nonEuclideanFruchtermanReingold } from "egraph";
import { Wrapper } from "../wrapper";

export function ExampleNonEuclideanForceSimulation() {
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

      const initialCoordinates = {};
      graph.nodeIndices().forEach((u, i) => {
        const r = 0.5;
        const t = (Math.PI * 2 * i) / graph.nodeCount();
        initialCoordinates[u] = [r * Math.cos(t), r * Math.sin(t)];
      });
      const coordinates = nonEuclideanFruchtermanReingold(
        graph,
        initialCoordinates,
        300,
        1
      );
      const displayR = 500;
      for (const u of graph.nodeIndices()) {
        const node = graph.nodeWeight(u);
        const [x, y] = coordinates[u];
        node.x = displayR * x;
        node.y = displayR * y;
      }
      data.groups = [
        { id: 1, x: 0, y: 0, width: 1000, height: 1000, type: "circle" },
      ];

      rendererRef.current.load(data);
      rendererRef.current.center();

      rendererRef.current.addEventListener("nodeclick", (event) => {
        const id = +event.detail.id;
        const centerNode = data.nodes.find((node) => node.id === id);
        const cx = centerNode.x / displayR;
        const cy = centerNode.y / displayR;
        for (const node of data.nodes) {
          const x = node.x / displayR;
          const y = node.y / displayR;
          const dx = x - cx;
          const dy = y - cy;
          const dr = 1 - cx * x - cy * y;
          const di = cy * x - cx * y;
          const d = dr * dr + di * di;
          node.x = ((dr * dx + di * dy) / d) * displayR;
          node.y = ((dr * dy - di * dx) / d) * displayR;
        }
        rendererRef.current.update();
      });
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
