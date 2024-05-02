import * as d3 from "d3";
import {
  DrawingEuclidean2d as Drawing,
  Graph,
  FullSgd as Sgd,
  Rng,
  fdeb,
} from "egraph/dist/web/egraph_wasm";
import { Wrapper } from "../wrapper";
import { useEffect, useRef } from "react";

export function ExampleEdgeBundling() {
  const rendererRef = useRef();
  useEffect(() => {
    window
      .fetch("/data/miserables.json")
      .then((response) => response.json())
      .then((data) => {
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

        const rng = Rng.seedFrom(0n);
        const drawing = Drawing.initialPlacement(graph);
        const sgd = new Sgd(graph, () => 100);
        const scheduler = sgd.scheduler(100, 0.1);
        scheduler.run((eta) => {
          sgd.shuffle(rng);
          sgd.applyWithDrawingEuclidean2d(drawing, eta);
        });

        for (const u of graph.nodeIndices()) {
          const node = graph.nodeWeight(u);
          node.x = drawing.x(u);
          node.y = drawing.y(u);
        }

        const bends = fdeb(graph, drawing);
        for (const e of graph.edgeIndices()) {
          graph.edgeWeight(e).bends = bends.get(e);
        }

        rendererRef.current.load(data);
        rendererRef.current.center();
      });
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
