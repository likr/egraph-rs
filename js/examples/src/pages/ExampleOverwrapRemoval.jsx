import { useEffect, useRef } from "react";
import * as d3 from "d3";
import {
  Graph,
  DrawingEuclidean2d as Drawing,
  OverwrapRemoval,
  FullSgd as Sgd,
  Rng,
} from "egraph/dist/web/egraph_wasm";
import data from "egraph-dataset/les_miserables.json";
import { Wrapper } from "../wrapper";

export function ExampleOverwrapRemoval() {
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
      const { source, target } = link;
      graph.addEdge(indices.get(source), indices.get(target), link);
    }

    const rng = Rng.seedFrom(4n);
    const drawing = Drawing.initialPlacement(graph);
    const sgd = new Sgd(graph, () => 100);
    const scheduler = sgd.scheduler(100, 0.1);
    const overwrapRemoval = new OverwrapRemoval(graph, () => 25);
    overwrapRemoval.iterations = 5;

    const draw = () => {
      if (!rendererRef.current || scheduler.isFinished()) {
        return;
      }
      scheduler.step((eta) => {
        sgd.shuffle(rng);
        sgd.applyWithDrawingEuclidean2d(drawing, eta);
        overwrapRemoval.applyWithDrawingEuclidean2d(drawing);
      });
      drawing.centralize();
      for (const u of graph.nodeIndices()) {
        const node = graph.nodeWeight(u);
        node.x = drawing.x(u);
        node.y = drawing.y(u);
      }
      rendererRef.current.update();
      window.requestAnimationFrame(draw);
    };

    rendererRef.current.load(data);
    rendererRef.current.focus(0, 0);
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
        default-node-width="50"
        default-node-height="50"
        default-node-fill-opacity="0.5"
        default-node-stroke-color="#fff"
        default-node-stroke-width="0"
        default-node-type="circle"
        default-link-stroke-color="#999"
        default-link-stroke-opacity="0.6"
        default-link-stroke-width="3"
        node-id-property="id"
        node-label-property="name"
        no-auto-centering
      />
    </Wrapper>
  );
}
