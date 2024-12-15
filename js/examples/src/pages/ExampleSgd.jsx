import React, { useEffect, useRef } from "react";
import * as d3 from "d3";
import {
  Graph,
<<<<<<< HEAD
  Drawing,
  DistanceAdjustedSparseSgd as Sgd,
  Rng,
} from "egraph/dist/web/egraph_wasm";
import data from "egraph-dataset/dwt_1005.json";
=======
  DrawingEuclidean2d as Drawing,
  SparseSgd as Sgd,
  Rng,
} from "egraph/dist/web/egraph_wasm";
import data from "egraph-dataset/USpowerGrid.json";
>>>>>>> efe22487a57cab11bc0ac1f51c85869669387983
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
      const { source, target } = link;
      graph.addEdge(indices.get(source), indices.get(target), link);
    }

    const rng = Rng.seedFrom(1n);
    const drawing = Drawing.initialPlacement(graph);
<<<<<<< HEAD
    const sgd = new Sgd(graph, () => 1, 300, rng);
    sgd.alpha = 1 - 0.5 ** 5;
    sgd.minimumDistance = 0.5;
    const scheduler = sgd.scheduler(15, 0.1);
=======
    const sgd = new Sgd(graph, () => 20, 200, rng);
    const scheduler = sgd.scheduler(100, 0.1);
>>>>>>> efe22487a57cab11bc0ac1f51c85869669387983

    const draw = () => {
      if (!rendererRef.current || scheduler.isFinished()) {
        return;
      }
      scheduler.step((eta) => {
        sgd.shuffle(rng);
<<<<<<< HEAD
        sgd.applyWithDistanceAdjustment(drawing, eta);
=======
        sgd.applyWithDrawingEuclidean2d(drawing, eta);
>>>>>>> efe22487a57cab11bc0ac1f51c85869669387983
      });
      drawing.centralize();
      for (const u of graph.nodeIndices()) {
        const node = graph.nodeWeight(u);
        node.x = drawing.x(u) * 20;
        node.y = drawing.y(u) * 20;
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
        default-node-width="5"
        default-node-height="5"
        default-node-stroke-color="#fff"
        default-node-stroke-width="1"
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
