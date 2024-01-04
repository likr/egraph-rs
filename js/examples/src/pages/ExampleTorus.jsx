import React, { useMemo, useState } from "react";
import * as d3 from "d3";
import * as eg from "egraph/dist/web/egraph_wasm";
// import data from "egraph-dataset/les_miserables.json";
// import data from "egraph-dataset/dodecahedral.json";
import data from "egraph-dataset/moebius_kantor.json";

export function ExampleTorus() {
  const width = 1000;
  const height = 1000;
  const [offset, setOffset] = useState([-width / 2, -width / 2]);
  const layout = useMemo(() => {
    const color = d3.scaleOrdinal(d3.schemeCategory10);
    const graph = new eg.Graph();
    const indices = new Map();
    for (const node of data.nodes) {
      node.fillColor = color(node.group);
      indices.set(node.id, graph.addNode(node));
    }
    for (const link of data.links) {
      const { source, target } = link;
      graph.addEdge(indices.get(source), indices.get(target), link);
    }

    const rng = eg.Rng.seedFrom(1n);
    const drawing = eg.DrawingTorus.initialPlacement(graph);
    const sgd = new eg.FullSgd(graph, () => 0.2);
    const scheduler = sgd.scheduler(20, 0.1);
    scheduler.run((eta) => {
      sgd.shuffle(rng);
      sgd.apply(drawing, eta);
    });
    for (const u of graph.nodeIndices()) {
      const node = graph.nodeWeight(u);
      node.x = drawing.x(u) * width;
      node.y = drawing.y(u) * height;
    }
    for (const e of graph.edgeIndices()) {
      const edge = graph.edgeWeight(e);
      const [u, v] = graph.edgeEndpoints(e);
      edge.segments = drawing.edgeSegments(u, v).map(([[x1, y1], [x2, y2]]) => [
        [x1 * width, y1 * height],
        [x2 * width, y2 * height],
      ]);
    }
    return data;
  }, []);
  const pos = {};
  for (const node of layout.nodes) {
    pos[node.id] = {
      x: node.x,
      y: node.y,
    };
  }

  return (
    <figure
      className="image is-1by1"
      style={{
        boxShadow: "0 0 1em",
      }}
    >
      <svg viewBox={`0 0 ${width} ${height}`}>
        <g>
          {layout.links.map((link) => {
            const path = d3.path();
            for (const segment of link.segments) {
              path.moveTo(segment[0][0], segment[0][1]);
              path.lineTo(segment[1][0], segment[1][1]);
            }
            return (
              <g key={`${link.source}:${link.target}`}>
                <path d={path} fill="none" stroke="#888" />
              </g>
            );
          })}
        </g>
        <g>
          {layout.nodes.map((node) => {
            const dxdy = [];
            for (let i = -1; i <= 1; ++i) {
              for (let j = -1; j <= 1; ++j) {
                dxdy.push([width * i, height * j]);
              }
            }
            return (
              <g key={node.id}>
                {dxdy.map(([dx, dy], i) => {
                  return (
                    <circle
                      key={i}
                      className="is-clickable"
                      style={{
                        transition: "transform 1s",
                      }}
                      transform={`translate(${pos[node.id].x + dx},${
                        pos[node.id].y + dy
                      })`}
                      r="10"
                      opacity="0.5"
                      onClick={() => {
                        setOffset([node.x, node.y]);
                      }}
                    />
                  );
                })}
              </g>
            );
          })}
        </g>
      </svg>
    </figure>
  );
}
