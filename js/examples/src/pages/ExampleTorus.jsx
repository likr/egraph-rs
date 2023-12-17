import React, { useMemo, useState } from "react";
import * as d3 from "d3";
import {
  Graph,
  DrawingTorus,
  FullSgd as Sgd,
  Rng,
} from "egraph/dist/web/egraph_wasm";
// import data from "egraph-dataset/les_miserables.json";
// import data from "egraph-dataset/dodecahedral.json";
import data from "egraph-dataset/moebius_kantor.json";

export function ExampleTorus() {
  const width = 1000;
  const height = 1000;
  const [offset, setOffset] = useState([0, 0]);
  const layout = useMemo(() => {
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
    const drawing = DrawingTorus.initialPlacement(graph);
    const sgd = new Sgd(graph, () => 0.2);
    const scheduler = sgd.scheduler(100, 0.1);
    scheduler.run((eta) => {
      sgd.shuffle(rng);
      sgd.apply(drawing, eta);
    });
    for (const u of graph.nodeIndices()) {
      const node = graph.nodeWeight(u);
      node.x = drawing.x(u) * width;
      node.y = drawing.y(u) * height;
    }
    return data;
  }, []);
  const pos = {};
  for (const node of layout.nodes) {
    pos[node.id] = {
      x: (node.x - offset[0] + width / 2 + width) % (width + 1),
      y: (node.y - offset[1] + width / 2 + width) % (height + 1),
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
            const x0 = pos[link.source].x;
            const y0 = pos[link.source].y;
            const x1 = pos[link.target].x;
            const y1 = pos[link.target].y;
            let path = d3.path();
            path.moveTo(x0, y0);
            path.lineTo(x1, y1);
            let d = Math.hypot(x1 - x0, y1 - y0);

            if (Math.hypot(x1 - width - x0, y1 - y0) < d) {
              const a = (y0 - y1) / (x0 - x1 + width);
              const y2 = y0 - x0 * a;
              path = d3.path();
              path.moveTo(0, y2);
              path.lineTo(x0, y0);
              path.moveTo(x1, y1);
              path.lineTo(width, y2);
              d = Math.hypot(x1 - width - x0, y1 - y0);
            }
            if (Math.hypot(x1 + width - x0, y1 - y0) < d) {
              const a = (y1 - y0) / (x1 + width - x0);
              const y2 = y1 - x1 * a;
              path = d3.path();
              path.moveTo(0, y2);
              path.lineTo(x1, y1);
              path.moveTo(x0, y0);
              path.lineTo(width, y2);
              d = Math.hypot(x1 + width - x0, y1 - y0);
            }
            if (Math.hypot(x1 - x0, y1 - height - y0) < d) {
              const a = (x0 - x1) / (y0 - y1 + height);
              const x2 = x0 - y0 * a;
              path = d3.path();
              path.moveTo(x2, 0);
              path.lineTo(x0, y0);
              path.moveTo(x1, y1);
              path.lineTo(x2, height);
              d = Math.hypot(x1 - x0, y1 - height - y0);
            }
            if (Math.hypot(x1 - x0, y1 + height - y0) < d) {
              const a = (x1 - x0) / (y1 + height - y0);
              const x2 = x1 - y1 * a;
              path = d3.path();
              path.moveTo(x2, 0);
              path.lineTo(x1, y1);
              path.moveTo(x0, y0);
              path.lineTo(x2, height);
              d = Math.hypot(x1 - x0, y1 + height - y0);
            }
            if (Math.hypot(x1 - width - x0, y1 - height - y0) < d) {
              const a = (y0 - y1 + height) / (x0 - x1 + width);
              const y2 = y0 - x0 * a;
              const x2 = -y2 / a;
              if (y2 < 0) {
                path = d3.path();
                path.moveTo(x2, 0);
                path.lineTo(x0, y0);
                path.moveTo(x1, y1);
                path.lineTo(width, y2 + height);
                path.moveTo(0, y2 + height);
                path.lineTo(x2, height);
              } else {
                path = d3.path();
                path.moveTo(0, y2);
                path.lineTo(x0, y0);
                path.moveTo(x1, y1);
                path.lineTo(x2 + width, height);
                path.moveTo(x2 + width, 0);
                path.lineTo(width, y2);
              }
              d = Math.hypot(x1 - width - x0, y1 - height - y0);
            }
            if (Math.hypot(x1 - width - x0, y1 + height - y0) < d) {
              const a = (y0 - y1 - height) / (x0 - x1 + width);
              const y2 = y0 - x0 * a;
              const x2 = (height - y2) / a;
              if (y2 > height) {
                path = d3.path();
                path.moveTo(x2, height);
                path.lineTo(x0, y0);
                path.moveTo(x1, y1);
                path.lineTo(width, y2 - height);
                path.moveTo(x2, 0);
                path.lineTo(0, y2 - height);
              } else {
                path = d3.path();
                path.moveTo(0, y2);
                path.lineTo(x0, y0);
                path.moveTo(x1, y1);
                path.lineTo(x2 + width, 0);
                path.moveTo(x2 + width, height);
                path.lineTo(width, y2);
              }
              d = Math.hypot(x1 - width - x0, y1 + height - y0);
            }
            if (Math.hypot(x1 + width - x0, y1 - height - y0) < d) {
              console.log("moge");
              const a = (y1 - height - y0) / (x1 + width - x0);
              const y2 = y1 - x1 * a;
              const x2 = -y2 / a;
              if (y2 > height) {
                path = d3.path();
                path.moveTo(x2, height);
                path.lineTo(x1, y1);
                path.moveTo(x0, y0);
                path.lineTo(width, y2 - height);
                path.moveTo(x2, 0);
                path.lineTo(0, y2 - height);
              } else {
                path = d3.path();
                path.moveTo(0, y2);
                path.lineTo(x1, y1);
                path.moveTo(x0, y0);
                path.lineTo(x2 + width, 0);
                path.moveTo(x2 + width, height);
                path.lineTo(width, y2);
              }
              d = Math.hypot(x1 + width - x0, y1 - height - y0);
            }
            if (Math.hypot(x1 + width - x0, y1 + height - y0) < d) {
              const a = (y1 + height - y0) / (x1 + width - x0);
              const y2 = y1 - x1 * a;
              const x2 = -y2 / a;
              if (y2 < 0) {
                path = d3.path();
                path.moveTo(x2, 0);
                path.lineTo(x1, y1);
                path.moveTo(x0, y0);
                path.lineTo(width, y2 + height);
                path.moveTo(0, y2 + height);
                path.lineTo(x2, height);
              } else {
                path = d3.path();
                path.moveTo(0, y2);
                path.lineTo(x1, y1);
                path.moveTo(x0, y0);
                path.lineTo(x2 + width, height);
                path.moveTo(x2 + width, 0);
                path.lineTo(width, y2);
              }
              d = Math.hypot(x1 + width - x0, y1 + height - y0);
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
