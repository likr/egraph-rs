import React, { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { Graph, nonEuclideanFruchtermanReingold } from "egraph";

async function fetchData() {
  const response = await fetch("/data/miserables.json");
  const data = await response.json();
  const color = d3.scaleOrdinal(d3.schemeCategory10);
  for (const node of data.nodes) {
    node.fillColor = color(node.group);
  }
  for (const link of data.links) {
    link.strokeWidth = Math.sqrt(link.value);
  }
  return data;
}

function layout(data) {
  const graph = new Graph();
  const indices = new Map();
  for (const node of data.nodes) {
    indices.set(node.id, graph.addNode(node));
  }
  for (const link of data.links) {
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
    0.5
  );
  for (const u of graph.nodeIndices()) {
    const node = graph.nodeWeight(u);
    const [x, y] = coordinates[u];
    node.x = x;
    node.y = y;
  }
}

function recenter(cx, cy, x, y) {
  const dx = x - cx;
  const dy = y - cy;
  const dr = 1 - cx * x - cy * y;
  const di = cy * x - cx * y;
  const d = dr * dr + di * di;
  return [(dr * dx + di * dy) / d, (dr * dy - di * dx) / d];
}

function scale(data, displayR) {
  const nodes = {};
  for (const node of data.nodes) {
    nodes[node.id] = node;
    node.x *= displayR;
    node.y *= displayR;
  }
  for (const link of data.links) {
    const { x: x1, y: y1 } = nodes[link.source];
    const { x: x2, y: y2 } = nodes[link.target];
    const b1 = (x1 * x1 + y1 * y1 + displayR * displayR) / 2;
    const b2 = (x2 * x2 + y2 * y2 + displayR * displayR) / 2;
    const d = x1 * y2 - y1 * x2;
    const cx = (link.cx = (y2 * b1 - y1 * b2) / d);
    const cy = (link.cy = (x1 * b2 - x2 * b1) / d);
    link.r = Math.sqrt(cx * cx + cy * cy - displayR * displayR);
    link.x1 = x1;
    link.y1 = y1;
    link.x2 = x2;
    link.y2 = y2;
  }
}

export function ExampleNonEuclideanForceSimulation() {
  const svgRef = useRef();
  const [data, setData] = useState({ nodes: [], links: [] });
  const displayR = 500;
  const margin = 50;

  useEffect(() => {
    const drag = d3.drag().on("drag", (event) => {
      const cx = -event.dx / displayR;
      const cy = -event.dy / displayR;
      setData((data) => {
        for (const node of data.nodes) {
          const [x, y] = recenter(cx, cy, node.x / displayR, node.y / displayR);
          node.x = x;
          node.y = y;
        }
        scale(data, displayR);
        return { ...data };
      });
    });
    d3.select(svgRef.current).call(drag);
  }, []);

  useEffect(() => {
    (async () => {
      const data = await fetchData();
      layout(data);
      scale(data, displayR);
      setData(data);
    })();
  }, []);

  const size = (displayR + margin) * 2;
  return (
    <div>
      <svg
        ref={svgRef}
        style={{ cursor: "move" }}
        viewBox={`${-margin} ${-margin} ${size} ${size}`}
      >
        <defs>
          {data.links.map((link) => {
            const path = d3.path();
            path.moveTo(0, 0);
            path.lineTo(link.x1, link.y1);
            path.lineTo(link.x2, link.y2);
            path.closePath();
            return (
              <clipPath
                key={`${link.source}:${link.target}`}
                id={`clip:${link.source}:${link.target}`}
              >
                <path
                  d={path.toString()}
                  stroke="#000"
                  strokeWidth={link.strokeWidth}
                />
              </clipPath>
            );
          })}
        </defs>
        <g transform={`translate(${displayR},${displayR})`}>
          <circle r={displayR} fill="none" stroke="#888" />
          <g>
            {data.links.map((link) => {
              return (
                <g key={`${link.source}:${link.target}`}>
                  <circle
                    cx={link.cx}
                    cy={link.cy}
                    r={link.r}
                    fill="none"
                    stroke="#999"
                    strokeWidth={link.strokeWidth}
                    opacity="0.6"
                    clipPath={`url(#clip:${link.source}:${link.target})`}
                  />
                </g>
              );
            })}
            {data.nodes.map((node) => {
              return (
                <g key={node.id} transform={`translate(${node.x},${node.y})`}>
                  <circle
                    r="5"
                    fill={node.fillColor}
                    stroke="#fff"
                    strokeWidth="1.5"
                  />
                  <text
                    className="is-unselectable"
                    textAnchor="middle"
                    dominantBaseline="central"
                  >
                    {node.name}
                  </text>
                </g>
              );
            })}
          </g>
        </g>
      </svg>
    </div>
  );
}
