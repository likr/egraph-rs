import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import {
  DrawingSpherical2d as Drawing,
  Graph,
  FullSgd as Sgd,
  Rng,
} from "egraph/dist/web/egraph_wasm";

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

  const rng = Rng.seedFrom(4n);
  const drawing = Drawing.initialPlacement(graph);
  const sgd = new Sgd(graph, () => 1);
  const scheduler = sgd.scheduler(100, 0.1);
  scheduler.run((eta) => {
    sgd.shuffle(rng);
    sgd.applyWithDrawingSpherical2d(drawing, eta);
  });

  for (const u of graph.nodeIndices()) {
    const node = graph.nodeWeight(u);
    node.lon = (180 * (drawing.lon(u) - Math.PI)) / Math.PI;
    node.lat = (180 * (drawing.lat(u) - Math.PI / 2)) / Math.PI;
  }
}

export function ExampleSphericalGeometry() {
  const svgRef = useRef();
  const [rotate, setRotate] = useState([0, 0]);
  const [data, setData] = useState({ nodes: [], links: [] });
  const displayR = 500;
  const margin = 50;

  useEffect(() => {
    (async () => {
      const data = await fetchData();
      layout(data);
      setData(data);
    })();
  }, []);

  useEffect(() => {
    const drag = d3.drag().on("drag", (event) => {
      setRotate((rotate) => [rotate[0] - event.dx, rotate[1] - event.dy]);
    });
    d3.select(svgRef.current).call(drag);
  }, []);

  const size = (displayR + margin) * 2;

  const projection = d3.geoAzimuthalEquidistant().rotate(rotate);
  const path = d3.geoPath(projection);
  return (
    <div>
      <svg
        ref={svgRef}
        style={{ cursor: "move" }}
        viewBox={`${-margin} ${-margin} ${size} ${size}`}
      >
        <g>
          <g>
            <path
              d={path(d3.geoGraticule10())}
              fill="none"
              stroke="#888"
              opacity="0.3"
            />
          </g>
          <g>
            {data.links.map((link) => {
              return (
                <g key={`${link.source}:${link.target}`}>
                  <path
                    d={path({
                      type: "LineString",
                      coordinates: [
                        [
                          data.nodes[link.source].lon,
                          data.nodes[link.source].lat,
                        ],
                        [
                          data.nodes[link.target].lon,
                          data.nodes[link.target].lat,
                        ],
                      ],
                    })}
                    fill="none"
                    stroke="#999"
                    strokeWidth={link.strokeWidth}
                    opacity="0.6"
                  />
                </g>
              );
            })}
            {data.nodes.map((node) => {
              return (
                <g key={node.id}>
                  <path
                    d={path({
                      type: "Point",
                      coordinates: [node.lon, node.lat],
                    })}
                    fill={node.fillColor}
                    stroke="#fff"
                    strokeWidth="1.5"
                  />
                </g>
              );
            })}
          </g>
        </g>
      </svg>
    </div>
  );
}
