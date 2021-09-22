import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import {
  Coordinates,
  Graph,
  Simulation,
  FruchtermanReingoldForce,
  SphericalSpace,
} from "egraph";

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

  const coordinates = Coordinates.initialPlacement(graph);
  graph.nodeIndices().forEach((u, i) => {
    coordinates.setX(u, (2 * Math.PI * i) / graph.nodeCount());
    coordinates.setY(u, i + 1);
  });
  const tangentSpace = Coordinates.initialPlacement(graph);
  const simulation = new Simulation();
  const forces = [new FruchtermanReingoldForce(graph, 0.5, 0.01)];
  simulation.run((alpha) => {
    for (const u of graph.nodeIndices()) {
      SphericalSpace.mapToTangentSpace(u, coordinates, tangentSpace);
      for (const force of forces) {
        force.applyToNode(u, tangentSpace, alpha);
      }
      SphericalSpace.updatePosition(u, coordinates, tangentSpace, 0.6);
    }
  });
  for (const u of graph.nodeIndices()) {
    const node = graph.nodeWeight(u);
    node.lon = (180 * (coordinates.x(u) - Math.PI)) / Math.PI;
    node.lat = (180 * (coordinates.y(u) - Math.PI / 2)) / Math.PI;
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
