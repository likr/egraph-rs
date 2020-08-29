import React from "react";
import * as d3 from "d3";
import {
  Graph,
  fm3,
  Simulation,
  forceNonconnected,
  initialPlacement,
} from "egraph";
import { Wrapper } from "../wrapper";

export class ExampleFM3 extends React.Component {
  componentDidMount() {
    const n = 10;
    const m = 4;
    const graph = new Graph();
    const indices = new Map();
    const index = (i, j, k) => k * n * n + i * n + j;
    for (let k = 0; k < m; ++k) {
      for (let i = 0; i < n; ++i) {
        for (let j = 0; j < n; ++j) {
          indices.set(index(i, j, k), graph.addNode({}));
        }
      }
      for (let i = 0; i < n; ++i) {
        for (let j = 1; j < n; ++j) {
          graph.addEdge(
            indices.get(index(i, j - 1, k)),
            indices.get(index(i, j, k)),
            {},
          );
        }
      }
      for (let i = 1; i < n; ++i) {
        for (let j = 0; j < n; ++j) {
          graph.addEdge(
            indices.get(index(i - 1, j, k)),
            indices.get(index(i, j, k)),
            {},
          );
        }
      }
    }

    const initialCoordinates = fm3(graph, 1, 200);
    const simulation = new Simulation(graph, (u) => initialCoordinates[u]);
    const forces = forceNonconnected(graph);
    const coordinates = simulation.run(forces);

    this.refs.renderer.load({
      nodes: graph.nodeIndices().map((u) => {
        const [x, y] = coordinates[u];
        return {
          x,
          y,
        };
      }),
      links: graph.edgeIndices().map((e) => {
        const [source, target] = graph.edgeEndpoints(e);
        return {
          source,
          target,
        };
      }),
    });
    this.refs.renderer.center();
  }

  render() {
    return (
      <Wrapper
        onResize={(width, height) => {
          this.refs.renderer.width = width;
          this.refs.renderer.height = height;
        }}
      >
        <eg-renderer
          ref="renderer"
          transition-duration="1000"
          default-node-width="10"
          default-node-height="10"
          default-node-stroke-color="#999"
          default-node-type="circle"
          default-link-stroke-color="#999"
          default-link-stroke-opacity="0.6"
          node-label-property="name"
          no-auto-centering
        />
      </Wrapper>
    );
  }
}
