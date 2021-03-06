import React from "react";
import * as d3 from "d3";
import {
  Graph,
  Simulation,
  ManyBodyForce,
  LinkForce,
  CenterForce,
  initialPlacement,
} from "egraph";
import { Wrapper } from "../wrapper";

export class ExampleForceDirected extends React.Component {
  componentDidMount() {
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

        const initialCoordinates = initialPlacement(graph);
        const simulation = new Simulation(graph, (u) => initialCoordinates[u]);

        const forces = [
          new ManyBodyForce(graph, () => ({ strength: -30 })),
          new LinkForce(graph, () => ({ distance: 30 })),
          new CenterForce(),
        ];

        const draw = () => {
          if (!this.refs.renderer || simulation.isFinished()) {
            return;
          }
          const coordinates = simulation.runStep(1, forces);
          for (const u of graph.nodeIndices()) {
            const node = graph.nodeWeight(u);
            const [x, y] = coordinates[u];
            node.x = x;
            node.y = y;
          }
          this.refs.renderer.update();
          this.refs.renderer.center();
          window.requestAnimationFrame(draw);
        };

        this.refs.renderer.load(data);
        draw();
      });
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
}
