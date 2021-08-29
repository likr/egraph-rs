import React from "react";
import * as d3 from "d3";
import {
  Coordinates,
  Graph,
  Simulation,
  ManyBodyForce,
  LinkForce,
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

        const coordinates = Coordinates.initialPlacement(graph);
        const simulation = new Simulation();
        const forces = [
          new ManyBodyForce(graph, () => ({ strength: -30 })),
          new LinkForce(graph, () => ({ distance: 30 })),
        ];

        const draw = () => {
          if (!this.refs.renderer || simulation.isFinished()) {
            return;
          }
          simulation.runStep(1, (alpha) => {
            for (const force of forces) {
              force.apply(coordinates, alpha);
            }
            coordinates.updatePosition(0.6);
            coordinates.centralize();
          });
          for (const u of graph.nodeIndices()) {
            const node = graph.nodeWeight(u);
            node.x = coordinates.x(u);
            node.y = coordinates.y(u);
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
          node-id-property="id"
          node-label-property="name"
          no-auto-centering
        />
      </Wrapper>
    );
  }
}
