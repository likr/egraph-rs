import React from "react";
import {
  SimulationBuilder,
  Graph,
  muQuasiBicliques,
} from "egraph/dist/web/egraph_wasm";
import { Wrapper } from "../wrapper";

export class ExampleBiclustering extends React.Component {
  componentDidMount() {
    const graph = new Graph();
    graph.addNode(0);
    graph.addNode(1);
    graph.addNode(2);
    graph.addNode(3);
    graph.addNode(4);
    graph.addNode(5);
    graph.addNode(6);
    graph.addNode(7);
    graph.addEdge(0, 3);
    graph.addEdge(0, 4);
    graph.addEdge(0, 5);
    graph.addEdge(0, 6);
    graph.addEdge(0, 7);
    graph.addEdge(1, 3);
    graph.addEdge(1, 4);
    graph.addEdge(1, 5);
    graph.addEdge(1, 6);
    graph.addEdge(1, 7);
    graph.addEdge(2, 3);
    graph.addEdge(2, 4);
    graph.addEdge(2, 5);
    graph.addEdge(2, 6);

    console.log(muQuasiBicliques(graph, [0, 1, 2], [3, 4, 5, 6, 7]));

    const builder = SimulationBuilder.defaultConnected();
    const simulation = builder.start(graph);
    for (const u of graph.nodes()) {
      const node = graph.node(u);
      node.x = simulation.x(u);
      node.y = simulation.y(u);
    }

    this.refs.renderer.load(graph.toJSON());
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
