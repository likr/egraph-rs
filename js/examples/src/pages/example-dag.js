import React from 'react'
import * as d3 from 'd3'
import {
  Graph,
  SimulationBuilder,
  ManyBodyForce,
  LinkForce,
  RadialForce,
  longestPathRanking
} from 'egraph'
import { Wrapper } from '../wrapper'

export class ExampleDag extends React.Component {
  componentDidMount() {
    window
      .fetch('/data/dependencies.json')
      .then((response) => response.json())
      .then((data) => {
        const graph = new Graph()
        const indices = new Map()
        for (const node of data.nodes) {
          const index = indices.size
          indices.set(node.id, index)
          graph.addNode(index, node)
        }
        for (const link of data.links) {
          link.strokeWidth = Math.sqrt(link.value)
          const { source, target } = link
          graph.addEdge(indices.get(source), indices.get(target), link)
        }
        const nodeSizeScale = d3
          .scaleSqrt()
          .domain(
            d3.extent(data.nodes, ({ id }) => graph.degree(indices.get(id)))
          )
          .range([10, 30])
        const nodeLabelFontSizeScale = d3
          .scaleSqrt()
          .domain(
            d3.extent(data.nodes, ({ id }) => graph.degree(indices.get(id)))
          )
          .range([1, 20])
        const color = d3.scaleOrdinal(d3.schemePaired)
        const ranking = longestPathRanking(graph)
        for (const u of graph.nodes()) {
          const node = graph.node(u)
          node.rank = ranking[u].toString()
          node.width = node.height = nodeSizeScale(graph.degree(u))
          node.labelFontSize = nodeLabelFontSizeScale(graph.degree(u))
          const path = node.id.substring(0, node.id.lastIndexOf('/'))
          node.fillColor = color(path)
          if (node.labelFontSize >= 5) {
            node.label = node.id.substring(node.id.lastIndexOf('/') + 1)
          }
        }
        const distance = 80
        const builder = new SimulationBuilder()
        const manyBodyForce = builder.add(new ManyBodyForce())
        builder.get(manyBodyForce).strength = () => -5 * distance
        const linkForce = builder.add(new LinkForce())
        builder.get(linkForce).strength = () => 0.05
        builder.get(linkForce).distance = (_, u, v) =>
          (ranking[u] - ranking[v]) * distance
        const radialForce = builder.add(new RadialForce())
        builder.get(radialForce).strength = () => 1
        builder.get(radialForce).radius = (graph, u) =>
          distance * (ranking[u] + 1)
        const simulation = builder.start(graph)

        for (const u of graph.nodes()) {
          const node = graph.node(u)
          node.x = simulation.x(u)
          node.y = simulation.y(u)
        }

        this.refs.renderer.load(data)
        this.refs.renderer.center()
      })
  }

  render() {
    return (
      <Wrapper
        onResize={(width, height) => {
          this.refs.renderer.width = width
          this.refs.renderer.height = height
        }}
      >
        <eg-renderer
          ref='renderer'
          transition-duration='1000'
          node-id-property='id'
          default-node-fill-color='orange'
          default-node-stroke-width='0'
          default-node-type='circle'
          default-node-label-font-family='impact'
          default-link-stroke-color='#999'
          default-link-stroke-opacity='0.6'
          default-link-target-marker-shape='triangle'
          no-auto-centering
        />
      </Wrapper>
    )
  }
}
