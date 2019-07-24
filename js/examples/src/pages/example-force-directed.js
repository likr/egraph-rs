import React from 'react'
import * as d3 from 'd3'
import { Graph, SimulationBuilder } from 'egraph'
import { Wrapper } from '../wrapper'

export class ExampleForceDirected extends React.Component {
  componentDidMount() {
    window
      .fetch('/data/miserables.json')
      .then((response) => response.json())
      .then((data) => {
        const color = d3.scaleOrdinal(d3.schemeCategory10)
        const graph = new Graph()
        for (const node of data.nodes) {
          node.fillColor = color(node.group)
          graph.addNode(node.id, node)
        }
        for (const link of data.links) {
          link.strokeWidth = Math.sqrt(link.value)
          const { source, target } = link
          graph.addEdge(source, target, link)
        }
        const builder = SimulationBuilder.defaultConnected()
        const simulation = builder.build(graph)

        const draw = () => {
          if (!this.refs.renderer || simulation.isFinished()) {
            return
          }
          window.requestAnimationFrame(draw)
          simulation.stepN(1)
          for (const u of graph.nodes()) {
            const node = graph.node(u)
            node.x = simulation.x(u)
            node.y = simulation.y(u)
          }
          this.refs.renderer.update()
          this.refs.renderer.center()
        }

        this.refs.renderer.load(data)
        draw()
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
          default-node-width='10'
          default-node-height='10'
          default-node-stroke-color='#fff'
          default-node-stroke-width='1.5'
          default-node-type='circle'
          default-link-stroke-color='#999'
          default-link-stroke-opacity='0.6'
          node-label-property='name'
          no-auto-centering
        />
      </Wrapper>
    )
  }
}
