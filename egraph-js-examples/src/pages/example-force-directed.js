import React from 'react'
import * as d3 from 'd3'
import { Simulation } from 'egraph/layout/force-directed'
import { Graph } from 'egraph/graph'
import { Wrapper } from '../wrapper'

const layout = (graph) => {
  const simulation = new Simulation()
  simulation.addManyBodyForce()
  simulation.addLinkForce(graph)
  simulation.addCenterForce()
  simulation.start(graph)
}

const draw = (renderer, graph) => {
  const color = d3.scaleOrdinal(d3.schemeCategory10)
  for (const node of graph.nodes) {
    node.properties.fillColor = color(node.properties.group)
  }
  for (const link of graph.edges) {
    link.properties.strokeWidth = Math.sqrt(link.properties.value)
  }
  renderer.load(graph)
  renderer.center()
}

export class ExampleForceDirected extends React.Component {
  componentDidMount () {
    window.fetch('/data/miserables.json')
      .then((response) => response.json())
      .then((data) => {
        const graph = new Graph()
        data.nodes.forEach((node) => {
          graph.addNode(node)
        })
        for (const link of data.links) {
          const { source, target } = link
          graph.addEdge(source, target, link)
        }
        layout(graph)
        draw(this.refs.renderer, graph)
      })
  }

  render () {
    return <Wrapper onResize={this.handleResize.bind(this)}>
      <eg-renderer
        ref='renderer'
        default-node-width='10'
        default-node-height='10'
        default-node-stroke-color='#fff'
        default-node-stroke-width='1.5'
        default-node-type='circle'
        default-link-stroke-color='#999'
        default-link-stroke-opacity='0.6'
        graph-links-property='edges'
        node-fill-color-property='properties.fillColor'
        node-label-property='properties.name'
        link-stroke-width-property='properties.strokeWidth'
        no-auto-centering
      />
    </Wrapper>
  }

  handleResize (width, height) {
    this.refs.renderer.width = width
    this.refs.renderer.height = height
  }
}
