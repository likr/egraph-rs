import React from 'react'
import * as d3 from 'd3'
import { Wrapper } from '../wrapper'
import {
  Graph,
  Simulation,
  ManyBodyForce,
  LinkForce,
  CenterForce
} from 'egraph-wasm'

const draw = (renderer, graph) => {
  const color = d3.scaleOrdinal(d3.schemeCategory10)
  for (const node of graph.nodes) {
    node.fillColor = color(node.group)
  }
  for (const link of graph.links) {
    link.strokeWidth = Math.sqrt(link.value)
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
        const mbForce = new ManyBodyForce()
        const lForce = new LinkForce()
        const cForce = new CenterForce()
        const simulation = new Simulation()
        simulation.add(mbForce.force())
        simulation.add(lForce.force())
        simulation.add(cForce.force())
        const start = window.performance.now()
        const layout = simulation.start(graph)
        const stop = window.performance.now()
        console.log(stop - start)
        for (const i of graph.nodeIndices()) {
          Object.assign(data.nodes[i], layout[i])
        }

        draw(this.refs.renderer, data)
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
        node-label-property='name'
        no-auto-centering
      />
    </Wrapper>
  }

  handleResize (width, height) {
    this.refs.renderer.width = width
    this.refs.renderer.height = height
  }
}
