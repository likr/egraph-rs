import React from 'react'
import * as d3 from 'd3'
import { Wrapper } from '../wrapper'
import { Graph } from 'egraph'
import {
  Simulation,
  ManyBodyForce,
  LinkForce,
  CenterForce
} from 'egraph/layout/force-directed'

const draw = (renderer, graph) => {
  const color = d3.scaleOrdinal(d3.schemeCategory10)
  for (const node of graph.nodes) {
    node.fillColor = color(node.d.group)
  }
  for (const link of graph.edges) {
    link.strokeWidth = Math.sqrt(link.d.value)
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
        data.nodes.forEach((node, i) => {
          graph.addNode(i, node)
        })
        for (const link of data.links) {
          const { source, target } = link
          graph.addEdge(source, target, link)
        }
        const mbForce = new ManyBodyForce()
        const lForce = new LinkForce()
        const cForce = new CenterForce()
        const simulation = new Simulation()
        simulation.add(mbForce)
        simulation.add(lForce)
        simulation.add(cForce)
        const start = window.performance.now()
        const layout = simulation.start(graph)
        const stop = window.performance.now()
        console.log(stop - start)
        for (const u of graph.nodes()) {
          Object.assign(graph.node(u), layout[u])
        }

        draw(this.refs.renderer, graph.toJSON())
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
        node-label-property='d.name'
        node-x-property='d.x'
        node-y-property='d.y'
        link-source-property='u'
        link-target-property='v'
        no-auto-centering
      />
    </Wrapper>
  }

  handleResize (width, height) {
    this.refs.renderer.width = width
    this.refs.renderer.height = height
  }
}
