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

const draw = (renderer, data) => {
  const color = d3.scaleOrdinal(d3.schemeCategory10)
  for (const node of data.nodes) {
    node.fillColor = color(node.group)
  }
  for (const link of data.links) {
    link.strokeWidth = Math.sqrt(link.value)
  }
  renderer.load(data)
  renderer.center()
}

export class ExampleForceDirected extends React.Component {
  componentDidMount() {
    window
      .fetch('/data/miserables.json')
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
          const node = graph.node(u)
          node.x = layout.nodes[u].x
          node.y = layout.nodes[u].y
        }

        draw(this.refs.renderer, graph.toJSON())
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
