import React from 'react'
import * as d3 from 'd3'
import {
  Simulation,
  ManyBodyForce,
  LinkForce,
  CenterForce
} from 'egraph/layout/force-directed'
import { ForceDirectedEdgeBundling } from 'egraph/edge_bundling'
import { Graph } from 'egraph'
import { Wrapper } from '../wrapper'

export class ExampleEdgeBundling extends React.Component {
  componentDidMount() {
    window
      .fetch('/data/miserables.json')
      .then((response) => response.json())
      .then((data) => {
        const color = d3.scaleOrdinal(d3.schemeCategory10)
        const graph = new Graph()
        data.nodes.forEach((node, i) => {
          node.fillColor = color(node.group)
          graph.addNode(i, node)
        })
        for (const link of data.links) {
          link.strokeWidth = Math.sqrt(link.value)
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
        const layout = simulation.start(graph)
        for (const u of graph.nodes()) {
          const node = graph.node(u)
          node.x = layout.nodes[u].x
          node.y = layout.nodes[u].y
        }

        const edgeBundling = new ForceDirectedEdgeBundling()
        const bends = edgeBundling.call(graph, layout.nodes)
        Array.from(graph.edges()).map(([u, v], i) => {
          graph.edge(u, v).bends = bends[i].bends.map(({ x, y }) => [x, y])
        })

        this.refs.renderer.load(graph.toJSON())
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