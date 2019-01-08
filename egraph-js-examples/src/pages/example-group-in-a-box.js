import React from 'react'
import * as d3 from 'd3'
import { egraph } from '../egraph'
import { Wrapper } from '../wrapper'

const layout = (mod, data) => {
  const { Graph } = mod
  const graph = new Graph()
  data.nodes.forEach((node) => {
    graph.addNode(node)
  })
  for (const link of data.links) {
    const { source, target } = link
    graph.addEdge(source, target, link)
  }

  const groupAccessor = i => graph.node(i).group

  const { ForceDirectedGrouping } = mod
  const grouping = new ForceDirectedGrouping()
  grouping.group(groupAccessor)
  grouping.size(i => 1000)
  const groups = grouping.call(graph, 600, 600)
  data.groups = Array.from(Object.values(groups))

  const { GroupCenterForce, GroupLinkForce, GroupManyBodyForce, GroupPositionForce } = mod
  const manyBodyForce = new GroupManyBodyForce()
  manyBodyForce.group(groupAccessor)
  manyBodyForce.strength(_ => -30)
  const linkForce = new GroupLinkForce()
  linkForce.inter_group(0.001)
  linkForce.group(groupAccessor)
  const positionForce = new GroupPositionForce()
  const centerForce = new GroupCenterForce()
  centerForce.group(groupAccessor)
  centerForce.groupX(g => groups[g].x)
  centerForce.groupY(g => groups[g].y)

  const { Simulation } = mod
  const simulation = new Simulation()
  simulation.add(manyBodyForce.force())
  simulation.add(linkForce.force())
  simulation.add(positionForce.force())
  simulation.add(centerForce.force())

  const layout = simulation.start(graph)
  for (const i of graph.nodeIndices()) {
    Object.assign(data.nodes[i], layout[i])
  }
}

export class ExampleGroupInABox extends React.Component {
  componentDidMount () {
    (async () => {
      const response = await window.fetch('/data/miserables.json')
      const data = await response.json()
      const mod = await egraph()

      const color = d3.scaleOrdinal(d3.schemeCategory10)
      for (const node of data.nodes) {
        node.fillColor = color(node.group)
      }
      for (const link of data.links) {
        link.strokeWidth = Math.sqrt(link.value)
      }

      layout(mod, data)
      this.refs.renderer.load(data)
      this.refs.renderer.center()
    })()
  }

  render () {
    return <Wrapper onResize={this.handleResize.bind(this)}>
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
        default-group-type='circle'
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
