import React from 'react'
import * as d3 from 'd3'
import {
  Graph,
  ForceDirectedGrouping,
  TreemapGrouping,
  Simulation,
  GroupCenterForce,
  GroupLinkForce,
  GroupManyBodyForce,
  GroupPositionForce
} from 'egraph-wasm'
import { Wrapper } from '../wrapper'

const grouper = (name, graph, groupAccessor) => {
  let grouping
  switch (name) {
    case 'treemap':
      grouping = new TreemapGrouping()
      grouping.group(groupAccessor)
      return grouping
    case 'force-directed':
    default:
      grouping = new ForceDirectedGrouping()
      grouping.group(groupAccessor)
      grouping.linkWeight((e) => graph.edge(e).value)
      grouping.manyBodyForceStrength((_) => -2000)
      return grouping
  }
}

const groupShape = (name) => {
  switch (name) {
    case 'treemap':
      return 'rect'
    case 'force-directed':
    default:
      return 'circle'
  }
}

const layout = (data, groupLayout) => {
  const graph = new Graph()
  data.nodes.forEach((node) => {
    graph.addNode(node)
  })
  for (const link of data.links) {
    const { source, target } = link
    graph.addEdge(source, target, link)
  }

  const groupAccessor = (i) => graph.node(i).group
  const grouping = grouper(groupLayout, graph, groupAccessor)
  const groups = grouping.call(graph, 600, 600)
  data.groups = Array.from(Object.values(groups))

  const manyBodyForce = new GroupManyBodyForce()
  manyBodyForce.group(groupAccessor)
  manyBodyForce.strength((_) => -30)
  const linkForce = new GroupLinkForce()
  linkForce.inter_group(0.001)
  linkForce.group(groupAccessor)
  const positionForce = new GroupPositionForce()
  const centerForce = new GroupCenterForce()
  centerForce.group(groupAccessor)
  centerForce.groupX((g) => groups[g].x)
  centerForce.groupY((g) => groups[g].y)

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
  componentDidMount() {
    window
      .fetch('/data/miserables.json')
      .then((response) => response.json())
      .then((data) => {
        const color = d3.scaleOrdinal(d3.schemeCategory10)
        for (const node of data.nodes) {
          node.fillColor = color(node.group)
        }
        for (const link of data.links) {
          link.strokeWidth = Math.sqrt(link.value)
        }
        this.data = data
        layout(data, this.refs.groupLayout.value)
        this.refs.renderer.load(data)
        this.refs.renderer.center()
      })
  }

  render() {
    return (
      <div>
        <div>
          <Wrapper onResize={this.handleResize.bind(this)}>
            <eg-renderer
              ref='renderer'
              transition-duration='1000'
              default-node-width='10'
              default-node-height='10'
              default-node-stroke-color='#fff'
              default-node-stroke-width='1.5'
              default-link-stroke-color='#999'
              default-link-stroke-opacity='0.6'
              default-group-type='circle'
              node-label-property='name'
              no-auto-update
              no-auto-centering
            />
          </Wrapper>
        </div>
        <div>
          <div className='field'>
            <label className='label'>Group Layout</label>
            <div className='control'>
              <div className='select is-fullwidth'>
                <select
                  ref='groupLayout'
                  defaultValue='force-directed'
                  onChange={this.handleChangeGroupLayout.bind(this)}
                >
                  <option value='force-directed'>Force-directed</option>
                  <option value='treemap'>Treemap</option>
                </select>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  handleResize(width, height) {
    this.refs.renderer.width = width
    this.refs.renderer.height = height
  }

  handleChangeGroupLayout() {
    this.refs.renderer.defaultGroupType = groupShape(
      this.refs.groupLayout.value
    )
    this.layout()
  }

  layout() {
    layout(this.data, this.refs.groupLayout.value)
    this.refs.renderer.update()
    this.refs.renderer.center()
  }
}
