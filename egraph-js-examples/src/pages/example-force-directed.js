import React from 'react'
import * as d3 from 'd3'
import {Simulation} from 'egraph/layout/force-directed'
import {Graph} from 'egraph/graph'
import {loadModule} from '../module'
import {Wrapper} from '../wrapper'

const layout = (Module, graph, data) => {
  const simulation = new Simulation(Module)
  simulation.addManyBodyForce()
  simulation.addLinkForce(graph)
  simulation.addCenterForce()
  simulation.start(graph)

  data.nodes.forEach((node, i) => {
    node.x = graph.getX(i)
    node.y = graph.getY(i)
  })
}

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
  componentDidMount () {
    loadModule().then(({Module}) => {
      window.fetch('/data/miserables.json')
        .then((response) => response.json())
        .then((data) => {
          const graph = new Graph(Module)
          data.nodes.forEach(() => {
            graph.addNode()
          })
          for (const {source, target} of data.links) {
            graph.addEdge(source, target)
          }
          layout(Module, graph, data)
          draw(this.refs.renderer, data)
        })
    })
  }

  render () {
    return <Wrapper onResize={this.handleResize.bind(this)}>
      <eg-renderer
        ref='renderer'
        width='960'
        height='600'
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
