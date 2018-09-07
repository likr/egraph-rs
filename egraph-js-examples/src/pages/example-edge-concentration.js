import React from 'react'
import {Simulation} from 'egraph/layout/force-directed'
import {QuasiBiclique} from 'egraph/biclustering'
import {EdgeConcentration} from 'egraph/edge-concentration'
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
  renderer.load(data)
  renderer.center()
}

export class ExampleEdgeConcentration extends React.Component {
  componentDidMount () {
    loadModule().then(({Module}) => {
      const graph = new Graph(Module)
      const source = new Array(3)
      for (let i = 0; i < 3; ++i) {
        source[i] = graph.addNode()
      }
      const target = new Array(5)
      for (let i = 0; i < 5; ++i) {
        target[i] = graph.addNode()
      }
      graph.addEdge(source[0], target[0])
      graph.addEdge(source[0], target[1])
      graph.addEdge(source[0], target[2])
      graph.addEdge(source[0], target[3])
      graph.addEdge(source[1], target[0])
      graph.addEdge(source[1], target[1])
      graph.addEdge(source[1], target[2])
      graph.addEdge(source[1], target[3])
      graph.addEdge(source[1], target[4])
      graph.addEdge(source[2], target[1])
      graph.addEdge(source[2], target[2])
      graph.addEdge(source[2], target[3])
      graph.addEdge(source[2], target[4])

      const biclustering = new QuasiBiclique(Module)
      biclustering.mu = 1
      biclustering.minSize = 0
      const biclusters = biclustering.call(graph, source, target)

      const edgeConcentration = new EdgeConcentration(Module)
      const transformed = edgeConcentration.call(graph, biclusters)

      const nodes = new Array(transformed.nodeCount())
      for (let i = 0; i < transformed.nodeCount(); ++i) {
        nodes[i] = {}
      }
      const links = new Array(transformed.edgeCount())
      for (let i = 0; i < transformed.edgeCount(); ++i) {
        links[i] = {
          source: transformed.source(i),
          target: transformed.target(i)
        }
      }
      const data = {nodes, links}

      layout(Module, transformed, data)
      draw(this.refs.renderer, data)
    })
  }

  render () {
    return <Wrapper onResize={this.handleResize.bind(this)}>
      <eg-renderer
        ref='renderer'
        default-node-width='10'
        default-node-height='10'
        default-node-fill-color='#000'
        default-node-stroke-width='0'
        default-node-type='circle'
        default-link-stroke-color='#999'
        default-link-stroke-opacity='0.6'
        no-auto-centering
      />
    </Wrapper>
  }

  handleResize (width, height) {
    this.refs.renderer.width = width
    this.refs.renderer.height = height
  }
}
