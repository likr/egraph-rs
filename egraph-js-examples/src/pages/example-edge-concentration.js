import React from 'react'
import {Simulation} from 'egraph/layout/force-directed'
import {QuasiBiclique} from 'egraph/biclustering'
import {EdgeConcentration} from 'egraph/edge-concentration'
import {Graph} from 'egraph/graph'
import {Wrapper} from '../wrapper'

const layout = (graph, data) => {
  const simulation = new Simulation()
  simulation.addManyBodyForce()
  simulation.addLinkForce(graph)
  simulation.addCenterForce()
  simulation.start(graph)
}

const draw = (renderer, graph) => {
  renderer.load(graph)
  renderer.center()
}

export class ExampleEdgeConcentration extends React.Component {
  componentDidMount () {
    const graph = new Graph()
    const source = new Array(3)
    for (let i = 0; i < 3; ++i) {
      source[i] = graph.addNode({fillColor: 'green'})
    }
    const target = new Array(5)
    for (let i = 0; i < 5; ++i) {
      target[i] = graph.addNode({fillColor: 'orange'})
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

    const biclustering = new QuasiBiclique()
    biclustering.mu = 1
    biclustering.minSize = 0
    const biclusters = biclustering.call(graph, source, target)

    const edgeConcentration = new EdgeConcentration()
    const transformed = edgeConcentration.call(graph, biclusters)

    layout(transformed)
    draw(this.refs.renderer, transformed)
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
        graph-links-property='edges'
        node-fill-color-property='properties.fillColor'
        no-auto-centering
      />
    </Wrapper>
  }

  handleResize (width, height) {
    this.refs.renderer.width = width
    this.refs.renderer.height = height
  }
}
