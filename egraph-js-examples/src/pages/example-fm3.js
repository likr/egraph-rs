import React from 'react'
import * as d3 from 'd3'
import {FM3} from 'egraph/layout/fm3'
import {Graph} from 'egraph/graph'
import {Wrapper} from '../wrapper'

const layout = (graph) => {
  const fm3 = new FM3()
  fm3.minSize = 10
  fm3.call(graph)
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

export class ExampleFM3 extends React.Component {
  componentDidMount () {
    setTimeout(() => {
      const components = 5
      const size = 10
      const rows = size
      const cols = size
      const graph = new Graph()
      for (let k = 0; k < components; ++k) {
        const offset = k * rows * cols
        for (let i = 0; i < rows * cols; ++i) {
          graph.addNode()
        }
        for (let i = 0; i < rows; ++i) {
          for (let j = 0; j < cols; ++j) {
            if (i !== rows - 1) {
              graph.addEdge(offset + i * cols + j, offset + (i + 1) * cols + j)
            }
            if (j !== cols - 1) {
              graph.addEdge(offset + i * cols + j, offset + i * cols + j + 1)
            }
          }
        }
      }
      layout(graph)
      draw(this.refs.renderer, graph)
    }, 0)
  }

  render () {
    return <Wrapper onResize={this.handleResize.bind(this)}>
      <eg-renderer
        ref='renderer'
        default-node-width='10'
        default-node-height='10'
        default-node-fill-color='black'
        default-node-stroke-color='#fff'
        default-node-stroke-width='1.5'
        default-node-type='circle'
        default-link-stroke-color='#999'
        default-link-stroke-opacity='0.6'
        graph-links-property='edges'
        no-auto-centering
      />
    </Wrapper>
  }

  handleResize (width, height) {
    this.refs.renderer.width = width
    this.refs.renderer.height = height
  }
}
