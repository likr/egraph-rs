import React from 'react'
import * as d3 from 'd3'
import { FM3 } from 'egraph/layout/fm3'
import { Graph } from 'egraph/graph'
import { Wrapper } from '../wrapper'

export class ExampleFM3 extends React.Component {
  componentDidMount () {
    import('egraph-wasm').then(({ Graph, FM3 }) => {
      const components = 10
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

      const fm3 = new FM3()
      fm3.min_size = 10
      fm3.step_iteration = 250
      fm3.unit_edge_length = 15
      fm3.position_force_strength = 0.01
      const layout = fm3.call(graph)

      const data = {}
      data.nodes = Array.from(graph.nodeIndices()).map((u, i) => {
        const { x, y } = layout[i]
        return { x, y }
      })
      data.links = Array.from(graph.edgeIndices()).map((e, i) => {
        const source = graph.source(e)
        const target = graph.target(e)
        return { source, target }
      })

      this.refs.renderer.load(data)
      this.refs.renderer.center()
    })
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
        no-auto-centering
      />
    </Wrapper>
  }

  handleResize (width, height) {
    this.refs.renderer.width = width
    this.refs.renderer.height = height
  }
}
