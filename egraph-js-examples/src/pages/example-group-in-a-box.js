import React from 'react'
import * as d3 from 'd3'
import {Algorithms} from 'egraph/algorithms'
import {Allocator} from 'egraph/allocator'
import {Simulation} from 'egraph/layout/force-directed'
import {Graph} from 'egraph/graph'
import {EdgeBundling} from 'egraph/edge-bundling'
import {loadModule} from '../module'

const countGroups = (nodes) => {
  const groupCount = new Map()
  for (const node of nodes) {
    if (!groupCount.has(node.group)) {
      groupCount.set(node.group, 0)
    }
    groupCount.set(node.group, groupCount.get(node.group) + 1)
  }
  const groups = Array.from(groupCount.entries()).map(([name, count]) => ({name, count}))
  groups.sort((a, b) => b.count - a.count)
  return groups
}

const layout = (Module, graph, data) => {
  const width = 800
  const height = 600
  const allocator = new Allocator(Module)
  const algorithms = new Algorithms(Module)

  const groups = countGroups(data.nodes)
  const values = groups.map(({count}) => count)
  const sumValues = values.reduce((a, b) => a + b)
  const normalizedValues = values.map((v) => v / sumValues * width * height)
  const tiles = algorithms.squarifiedTreemap(width, height, normalizedValues)

  const groupsPointer = allocator.alloc(16 * groups.length)
  tiles.forEach((tile, i) => {
    Module.HEAPF32[groupsPointer / 4 + 2 * i] = tile.x + tile.width / 2
    Module.HEAPF32[groupsPointer / 4 + 2 * i + 1] = tile.y + tile.height / 2
  })

  const groupMap = new Map(groups.map(({name}, i) => [name, i]))
  const nodeGroupsPointer = allocator.alloc(4 * graph.nodeCount())
  data.nodes.forEach((node, i) => {
    Module.HEAPU32[nodeGroupsPointer / 4 + i] = groupMap.get(node.group)
  })

  const simulation = new Simulation(Module)
  const f1 = simulation.addGroupManyBodyForce(groupsPointer, groups.length, nodeGroupsPointer, graph.nodeCount())
  const f2 = simulation.addGroupLinkForce(graph, nodeGroupsPointer)
  const f3 = simulation.addGroupCenterForce(groupsPointer, groups.length, nodeGroupsPointer, graph.nodeCount())
  simulation.setStrength(f1, 0.2)
  simulation.setStrength(f2, 0.1)
  simulation.setStrength(f3, 0.2)
  simulation.start(graph)

  const edgeBundling = new EdgeBundling(Module)
  const lines = edgeBundling.call(graph)

  tiles.forEach((tile, i) => {
    tile.name = groups[i].name
    tile.x += tile.width / 2
    tile.y += tile.height / 2
  })
  data.groups = tiles

  data.nodes.forEach((node, i) => {
    node.x = graph.getX(i)
    node.y = graph.getY(i)
  })

  data.links.forEach((link, i) => {
    link.bends = lines[i].map(({x, y}) => [x, y])
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

export class ExampleGroupInABox extends React.Component {
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
    return <eg-renderer
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
  }
}
