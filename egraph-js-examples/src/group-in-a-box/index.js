import {egraph} from 'egraph/loader'
import {Algorithms} from 'egraph/algorithms'
import {Allocator} from 'egraph/allocator'
import {Simulation} from 'egraph/layout/force-directed'
import {Graph} from 'egraph/graph'
import {EdgeBundling} from 'egraph/edge-bundling'

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
  const edgeBundling = new EdgeBundling(Module)

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

const draw = (data) => {
  const renderer = document.querySelector('eg-renderer')
  const color = window.d3.scaleOrdinal(window.d3.schemeCategory20)
  for (const node of data.nodes) {
    node.fillColor = color(node.group)
  }
  for (const link of data.links) {
    link.strokeWidth = Math.sqrt(link.value)
  }
  renderer.load(data)
  renderer.center()
}

egraph('../egraph.wasm').then(({Module}) => {
  window.fetch('../data/miserables.json')
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
      draw(data)
    })
})
