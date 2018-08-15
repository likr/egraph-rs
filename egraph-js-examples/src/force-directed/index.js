import egraph from 'egraph'
import {Graph} from 'egraph/graph'
import {Simulation} from 'egraph/layout/force-directed'
import {Allocator} from 'egraph/allocator'

const layout = (Module, graph, data) => {
  const allocator = new Allocator(Module)
  const groupAssignTreemap = Module.cwrap('group_assign_treemap', 'number', ['number', 'number', 'number', 'number', 'number'])

  const groupSet = new Set()
  for (const node of data.nodes) {
    groupSet.add(node.group)
  }
  const groupMap = new Map(Array.from(groupSet).map((g, i) => [g, i]))

  const nodeGroups = allocator.alloc(4 * graph.nodeCount())
  data.nodes.forEach((node, i) => {
    Module.HEAPU32[nodeGroups / 4 + i] = groupMap.get(node.group)
  })

  const groups = groupAssignTreemap(960, 600, groupSet.size, nodeGroups, graph.nodeCount())

  const simulation = new Simulation(Module)
  // simulation.addManyBodyForce()
  // simulation.addLinkForce(graph)
  simulation.addGroupForce(groups, groupSet.size, nodeGroups, graph.nodeCount())
  // simulation.addCenterForce()
  simulation.start(graph)

  data.nodes.forEach((node, i) => {
    node.x = graph.getX(i)
    node.y = graph.getY(i)
  })

  const connectedComponents = Module.cwrap('connected_components', 'number', ['number'])
  const components = connectedComponents(graph.pointer)
  data.nodes.forEach((node, i) => {
    node.component = Module.HEAPU32[components / 4 + i]
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

window.fetch('../egraph.wasm')
  .then((response) => response.arrayBuffer())
  .then((wasmBinary) => {
    egraph({wasmBinary})
      .then((Module) => {
        window.fetch('miserables.json')
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
  })
