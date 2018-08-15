import {egraph} from 'egraph/loader'
import {Simulation} from 'egraph/layout/force-directed'
import {Graph} from 'egraph/graph'

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
