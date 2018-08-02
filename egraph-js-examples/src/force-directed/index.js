import egraph from 'egraph'

class Graph {
  constructor (Module) {
    this.module = {
      graphNew: Module.cwrap('graph_new', 'number', []),
      graphAddNode: Module.cwrap('graph_add_node', 'number', ['number']),
      graphAddEdge: Module.cwrap('graph_add_edge', 'number', ['number', 'number', 'number']),
      graphNodeCount: Module.cwrap('graph_node_count', 'number', ['number']),
      graphEdgeCount: Module.cwrap('graph_edge_count', 'number', ['number']),
      graphGetX: Module.cwrap('graph_get_x', 'number', ['number', 'number']),
      graphGetY: Module.cwrap('graph_get_y', 'number', ['number', 'number'])
    }
    this.pointer = this.module.graphNew()
  }

  addNode () {
    return this.module.graphAddNode(this.pointer)
  }

  addEdge (u, v) {
    return this.module.graphAddEdge(this.pointer, u, v)
  }

  nodeCount () {
    return this.module.graphNodeCount(this.pointer)
  }

  edgeCount () {
    return this.module.graphEdgeCount(this.pointer)
  }

  getX (u) {
    return this.module.graphGetX(this.pointer, u)
  }

  getY (u) {
    return this.module.graphGetY(this.pointer, u)
  }
}

const layout = (Module, graph, data) => {
  const forceDirected = Module.cwrap('force_directed', 'void', ['number'])
  const connectedComponents = Module.cwrap('connected_components', 'number', ['number'])

  const start = Date.now()
  forceDirected(graph.pointer)
  const stop = Date.now()
  console.log(stop - start)
  data.nodes.forEach((node, i) => {
    node.x = graph.getX(i)
    node.y = graph.getY(i)
  })

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
