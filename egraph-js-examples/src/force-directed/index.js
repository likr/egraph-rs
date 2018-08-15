import egraph from 'egraph'

export class Graph {
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

export class Simulation {
  constructor (Module) {
    this.module = {
      simulationNew: Module.cwrap('simulation_new', 'number', []),
      simulationAddCenterForce: Module.cwrap('simulation_add_center_force', 'void', ['number']),
      simulationAddGroupForce: Module.cwrap('simulation_add_group_force', 'void', ['number', 'number', 'number', 'number', 'number']),
      simulationAddLinkForce: Module.cwrap('simulation_add_link_force', 'void', ['number', 'number']),
      simulationAddManyBodyForce: Module.cwrap('simulation_add_many_body_force', 'void', ['number']),
      simulationStart: Module.cwrap('simulation_start', 'void', ['number', 'number'])
    }
    this.pointer = this.module.simulationNew()
  }

  addCenterForce () {
    this.module.simulationAddCenterForce(this.pointer)
  }

  addGroupForce () {
  }

  addLinkForce (graph) {
    this.module.simulationAddLinkForce(this.pointer, graph.pointer)
  }

  addManyBodyForce () {
    this.module.simulationAddManyBodyForce(this.pointer)
  }

  start (graph) {
    const start = Date.now()
    this.module.simulationStart(this.pointer, graph.pointer)
    const stop = Date.now()
    console.log(stop - start)
  }
}

export class Allocator {
  constructor (Module) {
    this.module = {
      alloc: Module.cwrap('rust_alloc', 'number', ['number']),
      free: Module.cwrap('rust_free', 'void', ['number'])
    }
  }

  alloc (bytes) {
    return this.module.alloc(bytes)
  }

  free (pointer) {
    this.module.free(pointer)
  }
}

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
