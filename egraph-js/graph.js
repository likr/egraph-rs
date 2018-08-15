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
