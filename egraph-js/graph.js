export class Graph {
  constructor (Module, pointer) {
    this.module = {
      graphNew: Module.cwrap('graph_new', 'number', []),
      graphAddNode: Module.cwrap('graph_add_node', 'number', ['number']),
      graphAddEdge: Module.cwrap('graph_add_edge', 'number', ['number', 'number', 'number']),
      graphNodeCount: Module.cwrap('graph_node_count', 'number', ['number']),
      graphEdgeCount: Module.cwrap('graph_edge_count', 'number', ['number']),
      graphGetX: Module.cwrap('graph_get_x', 'number', ['number', 'number']),
      graphGetY: Module.cwrap('graph_get_y', 'number', ['number', 'number']),
      graphSetX: Module.cwrap('graph_set_x', 'void', ['number', 'number', 'number']),
      graphSetY: Module.cwrap('graph_set_y', 'void', ['number', 'number', 'number']),
      graphSource: Module.cwrap('graph_source', 'number', ['number', 'number']),
      graphTarget: Module.cwrap('graph_target', 'number', ['number', 'number'])
    }
    this.pointer = pointer || this.module.graphNew()
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

  setX (u, value) {
    return this.module.graphSetX(this.pointer, u, value)
  }

  setY (u, value) {
    return this.module.graphSetY(this.pointer, u, value)
  }

  source (i) {
    return this.module.graphSource(this.pointer, i)
  }

  target (i) {
    return this.module.graphTarget(this.pointer, i)
  }
}
