import {getModule} from '.'

export class Graph {
  constructor (pointer) {
    this.functions = getModule().functions
    this.pointer = pointer || this.functions.graphNew()
  }

  addNode () {
    return this.functions.graphAddNode(this.pointer)
  }

  addEdge (u, v) {
    return this.functions.graphAddEdge(this.pointer, u, v)
  }

  nodeCount () {
    return this.functions.graphNodeCount(this.pointer)
  }

  edgeCount () {
    return this.functions.graphEdgeCount(this.pointer)
  }

  getX (u) {
    return this.functions.graphGetX(this.pointer, u)
  }

  getY (u) {
    return this.functions.graphGetY(this.pointer, u)
  }

  setX (u, value) {
    return this.functions.graphSetX(this.pointer, u, value)
  }

  setY (u, value) {
    return this.functions.graphSetY(this.pointer, u, value)
  }

  source (i) {
    return this.functions.graphSource(this.pointer, i)
  }

  target (i) {
    return this.functions.graphTarget(this.pointer, i)
  }
}
