import {getModule} from '.'

class Node {
  constructor (pointer, properties = null) {
    this.functions = getModule().functions
    this.pointer = pointer
    this.properties = properties || {}
  }

  get x () {
    return this.functions.nodeGetX(this.pointer)
  }

  set x (value) {
    this.functions.nodeSetX(this.pointer, value)
  }

  get y () {
    return this.functions.nodeGetY(this.pointer)
  }

  set y (value) {
    this.functions.nodeSetY(this.pointer, value)
  }
}

class Edge {
  constructor (pointer, properties = null) {
    this.functions = getModule().functions
    this.pointer = pointer
    this.properties = properties || {}
  }

  get source () {
    return this.functions.edgeSource(this.pointer)
  }

  get target () {
    return this.functions.edgeTarget(this.pointer)
  }
}

export class Graph {
  constructor (pointer) {
    this.functions = getModule().functions
    this.pointer = pointer || this.functions.graphNew()
    this.nodeProperties = new Map()
    this.edgeProperties = new Map()
  }

  addNode (properties = null) {
    const index = this.functions.graphAddNode(this.pointer)
    this.nodeProperties.set(index, properties)
    return index
  }

  addEdge (u, v, properties = null) {
    const index = this.functions.graphAddEdge(this.pointer, u, v)
    this.edgeProperties.set(index, properties)
    return index
  }

  removeNode (index) {
    this.functions.graphRemoveNode(this.pointer, index)
  }

  removeEdge (index) {
    this.functions.graphRemoveEdge(this.pointer, index)
  }

  nodeCount () {
    return this.functions.graphNodeCount(this.pointer)
  }

  edgeCount () {
    return this.functions.graphEdgeCount(this.pointer)
  }

  nodeAt (i) {
    const pointer = this.functions.graphNodeAt(this.pointer, i)
    return new Node(pointer, this.nodeProperties.get(i))
  }

  edgeAt (i) {
    const pointer = this.functions.graphEdgeAt(this.pointer, i)
    return new Edge(pointer, this.edgeProperties.get(i))
  }

  degree (u) {
    return this.functions.graphDegree(this.pointer, u)
  }

  get nodes () {
    const graph = this
    const n = this.nodeCount()
    let nextIndex = 0
    return {
      [Symbol.iterator] () {
        return {
          next () {
            if (nextIndex < n) {
              return {
                value: graph.nodeAt(nextIndex++),
                done: false
              }
            }
            return {
              done: true
            }
          }
        }
      }
    }
  }

  get edges () {
    const graph = this
    const n = this.edgeCount()
    let nextIndex = 0
    return {
      [Symbol.iterator] () {
        return {
          next () {
            if (nextIndex < n) {
              return {
                value: graph.edgeAt(nextIndex++),
                done: false
              }
            }
            return {
              done: true
            }
          }
        }
      }
    }
  }
}
