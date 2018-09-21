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
  constructor (pointer, nodeProperties = null, edgeProperties = null) {
    const {Module, functions} = getModule()
    this.Module = Module
    this.functions = functions
    this.pointer = pointer || this.functions.graphNew()
    this.nodeProperties = nodeProperties || []
    this.edgeProperties = edgeProperties || []
  }

  addNode (properties = null) {
    const index = this.functions.graphAddNode(this.pointer)
    this.nodeProperties.push(properties)
    return index
  }

  addEdge (u, v, properties = null) {
    const index = this.functions.graphAddEdge(this.pointer, u, v)
    this.edgeProperties.push(properties)
    return index
  }

  removeNode (index) {
    this.nodeProperties.splice(index, 1)
    this.functions.graphRemoveNode(this.pointer, index)
  }

  removeEdge (index) {
    this.edgeProperties.splice(index, 1)
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
    return new Node(pointer, this.nodeProperties[i])
  }

  edgeAt (i) {
    const pointer = this.functions.graphEdgeAt(this.pointer, i)
    return new Edge(pointer, this.edgeProperties[i])
  }

  degree (u) {
    return this.functions.graphDegree(this.pointer, u)
  }

  filter (nodeMap, edgeMap) {
    const nodeIndices = new Set()
    const edgeIndices = new Set()
    const nodeMapPointer = this.Module.addFunction((index) => {
      const result = nodeMap(index)
      if (result) {
        nodeIndices.add(index)
      }
      return !!result
    }, 'ii')
    const edgeMapPointer = this.Module.addFunction((index) => {
      const result = edgeMap(index)
      if (result) {
        edgeIndices.add(index)
      }
      return !!result
    }, 'ii')
    const pointer = this.functions.graphFilter(this.pointer, nodeMapPointer, edgeMapPointer)
    const graph = new Graph(pointer,
      this.nodeProperties.filter((_, i) => nodeIndices.has(i)),
      this.edgeProperties.filter((_, i) => edgeIndices.has(i)))
    this.Module.removeFunction(nodeMapPointer)
    this.Module.removeFunction(edgeMapPointer)
    return graph
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
