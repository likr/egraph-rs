const privates = new WeakMap()

const p = (self) => privates.get(self)

const checkNode = (graph, u) => {
  if (!p(graph).nodes.has(u)) {
    throw new Error(`Invalid node: ${u}`)
  }
}

export class Graph {
  constructor() {
    privates.set(this, {
      nodes: new Map(),
      numNodes: 0,
      numEdges: 0
    })
  }

  node(u) {
    const nodes = p(this).nodes
    if (nodes.get(u)) {
      return nodes.get(u).data
    }
    return null
  }

  edge(u, v) {
    const nodes = p(this).nodes
    if (nodes.get(u) && nodes.get(u).outNodes.get(v)) {
      return nodes.get(u).outNodes.get(v)
    }
    return null
  }

  *nodes() {
    for (const key of p(this).nodes.keys()) {
      yield key
    }
  }

  *edges() {
    for (const u of this.nodes()) {
      for (const v of this.outNodes(u)) {
        yield [u, v]
      }
    }
  }

  *outNodes(u) {
    checkNode(this, u)
    for (const key of p(this)
      .nodes.get(u)
      .outNodes.keys()) {
      yield key
    }
  }

  *inNodes(u) {
    checkNode(this, u)
    for (const key of p(this)
      .nodes.get(u)
      .inNodes.keys()) {
      yield key
    }
  }

  *outEdges(u) {
    for (const v of this.outNodes(u)) {
      yield [u, v]
    }
  }

  *inEdges(u) {
    for (const v of this.inNodes(u)) {
      yield [v, u]
    }
  }

  nodeCount() {
    return p(this).numNodes
  }

  edgeCount() {
    return p(this).numEdges
  }

  degree(u) {
    return this.outDegree(u) + this.inDegree(u)
  }

  outDegree(u) {
    checkNode(this, u)
    return p(this).nodes.get(u).outNodes.size
  }

  inDegree(u) {
    checkNode(this, u)
    return p(this).nodes.get(u).inNodes.size
  }

  addNode(u, obj = {}) {
    if (this.node(u)) {
      throw new Error(`Duplicated node: ${u}`)
    }
    p(this).nodes.set(u, {
      outNodes: new Map(),
      inNodes: new Map(),
      data: obj
    })
    p(this).numNodes++
    return this
  }

  addEdge(u, v, obj = {}) {
    checkNode(this, u)
    checkNode(this, v)
    if (this.edge(u, v)) {
      throw new Error(`Duplicated edge: (${u}, ${v})`)
    }
    p(this).numEdges++
    p(this)
      .nodes.get(u)
      .outNodes.set(v, obj)
    p(this)
      .nodes.get(v)
      .inNodes.set(u, obj)
    return this
  }

  removeNode(u) {
    for (const v of this.outNodes(u)) {
      this.removeEdge(u, v)
    }
    for (const v of this.inNodes(u)) {
      this.removeEdge(v, u)
    }
    p(this).nodes.delete(u)
    p(this).numNodes--
    return this
  }

  removeEdge(u, v) {
    if (this.edge(u, v) === null) {
      throw Error(`Invalid edge: (${u}, ${v})`)
    }
    p(this)
      .nodes.get(u)
      .outNodes.delete(v)
    p(this)
      .nodes.get(v)
      .inNodes.delete(u)
    p(this).numEdges--
    return this
  }

  toJSON() {
    return {
      nodes: Array.from(this.nodes()).map((u) =>
        Object.assign(this.node(u), { id: u })
      ),
      links: Array.from(this.edges()).map(([u, v]) =>
        Object.assign(this.edge(u, v), { source: u, target: v })
      )
    }
  }

  toString() {
    return JSON.stringify(this.toJSON())
  }
}
