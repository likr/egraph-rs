export class Algorithm {
  constructor (Module) {
    this.Module = Module
    this.module = {
      connectedComponents: Module.cwrap('connected_components', 'number', ['number'])
    }
  }

  connectedComponents (graph) {
    const components = this.module.connectedComponents(graph.pointer)
    const n = graph.nodeCount()
    const result = new Array(n)
    for (let i = 0; i < n; ++i) {
      result[i] = this.Module.HEAPU32[components / 4 + i]
    }
    return result
  }
}
