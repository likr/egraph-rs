import {getModule} from '.'
import {alloc, free} from './allocator'
import {Graph} from './graph'

export class EdgeConcentration {
  constructor () {
    this.functions = getModule().functions
  }

  call (graph, biclusters) {
    const result = this.functions.edgeConcentration(graph.pointer, biclusters.pointer)
    const transformed = new Graph(result)
    for (let i = 0; i < graph.nodeCount(); ++i) {
      transformed.nodeProperties.set(i, graph.nodeProperties.get(i))
    }
    return transformed
  }
}

export class InterGroupEdgeConcentration {
  constructor () {
    const {Module, functions} = getModule()
    this.Module = Module
    this.functions = functions
  }

  call (graph, groups, biclustering) {
    const pGroups = alloc(4 * groups.length)
    groups.forEach((g, i) => {
      this.Module.HEAPU32[pGroups / 4 + i] = g
    })
    const result = this.functions.interGroupEdgeConcentration(graph.pointer, pGroups, biclustering.pointer)
    free(pGroups)
    return new Graph(result)
  }
}
