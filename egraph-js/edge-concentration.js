import {Allocator} from './allocator'
import {Graph} from './graph'

export class EdgeConcentration {
  constructor (Module) {
    this.Module = Module
    this.module = {
      edgeConcentration: Module.cwrap('edge_concentration', 'number', ['number', 'number'])
    }
  }

  call (graph, biclusters) {
    return new Graph(this.Module, this.module.edgeConcentration(graph.pointer, biclusters.pointer))
  }
}

export class InterGroupEdgeConcentration {
  constructor (Module) {
    this.Module = Module
    this.module = {
      interGroupEdgeConcentration: Module.cwrap('inter_group_edge_concentration', 'number', ['number', 'number', 'number'])
    }
  }

  call (graph, groups, biclustering) {
    const allocator = new Allocator(this.Module)
    const pGroups = allocator.alloc(4 * groups.length)
    groups.forEach((g, i) => {
      this.Module.HEAPU32[pGroups / 4 + i] = g
    })
    const result = this.module.interGroupEdgeConcentration(graph.pointer, pGroups, biclustering.pointer)
    allocator.free(pGroups)
    return new Graph(this.Module, result)
  }
}
