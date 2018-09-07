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
