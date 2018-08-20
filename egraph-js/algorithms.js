import {Allocator} from './allocator'

export class Algorithms {
  constructor (Module) {
    this.Module = Module
    this.module = {
      connectedComponents: Module.cwrap('connected_components', 'number', ['number']),
      squarifiedTreemap: Module.cwrap('squarified_treemap', 'number', ['number', 'number', 'number', 'number'])
    }
  }

  connectedComponents (graph) {
    const allocator = new Allocator(this.Module)
    const components = this.module.connectedComponents(graph.pointer)
    const n = graph.nodeCount()
    const result = new Array(n)
    for (let i = 0; i < n; ++i) {
      result[i] = this.Module.HEAPU32[components / 4 + i]
    }
    allocator.free(components)
    return result
  }

  squarifiedTreemap (width, height, values) {
    const allocator = new Allocator(this.Module)
    const pointer = allocator.alloc(8 * values.length)
    values.forEach((v, i) => {
      this.Module.HEAPF64[pointer / 8 + i] = v
    })
    const tiles = this.module.squarifiedTreemap(width, height, pointer, values.length)
    const result = values.map((_, i) => {
      return {
        x: this.Module.HEAPF64[tiles / 8 + i * 4],
        y: this.Module.HEAPF64[tiles / 8 + i * 4 + 1],
        width: this.Module.HEAPF64[tiles / 8 + i * 4 + 2],
        height: this.Module.HEAPF64[tiles / 8 + i * 4 + 3]
      }
    })
    allocator.free(tiles)
    return result
  }
}
