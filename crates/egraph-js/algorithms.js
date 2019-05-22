import {getModule} from '.'
import {alloc, free} from './allocator'

export const connectedComponents = (graph) => {
  const {Module, functions} = module()
  const components = functions.connectedComponents(graph.pointer)
  const n = graph.nodeCount()
  const result = new Array(n)
  for (let i = 0; i < n; ++i) {
    result[i] = Module.HEAPU32[components / 4 + i]
  }
  free(components)
  return result
}

export const squarifiedTreemap = (width, height, values) => {
  const {Module, functions} = getModule()
  const pointer = alloc(8 * values.length)
  values.forEach((v, i) => {
    Module.HEAPF64[pointer / 8 + i] = v
  })
  const tiles = functions.squarifiedTreemap(width, height, pointer, values.length)
  const result = values.map((_, i) => {
    return {
      x: Module.HEAPF64[tiles / 8 + i * 4],
      y: Module.HEAPF64[tiles / 8 + i * 4 + 1],
      width: Module.HEAPF64[tiles / 8 + i * 4 + 2],
      height: Module.HEAPF64[tiles / 8 + i * 4 + 3]
    }
  })
  free(pointer)
  free(tiles)
  return result
}
