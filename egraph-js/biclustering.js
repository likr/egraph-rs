import {getModule} from '.'
import {alloc, free} from './allocator'

export class Biclusters {
  constructor (pointer) {
    const {Module, functions} = getModule()
    this.Module = Module
    this.functions = functions
    this.pointer = pointer
  }

  source (i) {
    const n = this.functions.biclusterSourceLength(this.pointer, i)
    const sourcePointer = this.functions.biclusterSource(this.pointer, i)
    const result = new Array(n)
    for (let i = 0; i < n; ++i) {
      result[i] = this.Module.HEAPU32[sourcePointer / 4 + i]
    }
    free(sourcePointer)
    return result
  }

  target (i) {
    const n = this.functions.biclusterSourceLength(this.pointer, i)
    const targetPointer = this.functions.biclusterTarget(this.pointer, i)
    const result = new Array(n)
    for (let i = 0; i < n; ++i) {
      result[i] = this.Module.HEAPU32[targetPointer / 4 + i]
    }
    free(targetPointer)
    return result
  }

  get length () {
    return this.functions.biclusterLength(this.pointer)
  }
}

export class QuasiBiclique {
  constructor () {
    const {Module, functions} = getModule()
    this.Module = Module
    this.functions = functions
    this.pointer = this.functions.quasiBicliqueNew()
  }

  call (graph, source, target) {
    const sourcePointer = alloc(4 * source.length)
    source.map((u, i) => {
      this.Module.HEAPU32[sourcePointer / 4 + i] = u
    })
    const targetPointer = alloc(4 * target.length)
    target.map((v, i) => {
      this.Module.HEAPU32[targetPointer / 4 + i] = v
    })
    const biclusters = new Biclusters(this.functions.quasiBicliqueCall(this.pointer, graph.pointer, sourcePointer, source.length, targetPointer, target.length))
    free(sourcePointer)
    free(targetPointer)
    return biclusters
  }

  get mu () {
    return this.functions.quasiBicliqueGetMu(this.pointer)
  }

  set mu (value) {
    this.functions.quasiBicliqueSetMu(this.pointer, value)
  }

  get minSize () {
    return this.functions.quasiBicliqueGetMinSize(this.pointer)
  }

  set minSize (value) {
    this.functions.quasiBicliqueSetMinSize(this.pointer, value)
  }
}
