import {Allocator} from './allocator'

export class Biclusters {
  constructor (Module, pointer) {
    this.Module = Module
    this.module = {
      biclusterLength: Module.cwrap('bicluster_length', 'number', ['number']),
      biclusterSource: Module.cwrap('bicluster_source', 'number', ['number', 'number']),
      biclusterSourceLength: Module.cwrap('bicluster_source_length', 'number', ['number', 'number']),
      biclusterTarget: Module.cwrap('bicluster_target', 'number', ['number', 'number']),
      biclusterTargetLength: Module.cwrap('bicluster_target_length', 'number', ['number', 'number'])
    }
    this.pointer = pointer
  }

  source (i) {
    const allocator = new Allocator(this.Module)
    const n = this.module.biclusterSourceLength(this.pointer, i)
    const sourcePointer = this.module.biclusterSource(this.pointer, i)
    const result = new Array(n)
    for (let i = 0; i < n; ++i) {
      result[i] = this.Module.HEAPU32[sourcePointer / 4 + i]
    }
    allocator.free(sourcePointer)
    return result
  }

  target (i) {
    const allocator = new Allocator(this.Module)
    const n = this.module.biclusterSourceLength(this.pointer, i)
    const targetPointer = this.module.biclusterTarget(this.pointer, i)
    const result = new Array(n)
    for (let i = 0; i < n; ++i) {
      result[i] = this.Module.HEAPU32[targetPointer / 4 + i]
    }
    allocator.free(targetPointer)
    return result
  }

  get length () {
    return this.module.biclusterLength(this.pointer)
  }
}

export class QuasiBiclique {
  constructor (Module) {
    this.Module = Module
    this.module = {
      quasiBicliqueNew: Module.cwrap('quasi_biclique_new', 'number', []),
      quasiBicliqueCall: Module.cwrap('quasi_biclique_call', 'number', ['number', 'number', 'number', 'number', 'number', 'number']),
      quasiBicliqueGetMu: Module.cwrap('quasi_biclique_get_mu', 'number', ['number']),
      quasiBicliqueSetMu: Module.cwrap('quasi_biclique_set_mu', 'void', ['number', 'number']),
      quasiBicliqueGetMinSize: Module.cwrap('quasi_biclique_get_min_size', 'number', ['number']),
      quasiBicliqueSetMinSize: Module.cwrap('quasi_biclique_set_min_size', 'void', ['number', 'number'])
    }
    this.pointer = this.module.quasiBicliqueNew()
  }

  call (graph, source, target) {
    const allocator = new Allocator(this.Module)
    const sourcePointer = allocator.alloc(4 * source.length)
    source.map((u, i) => {
      this.Module.HEAPU32[sourcePointer / 4 + i] = u
    })
    const targetPointer = allocator.alloc(4 * target.length)
    target.map((v, i) => {
      this.Module.HEAPU32[targetPointer / 4 + i] = v
    })
    const biclusters = new Biclusters(this.Module, this.module.quasiBicliqueCall(this.pointer, graph.pointer, sourcePointer, source.length, targetPointer, target.length))
    allocator.free(sourcePointer)
    allocator.free(targetPointer)
    return biclusters
  }

  get mu () {
    return this.module.quasiBicliqueGetMu(this.pointer)
  }

  set mu (value) {
    this.module.quasiBicliqueSetMu(this.pointer, value)
  }

  get minSize () {
    return this.module.quasiBicliqueGetMinSize(this.pointer)
  }

  set minSize (value) {
    this.module.quasiBicliqueSetMinSize(this.pointer, value)
  }
}
