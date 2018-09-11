import {getModule} from '.'
import {alloc, free} from './allocator'

export class Group {
  constructor (pointer) {
    this.functions = getModule().functions
    this.pointer = pointer
  }

  delete () {
    free(this.pointer)
  }

  x (i) {
    return this.functions.groupX(this.pointer, i)
  }

  y (i) {
    return this.functions.groupY(this.pointer, i)
  }

  width (i) {
    return this.functions.groupWidth(this.pointer, i)
  }

  height (i) {
    return this.functions.groupHeight(this.pointer, i)
  }
}

const applyGrouping = (Module, f, p, width, height, values) => {
  const pointer = alloc(8 * values.length)
  values.forEach((v, i) => {
    Module.HEAPF64[pointer / 8 + i] = v
  })
  const groups = new Group(f(p, width, height, pointer, values.length))
  const result = values.map((_, i) => {
    return {
      x: groups.x(i),
      y: groups.y(i),
      width: groups.width(i),
      height: groups.height(i)
    }
  })
  groups.delete()
  return result
}

export class ForceDirectedGrouping {
  constructor (graph) {
    const {Module, functions} = getModule()
    this.Module = Module
    this.functions = functions
    this.pointer = this.functions.force_directedGroupingNew(graph.pointer)
  }

  call (width, height, values) {
    return applyGrouping(this.Module, this.functions.force_directedGroupingCall, this.pointer, width, height, values)
  }
}

export class RadialGrouping {
  constructor () {
    const {Module, functions} = getModule()
    this.Module = Module
    this.functions = functions
    this.pointer = this.functions.radialGroupingNew()
  }

  call (width, height, values) {
    return applyGrouping(this.Module, this.functions.radialGroupingCall, this.pointer, width, height, values)
  }
}

export class TreemapGrouping {
  constructor () {
    const {Module, functions} = getModule()
    this.Module = Module
    this.functions = functions
    this.pointer = this.functions.treemapGroupingNew()
  }

  call (width, height, values) {
    const sumValues = values.reduce((a, b) => a + b, 0)
    const normalizedValues = values.map((v) => v / sumValues * width * height)
    return applyGrouping(this.Module, this.functions.treemapGroupingCall, this.pointer, width, height, normalizedValues)
  }
}
