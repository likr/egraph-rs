import {Allocator} from './allocator'

export class Group {
  constructor (Module, pointer) {
    this.Module = Module
    this.module = {
      groupX: Module.cwrap('group_x', 'number', ['number', 'number']),
      groupY: Module.cwrap('group_y', 'number', ['number', 'number']),
      groupWidth: Module.cwrap('group_width', 'number', ['number', 'number']),
      groupHeight: Module.cwrap('group_height', 'number', ['number', 'number'])
    }
    this.pointer = pointer
  }

  delete () {
    const allocator = new Allocator(this.Module)
    allocator.free(this.pointer)
  }

  x (i) {
    return this.module.groupX(this.pointer, i)
  }

  y (i) {
    return this.module.groupY(this.pointer, i)
  }

  width (i) {
    return this.module.groupWidth(this.pointer, i)
  }

  height (i) {
    return this.module.groupHeight(this.pointer, i)
  }
}

const applyGrouping = (Module, f, p, width, height, values) => {
  const allocator = new Allocator(Module)
  const pointer = allocator.alloc(8 * values.length)
  values.forEach((v, i) => {
    Module.HEAPF64[pointer / 8 + i] = v
  })
  const groups = new Group(Module, f(p, width, height, pointer, values.length))
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
  constructor (Module, graph) {
    this.Module = Module
    this.module = {
      force_directedGroupingNew: Module.cwrap('force_directed_grouping_new', 'number', ['number']),
      force_directedGroupingCall: Module.cwrap('force_directed_grouping_call', 'number', ['number', 'number', 'number', 'number'])
    }
    this.pointer = this.module.force_directedGroupingNew(graph.pointer)
  }

  call (width, height, values) {
    return applyGrouping(this.Module, this.module.force_directedGroupingCall, this.pointer, width, height, values)
  }
}

export class RadialGrouping {
  constructor (Module) {
    this.Module = Module
    this.module = {
      radialGroupingNew: Module.cwrap('radial_grouping_new', 'number', []),
      radialGroupingCall: Module.cwrap('radial_grouping_call', 'number', ['number', 'number', 'number', 'number'])
    }
    this.pointer = this.module.radialGroupingNew()
  }

  call (width, height, values) {
    return applyGrouping(this.Module, this.module.radialGroupingCall, this.pointer, width, height, values)
  }
}

export class TreemapGrouping {
  constructor (Module) {
    this.Module = Module
    this.module = {
      treemapGroupingNew: Module.cwrap('treemap_grouping_new', 'number', []),
      treemapGroupingCall: Module.cwrap('treemap_grouping_call', 'number', ['number', 'number', 'number', 'number'])
    }
    this.pointer = this.module.treemapGroupingNew()
  }

  call (width, height, values) {
    const sumValues = values.reduce((a, b) => a + b, 0)
    const normalizedValues = values.map((v) => v / sumValues * width * height)
    return applyGrouping(this.Module, this.module.treemapGroupingCall, this.pointer, width, height, normalizedValues)
  }
}
