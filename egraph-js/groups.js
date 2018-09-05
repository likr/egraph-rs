import {Allocator} from './allocator'

export class RadialGrouping {
  constructor (Module) {
    this.Module = Module
    this.module = {
      radialGroupingNew: Module.cwrap('radial_grouping_new', 'number', []),
      radialGroupingCall: Module.cwrap('radial_grouping_call', 'number', ['number', 'number', 'number', 'number']),
      groupX: Module.cwrap('group_x', 'number', ['number', 'number']),
      groupY: Module.cwrap('group_y', 'number', ['number', 'number']),
      groupWidth: Module.cwrap('group_width', 'number', ['number', 'number']),
      groupHeight: Module.cwrap('group_height', 'number', ['number', 'number'])
    }
    this.pointer = this.module.radialGroupingNew()
  }

  call (width, height, values) {
    const allocator = new Allocator(this.Module)
    const pointer = allocator.alloc(8 * values.length)
    values.forEach((v, i) => {
      this.Module.HEAPF64[pointer / 8 + i] = v
    })
    const groups = this.module.radialGroupingCall(this.pointer, width, height, pointer, values.length)
    const result = values.map((_, i) => {
      return {
        x: this.module.groupX(groups, i),
        y: this.module.groupY(groups, i),
        width: this.module.groupWidth(groups, i),
        height: this.module.groupHeight(groups, i)
      }
    })
    // allocator.free(pointer)
    // allocator.free(groups)
    return result
  }
}
