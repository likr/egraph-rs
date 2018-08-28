export class EdgeBundling {
  constructor (Module) {
    this.Module = Module
    this.module = {
      edgeBundlingNew: Module.cwrap('edge_bundling_new', 'number', []),
      edgeBundlingCall: Module.cwrap('edge_bundling_call', 'number', ['number']),
      edgeBundlingGetCycles: Module.cwrap('edge_bundling_get_cycles', 'number', ['number']),
      edgeBundlingGetS0: Module.cwrap('edge_bundling_get_s0', 'number', ['number']),
      edgeBundlingGetI0: Module.cwrap('edge_bundling_get_i0', 'number', ['number']),
      edgeBundlingGetSStep: Module.cwrap('edge_bundling_get_s_step', 'number', ['number']),
      edgeBundlingGetIStep: Module.cwrap('edge_bundling_get_i_step', 'number', ['number']),
      edgeBundlingSetCycles: Module.cwrap('edge_bundling_set_cycles', 'void', ['number', 'number']),
      edgeBundlingSetS0: Module.cwrap('edge_bundling_set_s0', 'void', ['number', 'number']),
      edgeBundlingSetI0: Module.cwrap('edge_bundling_set_i0', 'void', ['number', 'number']),
      edgeBundlingSetSStep: Module.cwrap('edge_bundling_set_s_step', 'void', ['number', 'number']),
      edgeBundlingSetIStep: Module.cwrap('edge_bundling_set_i_step', 'void', ['number', 'number']),
      linesAt: Module.cwrap('lines_at', 'number', ['number', 'number']),
      linePoints: Module.cwrap('line_points', 'number', ['number']),
      linePointsAt: Module.cwrap('line_points_at', 'number', ['number', 'number']),
      linePointsLength: Module.cwrap('line_points_length', 'number', ['number']),
      pointX: Module.cwrap('point_x', 'number', ['number']),
      pointY: Module.cwrap('point_y', 'number', ['number'])
    }
    this.pointer = this.module.edgeBundlingNew()
  }

  call (graph) {
    const linesPointer = this.module.edgeBundlingCall(this.pointer, graph.pointer)
    const n = graph.edgeCount()
    const lines = new Array(n)
    for (let i = 0; i < n; ++i) {
      const linePointer = this.module.linesAt(linesPointer, i)
      const len = this.module.linePointsLength(linePointer)
      lines[i] = new Array(len - 2)
      for (let j = 1; j < len - 1; ++j) {
        const pointPointer = this.module.linePointsAt(linePointer, j)
        lines[i][j - 1] = {
          x: this.module.pointX(pointPointer),
          y: this.module.pointY(pointPointer)
        }
      }
    }
    return lines
  }

  get cycles () {
    return this.module.edgeBundlingGetCycles(this.pointer)
  }

  set cycles (value) {
    this.module.edgeBundlingSetCycles(this.pointer, value)
  }

  get s0 () {
    return this.module.edgeBundlingGetS0(this.pointer)
  }

  set s0 (value) {
    this.module.edgeBundlingSetS0(this.pointer, value)
  }

  get i0 () {
    return this.module.edgeBundlingGetI0(this.pointer)
  }

  set i0 (value) {
    this.module.edgeBundlingSetI0(this.pointer, value)
  }

  get sStep () {
    return this.module.edgeBundlingGetSStep(this.pointer)
  }

  set sStep (value) {
    this.module.edgeBundlingSetSStep(this.pointer, value)
  }

  get iStep () {
    return this.module.edgeBundlingGetIStep(this.pointer)
  }

  set iStep (value) {
    this.module.edgeBundlingSetIStep(this.pointer, value)
  }
}
