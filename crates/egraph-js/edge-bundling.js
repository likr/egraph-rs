import {getModule} from '.'

export class EdgeBundling {
  constructor () {
    const {Module, functions} = getModule()
    this.Module = Module
    this.functions = functions
    this.pointer = this.functions.edgeBundlingNew()
  }

  call (graph) {
    const linesPointer = this.functions.edgeBundlingCall(this.pointer, graph.pointer)
    const n = graph.edgeCount()
    const lines = new Array(n)
    for (let i = 0; i < n; ++i) {
      const linePointer = this.functions.linesAt(linesPointer, i)
      const len = this.functions.linePointsLength(linePointer)
      lines[i] = new Array(len - 2)
      for (let j = 1; j < len - 1; ++j) {
        const pointPointer = this.functions.linePointsAt(linePointer, j)
        lines[i][j - 1] = {
          x: this.functions.pointX(pointPointer),
          y: this.functions.pointY(pointPointer)
        }
      }
    }
    return lines
  }

  get cycles () {
    return this.functions.edgeBundlingGetCycles(this.pointer)
  }

  set cycles (value) {
    this.functions.edgeBundlingSetCycles(this.pointer, value)
  }

  get s0 () {
    return this.functions.edgeBundlingGetS0(this.pointer)
  }

  set s0 (value) {
    this.functions.edgeBundlingSetS0(this.pointer, value)
  }

  get i0 () {
    return this.functions.edgeBundlingGetI0(this.pointer)
  }

  set i0 (value) {
    this.functions.edgeBundlingSetI0(this.pointer, value)
  }

  get sStep () {
    return this.functions.edgeBundlingGetSStep(this.pointer)
  }

  set sStep (value) {
    this.functions.edgeBundlingSetSStep(this.pointer, value)
  }

  get iStep () {
    return this.functions.edgeBundlingGetIStep(this.pointer)
  }

  set iStep (value) {
    this.functions.edgeBundlingSetIStep(this.pointer, value)
  }
}
