import {getModule} from '..'

export class FM3 {
  constructor () {
    this.functions = getModule().functions
    this.pointer = this.functions.layoutFM3New()
  }

  call (graph) {
    const start = Date.now()
    this.functions.layoutFM3Call(this.pointer, graph.pointer)
    const stop = Date.now()
    console.log(stop - start)
  }

  get minSize () {
    return this.functions.layoutFM3GetMinSize(this.pointer)
  }

  set minSize (value) {
    this.functions.layoutFM3SetMinSize(this.pointer, value)
  }

  get stepIteration () {
    return this.functions.layoutFM3GetStepIteration(this.pointer)
  }

  set stepIteration (value) {
    this.functions.layoutFM3SetStepIteration(this.pointer, value)
  }

  get unitEdgeLength () {
    return this.functions.layoutFM3GetUnitEdgeLength(this.pointer)
  }

  set unitEdgeLength (value) {
    this.functions.layoutFM3SetUnitEdgeLength(this.pointer, value)
  }
}
