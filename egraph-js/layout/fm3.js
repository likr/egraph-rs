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
}
