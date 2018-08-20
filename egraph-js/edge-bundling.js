export class EdgeBundling {
  constructor (Module) {
    this.Module = Module
    this.module = {
      edgeBundling: Module.cwrap('edge_bundling', 'number', ['number']),
      linesAt: Module.cwrap('lines_at', 'number', ['number', 'number']),
      linePoints: Module.cwrap('line_points', 'number', ['number']),
      linePointsAt: Module.cwrap('line_points_at', 'number', ['number', 'number']),
      linePointsLength: Module.cwrap('line_points_length', 'number', ['number']),
      pointX: Module.cwrap('point_x', 'number', ['number']),
      pointY: Module.cwrap('point_y', 'number', ['number'])
    }
  }

  call (graph) {
    const linesPointer = this.module.edgeBundling(graph.pointer)
    const n = graph.edgeCount()
    const lines = new Array(n)
    for (let i = 0; i < n; ++i) {
      const linePointer = this.module.linesAt(linesPointer, i)
      const len = this.module.linePointsLength(linePointer)
      lines[i] = new Array(len)
      for (let j = 0; j < len; ++j) {
        const pointPointer = this.module.linePointsAt(linePointer, j)
        lines[i][j] = {
          x: this.module.pointX(pointPointer),
          y: this.module.pointY(pointPointer)
        }
      }
    }
    return lines
  }
}
