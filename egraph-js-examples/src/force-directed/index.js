import egraph from 'egraph'

const layout = (Module, data) => {
  const forceDirected = Module.cwrap('force_directed', 'void', ['number', 'number', 'number', 'number'])

  const numVertices = data.nodes.length
  const numEdges = data.links.length
  const edges = Module._malloc(8 * numEdges)
  const results = Module._malloc(16 * numVertices + 64)

  data.links.forEach((link, i) => {
    Module.HEAPU32[edges / 4 + 2 * i] = link.source
    Module.HEAPU32[edges / 4 + 2 * i + 1] = link.target
  })

  const start = Date.now()
  forceDirected(numVertices, numEdges, edges, results)
  const stop = Date.now()

  data.nodes.forEach((node, i) => {
    node.x = Module.HEAPF64[results / 8 + 4 + 2 * i]
    node.y = Module.HEAPF64[results / 8 + 4 + 2 * i + 1]
  })

  console.log(stop - start)
}

const draw = (data) => {
  const renderer = document.querySelector('eg-renderer')
  const color = window.d3.scaleOrdinal(window.d3.schemeCategory20)
  for (const node of data.nodes) {
    node.fillColor = color(node.group)
  }
  for (const link of data.links) {
    link.strokeWidth = Math.sqrt(link.value)
  }
  renderer.load(data)
  renderer.center()
}

window.fetch('../egraph.wasm')
  .then((response) => response.arrayBuffer())
  .then((wasmBinary) => {
    egraph({wasmBinary})
      .then((Module) => {
        window.fetch('miserables.json')
          .then((response) => response.json())
          .then((data) => {
            layout(Module, data)
            draw(data)
          })
      })
  })
