import egraph from 'egraph'

window.fetch('../egraph.wasm')
  .then((response) => response.arrayBuffer())
  .then((wasmBinary) => egraph({wasmBinary}))
  .then((module) => {
    console.log('loaded')
  })
