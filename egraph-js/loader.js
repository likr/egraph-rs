import base from '.'

export const egraph = (url = 'egraph.wasm') => {
  return new Promise((resolve, reject) => {
    window.fetch(url)
      .then((response) => response.arrayBuffer())
      .then((wasmBinary) => {
        base({wasmBinary}).then((Module) => {
          resolve({Module})
        })
      })
  })
}
