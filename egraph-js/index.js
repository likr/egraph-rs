import base from './egraph'

const state = {
  Module: null
}

export const module = () => {
  if (!initialized()) {
    throw new Error('Module is not initialized')
  }
  return state.Module
}

export const initialized = () => {
  return state.Module != null
}

export const load = (url = 'egraph.wasm') => {
  return new Promise((resolve, reject) => {
    window.fetch(url)
      .then((response) => response.arrayBuffer())
      .then((wasmBinary) => {
        base({wasmBinary}).then((Module) => {
          state.Module = Module
          resolve({Module})
        })
      })
  })
}
