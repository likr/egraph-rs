import {egraph} from 'egraph/loader'

const mod = egraph('/egraph.wasm')

export const loadModule = () => mod
