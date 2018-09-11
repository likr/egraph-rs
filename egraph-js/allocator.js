import {getModule} from '.'

export const alloc = (bytes) => {
  const {functions} = getModule()
  return functions.alloc(bytes)
}

export const free = (pointer) => {
  const {functions} = getModule()
  return functions.free(pointer)
}
