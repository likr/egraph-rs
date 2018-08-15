export class Allocator {
  constructor (Module) {
    this.module = {
      alloc: Module.cwrap('rust_alloc', 'number', ['number']),
      free: Module.cwrap('rust_free', 'void', ['number'])
    }
  }

  alloc (bytes) {
    return this.module.alloc(bytes)
  }

  free (pointer) {
    this.module.free(pointer)
  }
}
