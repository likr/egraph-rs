const egraph = require('./egraph')

egraph().then((Module) => {
  Module.cwrap('force_directed', 'void', ['number', 'number', 'number', 'number'])
})
