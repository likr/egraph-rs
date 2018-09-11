# egraph

Graph drawing library for the Web using Rust and WebAssembly.

# Example

```javascript
import {load} from 'egraph'
import {Graph} from 'egraph/graph'
import {Simulation} from 'egraph/layout/force-directed'

load('egraph.wasm').then(() => {
  const graph = new Graph()
  const u = graph.addNode()
  const v = graph.addNode()
  const w = graph.addNode()
  graph.addEdge(u, v)
  graph.addEdge(u, w)
  graph.addEdge(v, w)

  const simulation = new Simulation()
  simulation.addManyBodyForce()
  simulation.addLinkForce(graph)
  simulation.addCenterForce()
  simulation.start(graph)

  for (const node of graph.nodes) {
    console.log(node.x, node.y)
  }
})
```

# Demos

See https://egraph.likr-lab.com/
