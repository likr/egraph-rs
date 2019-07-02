# egraph

egraph is a fast graph drawing library implemented in Rust and WebAssembly.

## Install

```shell-session
$ npm install egraph
```

## Usage

```javascript
import { Graph } from 'egraph'
import {
  Simulation,
  ManyBodyForce,
  LinkForce,
  CenterForce
} from 'egraph/layout/force-directed'

// creating a graph
const graph = new Graph()

// adding vertices
graph.addNode(0)
graph.addNode(1)
graph.addNode(2)

// adding edges
graph.addEdge(0, 1)
graph.addEdge(0, 2)
graph.addEdge(1, 2)

// creating forces
const manyBodyForce = new ManyBodyForce()
const linkForce = new LinkForce()
const centerForce = new CenterForce()

// creating simulation
const simulation = new Simulation()
simulation.add(manyBodyForce)
simulation.add(linkForce)
simulation.add(centerForce)

// position calculation
const layout = simulation.start(graph)

// printing result
for (const u of graph.nodes()) {
  console.log(layout.nodes[u])
}
```

egraph-wasm is implemented using wasm-bindgen.
For more detailed usage, please read [wasm-bindgen document](https://rustwasm.github.io/wasm-bindgen/).

## Examples

### Drawing SVG with React.js

```javascript
import React from 'react'
import { render } from 'react-dom'
import { Graph } from 'egraph'
import {
  Simulation,
  ManyBodyForce,
  LinkForce,
  CenterForce
} from 'egraph/layout/force-directed'

const graph = new Graph()
graph.addNode(0)
graph.addNode(1)
graph.addNode(2)
graph.addEdge(0, 1)
graph.addEdge(0, 2)
graph.addEdge(1, 2)

const manyBodyForce = new ManyBodyForce()
const linkForce = new LinkForce()
linkForce.distance = () => 200
const centerForce = new CenterForce()

const simulation = new Simulation()
simulation.add(manyBodyForce)
simulation.add(linkForce)
simulation.add(centerForce)

const layout = simulation.start(graph)

const width = 600
const height = 400
render(
  <div>
    <svg width={width} height={height}>
      <g transform={`translate(${width / 2},${height / 2})`}>
        <g>
          {Array.from(graph.edges()).map(([u, v]) => {
            const { x: x0, y: y0 } = positions[u]
            const { x: x1, y: y1 } = positions[v]
            return (
              <g key={e}>
                <path d={`M ${x0} ${y0} L ${x1} ${y1}`} stroke='#000' />
              </g>
            )
          })}
        </g>
        <g>
          {Array.from(graph.nodes()).map((u) => {
            const { x, y } = positions[u]
            return (
              <g key={i} transform={`translate(${x},${y})`}>
                <circle r='10' fill='#000' />
              </g>
            )
          })}
        </g>
      </g>
    </svg>
  </div>,
  document.querySelector('#display')
)
```

### Loading JSON

```javascript
import { Graph } from 'egraph'
import {
  Simulation,
  ManyBodyForce,
  LinkForce,
  CenterForce
} from 'egraph/layout/force-directed'
;(async () => {
  const request = await fetch('miserables.json')
  const data = await request.json()

  const { Graph } = mod
  const graph = new Graph()
  for (const node of data.nodes) {
    graph.addNode(node.id, node)
  }
  for (const link of data.links) {
    graph.addEdge(link.source, link.target, link)
  }

  for (const u of graph.nodes()) {
    console.log(graph.node(u))
  }
  for (const [u, v] of graph.edges()) {
    console.log(graph.edge(u, v))
  }
})()
```

## License

MIT
