# egraph-wasm

egraph-wasm is a fast graph drawing library inspired by d3-force.

## Install

```shell-session
$ npm install egraph-wasm
```

## Usage

```javascript
import('egraph-wasm').then(mod => {
  // creating a graph
  const { Graph } = mod
  const graph = new Graph()

  // adding vertices
  const a = graph.addVertex()
  const b = graph.addVertex()
  const c = graph.addVertex()

  // adding edges
  graph.addEdge(a, b)
  graph.addEdge(a, c)
  graph.addEdge(b, c)

  // creating forces
  const { ManyBodyForce, LinkForce, CenterForce } = mod
  const manyBodyForce = new ManyBodyForce()
  const linkForce = new LinkForce()
  const centerForce = new CenterForce()

  // creating simulation
  const { Simulation } = mod
  const simulation = new Simulation()
  simulation.add(manyBodyForce.force())
  simulation.add(linkForce.force())
  simulation.add(centerForce.force())

  // position calculation
  const positions = simulation.start(graph)

  // printing result
  for (const i of graph.nodeIndices()) {
    console.log(positions[i])
  }
})
```

egraph-wasm is implemented using wasm-bindgen.
For more detailed usage, please read [wasm-bindgen document](https://rustwasm.github.io/wasm-bindgen/).


## Examples

### Drawing SVG with React.js

```javascript
(async () => {
  const mod = await import('egraph-wasm')
  const { Graph } = mod
  const graph = new Graph()
  const a = graph.addNode()
  const b = graph.addNode()
  const c = graph.addNode()
  graph.addEdge(a, b)
  graph.addEdge(a, c)
  graph.addEdge(b, c)

  const { ManyBodyForce, LinkForce, CenterForce } = mod
  const manyBodyForce = new ManyBodyForce()
  const linkForce = new LinkForce()
  linkForce.distance(() => 200)
  const centerForce = new CenterForce()

  const { Simulation } = mod
  const simulation = new Simulation()
  simulation.add(manyBodyForce.force())
  simulation.add(linkForce.force())
  simulation.add(centerForce.force())

  const positions = simulation.start(graph)

  const width = 600
  const height = 400
  render(<div>
    <svg width={width} height={height}>
      <g transform={`translate(${width / 2},${height / 2})`}>
        <g>
          {
            Array.from(graph.edgeIndices()).map(e => {
              const source = graph.source(e)
              const target = graph.target(e)
              const { x: x0, y: y0 } = positions[source]
              const { x: x1, y: y1 } = positions[target]
              return <g key={e}>
                <path d={`M ${x0} ${y0} L ${x1} ${y1}`} stroke='#000' />
              </g>
            })
          }
        </g>
        <g>
          {
            Array.from(graph.nodeIndices()).map(i => {
            const { x, y } = positions[i]
            return <g key={i} transform={`translate(${x},${y})`}>
              <circle r='10' fill='#000' />
            </g>
            })
          }
        </g>
      </g>
    </svg>
  </div>, document.querySelector('#display'))
})()
```

### Loading JSON

```javascript
(async () => {
  const mod = await import('egraph-wasm')
  const request = await fetch('miserables.json')
  const data = await request.json()

  const { Graph } = mod
  const graph = new Graph()
  const ids = new Map()
  for (const node of data.nodes) {
    ids.set(node.id, graph.addVertex(node))
  }
  for (const link of data.links) {
    graph.addEdge(ids.get(link.source), ids.get(link.target), link)
  }

  for (const node of graph.nodes()) {
    console.log(node)
  }
  for (const edge of graph.edges()) {
    console.log(edge)
  }
})()
```

## License

MIT
