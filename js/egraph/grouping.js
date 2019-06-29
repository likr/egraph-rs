import { Graph } from '.'
import { aggregateNodes, aggregateEdges } from '@egraph/grouping'

export * from '@egraph/grouping'

export class ForceDirectedGrouping {
  constructor() {
    this.group = () => 0
    this.nodeWeight = () => 1000
    this.edgeWeight = () => 1
  }

  call(graph, simulation) {
    const groupGraph = new Graph()
    for (const node of aggregateNodes(graph, this.group, this.nodeWeight)) {
      groupGraph.addNode(node.id, node)
    }
    for (const link of aggregateEdges(graph, this.group, this.edgeWeight)) {
      groupGraph.addEdge(link.source, link.target, link)
    }
    const layout = simulation.start(groupGraph)

    const result = {}
    for (const { id: u, x, y } of layout.nodes) {
      const size = Math.sqrt(groupGraph.node(u).weight)
      result[u] = {
        type: 'rect',
        x,
        y,
        width: size,
        height: size
      }
    }
    return result
  }
}
