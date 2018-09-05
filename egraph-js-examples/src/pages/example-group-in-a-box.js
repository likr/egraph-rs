import React from 'react'
import * as d3 from 'd3'
import {Algorithms} from 'egraph/algorithms'
import {Allocator} from 'egraph/allocator'
import {Simulation} from 'egraph/layout/force-directed'
import {Graph} from 'egraph/graph'
import {ForceDirectedGrouping} from 'egraph/grouping'
import {EdgeBundling} from 'egraph/edge-bundling'
import {loadModule} from '../module'
import {Wrapper} from '../wrapper'

const countGroups = (nodes) => {
  const groupCount = new Map()
  for (const node of nodes) {
    if (!groupCount.has(node.group)) {
      groupCount.set(node.group, 0)
    }
    groupCount.set(node.group, groupCount.get(node.group) + 1)
  }
  const groups = Array.from(groupCount.entries()).map(([name, count]) => ({name, count}))
  groups.sort((a, b) => b.count - a.count)
  return groups
}

const rectGroups = (groups, width, height, Module) => {
  const algorithms = new Algorithms(Module)
  const values = groups.map(({count}) => count)
  const sumValues = values.reduce((a, b) => a + b)
  const normalizedValues = values.map((v) => v / sumValues * width * height)
  return algorithms.squarifiedTreemap(width, height, normalizedValues)
}

const circleGroups = (groups, width, height) => {
  const tree = {
    name: '',
    children: groups.map(({name, count}) => {
      return {
        name,
        size: count
      }
    })
  }
  const root = d3.hierarchy(tree)
    .sum((d) => d.size)
    .sort((a, b) => b.value - a.value)
  const pack = d3.pack().size([width, height])
  const tiles = pack(root).descendants()
    .map((node) => {
      return {
        x: node.x - node.r,
        y: node.y - node.r,
        width: node.r * 2,
        height: node.r * 2
      }
    })
  tiles.shift(0)
  for (const tile of tiles) {
    tile.type = 'circle'
  }
  return tiles
}

const forceDirectedGroups = (groups, width, height, Module, data) => {
  const groupGraph = new Graph(Module)
  const nodeGroups = new Map(data.nodes.map((node, i) => [i, node.group]))
  const n = groups.length
  groups.forEach(() => {
    groupGraph.addNode()
  })
  for (let i = 0; i < n; ++i) {
    const g1 = groups[i]
    for (let j = i + 1; j < n; ++j) {
      const g2 = groups[j]
      const interGroupLinks = data.links.filter(({source, target}) => {
        return (nodeGroups.get(source) === g1.name && nodeGroups.get(target) === g2.name) || (nodeGroups.get(source) === g2.name && nodeGroups.get(target) === g1.name)
      })
      if (interGroupLinks.length > 0) {
        groupGraph.addEdge(i, j)
      }
    }
  }
  const grouping = new ForceDirectedGrouping(Module, groupGraph)
  const values = groups.map(({count}) => count)
  const tiles = grouping.call(width, height, values)
  for (const tile of tiles) {
    tile.type = 'circle'
  }
  return tiles
}

const layoutGroups = (type, groups, width, height, Module, data) => {
  switch (type) {
    case 'circle':
      return circleGroups(groups, width, height)
    case 'force-directed':
      return forceDirectedGroups(groups, width, height, Module, data)
    case 'rect':
      return rectGroups(groups, width, height, Module)
  }
  throw new Error(`Unsupported layout type: ${type}`)
}

const layout = (Module, graph, data, options) => {
  const width = 800
  const height = 600
  const allocator = new Allocator(Module)

  const groups = countGroups(data.nodes)
  const tiles = layoutGroups(options.type, groups, width, height, Module, data)

  const groupsPointer = allocator.alloc(16 * groups.length)
  tiles.forEach((tile, i) => {
    Module.HEAPF32[groupsPointer / 4 + 2 * i] = tile.x + tile.width / 2
    Module.HEAPF32[groupsPointer / 4 + 2 * i + 1] = tile.y + tile.height / 2
  })

  const groupMap = new Map(groups.map(({name}, i) => [name, i]))
  const nodeGroupsPointer = allocator.alloc(4 * graph.nodeCount())
  data.nodes.forEach((node, i) => {
    Module.HEAPU32[nodeGroupsPointer / 4 + i] = groupMap.get(node.group)
  })

  const simulation = new Simulation(Module)
  const f1 = simulation.addGroupManyBodyForce(groupsPointer, groups.length, nodeGroupsPointer, graph.nodeCount())
  const f2 = simulation.addGroupLinkForce(graph, nodeGroupsPointer)
  const f3 = simulation.addGroupCenterForce(groupsPointer, groups.length, nodeGroupsPointer, graph.nodeCount())
  simulation.setStrength(f1, 0.2)
  simulation.setStrength(f2, 0.1)
  simulation.setStrength(f3, 0.3)
  simulation.start(graph)

  const edgeBundling = new EdgeBundling(Module)
  edgeBundling.cycles = options.cycles
  edgeBundling.s0 = options.s0
  edgeBundling.i0 = options.i0
  edgeBundling.sStep = options.sStep
  edgeBundling.iStep = options.iStep
  const lines = edgeBundling.call(graph)

  tiles.forEach((tile, i) => {
    tile.label = groups[i].name.toString()
    tile.x += tile.width / 2
    tile.y += tile.height / 2
  })
  data.groups = tiles

  data.nodes.forEach((node, i) => {
    node.x = graph.getX(i)
    node.y = graph.getY(i)
  })

  data.links.forEach((link, i) => {
    link.bends = lines[i].map(({x, y}) => [x, y])
  })
}

export class ExampleGroupInABox extends React.Component {
  componentDidMount () {
    window.fetch('/data/miserables.json')
      .then((response) => response.json())
      .then((data) => {
        const color = d3.scaleOrdinal(d3.schemeCategory10)
        for (const node of data.nodes) {
          node.fillColor = color(node.group)
        }
        for (const link of data.links) {
          link.strokeWidth = Math.sqrt(link.value)
        }

        this.data = data
        this.layout().then(() => {
          this.refs.renderer.center()
        })
      })
  }

  render () {
    return <div>
      <div>
        <Wrapper onResize={this.handleResize.bind(this)}>
          <eg-renderer
            ref='renderer'
            transition-duration='1000'
            default-node-width='10'
            default-node-height='10'
            default-node-stroke-color='#fff'
            default-node-stroke-width='1.5'
            default-node-type='circle'
            default-link-stroke-color='#999'
            default-link-stroke-opacity='0.6'
            node-label-property='name'
            no-auto-centering
          />
        </Wrapper>
      </div>
      <div>
        <form onSubmit={this.handleSubmitOptionsForm.bind(this)}>
          <h3 className='title'>Group-In-a-Box Options</h3>
          <div className='field'>
            <label className='label'>Type</label>
            <div className='control'>
              <div className='select is-fullwidth'>
                <select ref='type' defaultValue='rect'>
                  <option value='rect'>Treemap</option>
                  <option value='circle'>Circle Packing</option>
                  <option value='force-directed'>Force-directed</option>
                </select>
              </div>
            </div>
          </div>
          <h3 className='title'>Edge Bundling Options</h3>
          <div className='field'>
            <label className='label'>Cycle</label>
            <div className='control'>
              <input ref='cycles' className='input' type='number' min='1' step='1' defaultValue='6' />
            </div>
          </div>
          <div className='field'>
            <label className='label'>S0</label>
            <div className='control'>
              <input ref='s0' className='input' type='number' min='0' step='0.01' defaultValue='0.1' />
            </div>
          </div>
          <div className='field'>
            <label className='label'>S Step</label>
            <div className='control'>
              <input ref='sStep' className='input' type='number' min='Step' step='0.01' defaultValue='0.5' />
            </div>
          </div>
          <div className='field'>
            <label className='label'>I0</label>
            <div className='control'>
              <input ref='i0' className='input' type='number' min='0' step='1' defaultValue='90' />
            </div>
          </div>
          <div className='field'>
            <label className='label'>I Step</label>
            <div className='control'>
              <input ref='iStep' className='input' type='number' min='Step' step='0.01' defaultValue='0.6' />
            </div>
          </div>
          <div className='field'>
            <div className='control'>
              <button className='button' type='submit'>Update</button>
            </div>
          </div>
        </form>
      </div>
    </div>
  }

  handleResize (width, height) {
    this.refs.renderer.width = width
    this.refs.renderer.height = height
  }

  handleSubmitOptionsForm (event) {
    event.preventDefault()
    this.layout()
  }

  layout () {
    return loadModule().then(({Module}) => {
      const graph = new Graph(Module)
      this.data.nodes.forEach(() => {
        graph.addNode()
      })
      for (const {source, target} of this.data.links) {
        graph.addEdge(source, target)
      }

      layout(Module, graph, this.data, {
        type: this.refs.type.value,
        cycles: +this.refs.cycles.value,
        s0: +this.refs.s0.value,
        i0: +this.refs.i0.value,
        sStep: +this.refs.sStep.value,
        iStep: +this.refs.iStep.value
      })

      this.refs.renderer.load(this.data)
    })
  }
}
