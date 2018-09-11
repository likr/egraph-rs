import base from './egraph.js'

const state = {
  Module: null,
  functions: null
}

export const getModule = () => {
  if (!initialized()) {
    throw new Error('Module is not initialized')
  }
  return state
}

export const initialized = () => {
  return state.Module != null
}

const initializeFunctions = (Module) => {
  return {
    graphNew: Module.cwrap('graph_new', 'number', []),
    graphAddNode: Module.cwrap('graph_add_node', 'number', ['number']),
    graphAddEdge: Module.cwrap('graph_add_edge', 'number', ['number', 'number', 'number']),
    graphNodeCount: Module.cwrap('graph_node_count', 'number', ['number']),
    graphEdgeCount: Module.cwrap('graph_edge_count', 'number', ['number']),
    graphGetX: Module.cwrap('graph_get_x', 'number', ['number', 'number']),
    graphGetY: Module.cwrap('graph_get_y', 'number', ['number', 'number']),
    graphSetX: Module.cwrap('graph_set_x', 'void', ['number', 'number', 'number']),
    graphSetY: Module.cwrap('graph_set_y', 'void', ['number', 'number', 'number']),
    graphSource: Module.cwrap('graph_source', 'number', ['number', 'number']),
    graphTarget: Module.cwrap('graph_target', 'number', ['number', 'number']),
    graphNodeAt: Module.cwrap('graph_node_at', 'number', ['number', 'number']),
    graphEdgeAt: Module.cwrap('graph_edge_at', 'number', ['number', 'number']),
    nodeGetX: Module.cwrap('node_get_x', 'number', ['number']),
    nodeGetY: Module.cwrap('node_get_y', 'number', ['number']),
    nodeSetX: Module.cwrap('node_set_x', 'void', ['number', 'number']),
    nodeSetY: Module.cwrap('node_set_y', 'void', ['number', 'number']),
    edgeSource: Module.cwrap('edge_source', 'number', ['number']),
    edgeTarget: Module.cwrap('edge_target', 'number', ['number']),
    simulationNew: Module.cwrap('simulation_new', 'number', []),
    simulationAddCenterForce: Module.cwrap('simulation_add_center_force', 'number', ['number']),
    simulationAddGroupCenterForce: Module.cwrap('simulation_add_group_center_force', 'number', ['number', 'number', 'number', 'number', 'number']),
    simulationAddGroupLinkForce: Module.cwrap('simulation_add_group_link_force', 'number', ['number', 'number', 'number', 'number', 'number']),
    simulationAddGroupManyBodyForce: Module.cwrap('simulation_add_group_many_body_force', 'number', ['number', 'number', 'number', 'number', 'number']),
    simulationAddLinkForce: Module.cwrap('simulation_add_link_force', 'number', ['number', 'number']),
    simulationAddManyBodyForce: Module.cwrap('simulation_add_many_body_force', 'number', ['number']),
    simulationStart: Module.cwrap('simulation_start', 'void', ['number', 'number']),
    simulationGetStrength: Module.cwrap('simulation_get_strength', 'number', ['number']),
    simulationSetStrength: Module.cwrap('simulation_set_strength', 'void', ['number', 'number']),
    connectedComponents: Module.cwrap('connected_components', 'number', ['number']),
    squarifiedTreemap: Module.cwrap('squarified_treemap', 'number', ['number', 'number', 'number', 'number']),
    biclusterLength: Module.cwrap('bicluster_length', 'number', ['number']),
    biclusterSource: Module.cwrap('bicluster_source', 'number', ['number', 'number']),
    biclusterSourceLength: Module.cwrap('bicluster_source_length', 'number', ['number', 'number']),
    biclusterTarget: Module.cwrap('bicluster_target', 'number', ['number', 'number']),
    biclusterTargetLength: Module.cwrap('bicluster_target_length', 'number', ['number', 'number']),
    quasiBicliqueNew: Module.cwrap('quasi_biclique_new', 'number', []),
    quasiBicliqueCall: Module.cwrap('quasi_biclique_call', 'number', ['number', 'number', 'number', 'number', 'number', 'number']),
    quasiBicliqueGetMu: Module.cwrap('quasi_biclique_get_mu', 'number', ['number']),
    quasiBicliqueSetMu: Module.cwrap('quasi_biclique_set_mu', 'void', ['number', 'number']),
    quasiBicliqueGetMinSize: Module.cwrap('quasi_biclique_get_min_size', 'number', ['number']),
    quasiBicliqueSetMinSize: Module.cwrap('quasi_biclique_set_min_size', 'void', ['number', 'number']),
    edgeBundlingNew: Module.cwrap('edge_bundling_new', 'number', []),
    edgeBundlingCall: Module.cwrap('edge_bundling_call', 'number', ['number']),
    edgeBundlingGetCycles: Module.cwrap('edge_bundling_get_cycles', 'number', ['number']),
    edgeBundlingGetS0: Module.cwrap('edge_bundling_get_s0', 'number', ['number']),
    edgeBundlingGetI0: Module.cwrap('edge_bundling_get_i0', 'number', ['number']),
    edgeBundlingGetSStep: Module.cwrap('edge_bundling_get_s_step', 'number', ['number']),
    edgeBundlingGetIStep: Module.cwrap('edge_bundling_get_i_step', 'number', ['number']),
    edgeBundlingSetCycles: Module.cwrap('edge_bundling_set_cycles', 'void', ['number', 'number']),
    edgeBundlingSetS0: Module.cwrap('edge_bundling_set_s0', 'void', ['number', 'number']),
    edgeBundlingSetI0: Module.cwrap('edge_bundling_set_i0', 'void', ['number', 'number']),
    edgeBundlingSetSStep: Module.cwrap('edge_bundling_set_s_step', 'void', ['number', 'number']),
    edgeBundlingSetIStep: Module.cwrap('edge_bundling_set_i_step', 'void', ['number', 'number']),
    linesAt: Module.cwrap('lines_at', 'number', ['number', 'number']),
    linePoints: Module.cwrap('line_points', 'number', ['number']),
    linePointsAt: Module.cwrap('line_points_at', 'number', ['number', 'number']),
    linePointsLength: Module.cwrap('line_points_length', 'number', ['number']),
    pointX: Module.cwrap('point_x', 'number', ['number']),
    pointY: Module.cwrap('point_y', 'number', ['number']),
    edgeConcentration: Module.cwrap('edge_concentration', 'number', ['number', 'number']),
    interGroupEdgeConcentration: Module.cwrap('inter_group_edge_concentration_with_quasi_biclique', 'number', ['number', 'number', 'number']),
    groupX: Module.cwrap('group_x', 'number', ['number', 'number']),
    groupY: Module.cwrap('group_y', 'number', ['number', 'number']),
    groupWidth: Module.cwrap('group_width', 'number', ['number', 'number']),
    groupHeight: Module.cwrap('group_height', 'number', ['number', 'number']),
    force_directedGroupingNew: Module.cwrap('force_directed_grouping_new', 'number', ['number']),
    force_directedGroupingCall: Module.cwrap('force_directed_grouping_call', 'number', ['number', 'number', 'number', 'number']),
    radialGroupingNew: Module.cwrap('radial_grouping_new', 'number', []),
    radialGroupingCall: Module.cwrap('radial_grouping_call', 'number', ['number', 'number', 'number', 'number']),
    treemapGroupingNew: Module.cwrap('treemap_grouping_new', 'number', []),
    treemapGroupingCall: Module.cwrap('treemap_grouping_call', 'number', ['number', 'number', 'number', 'number']),
    alloc: Module.cwrap('rust_alloc', 'number', ['number']),
    free: Module.cwrap('rust_free', 'void', ['number'])
  }
}

export const load = (url = 'egraph.wasm') => {
  return new Promise((resolve, reject) => {
    window.fetch(url)
      .then((response) => response.arrayBuffer())
      .then((wasmBinary) => {
        base({wasmBinary}).then((Module) => {
          state.Module = Module
          state.functions = initializeFunctions(Module)
          resolve(state)
        })
      })
  })
}
