export class Simulation {
  constructor (Module) {
    this.module = {
      simulationNew: Module.cwrap('simulation_new', 'number', []),
      simulationAddCenterForce: Module.cwrap('simulation_add_center_force', 'number', ['number']),
      simulationAddGroupCenterForce: Module.cwrap('simulation_add_group_center_force', 'number', ['number', 'number', 'number', 'number', 'number']),
      simulationAddGroupLinkForce: Module.cwrap('simulation_add_group_link_force', 'number', ['number', 'number', 'number', 'number', 'number']),
      simulationAddGroupManyBodyForce: Module.cwrap('simulation_add_group_many_body_force', 'number', ['number', 'number', 'number', 'number', 'number']),
      simulationAddLinkForce: Module.cwrap('simulation_add_link_force', 'number', ['number', 'number']),
      simulationAddManyBodyForce: Module.cwrap('simulation_add_many_body_force', 'number', ['number']),
      simulationStart: Module.cwrap('simulation_start', 'void', ['number', 'number']),
      simulationGetStrength: Module.cwrap('simulation_get_strength', 'number', ['number']),
      simulationSetStrength: Module.cwrap('simulation_set_strength', 'void', ['number', 'number'])
    }
    this.pointer = this.module.simulationNew()
  }

  addCenterForce () {
    return this.module.simulationAddCenterForce(this.pointer)
  }

  addGroupCenterForce (groups, numGroups, nodeGroups, numNodes) {
    return this.module.simulationAddGroupCenterForce(this.pointer, groups, numGroups, nodeGroups, numNodes)
  }

  addGroupLinkForce (graph, nodeGroups, intraGroup = 0.5, interGroup = 0.01) {
    return this.module.simulationAddGroupLinkForce(this.pointer, graph.pointer, nodeGroups, intraGroup, interGroup)
  }

  addGroupManyBodyForce (groups, numGroups, nodeGroups, numNodes) {
    return this.module.simulationAddGroupManyBodyForce(this.pointer, groups, numGroups, nodeGroups, numNodes)
  }

  addLinkForce (graph) {
    return this.module.simulationAddLinkForce(this.pointer, graph.pointer)
  }

  addManyBodyForce () {
    return this.module.simulationAddManyBodyForce(this.pointer)
  }

  start (graph) {
    const start = Date.now()
    this.module.simulationStart(this.pointer, graph.pointer)
    const stop = Date.now()
    console.log(stop - start)
  }

  getStrength (index) {
    return this.module.simulationGetStrength(this.pointer, index)
  }

  setStrength (index, strength) {
    this.module.simulationSetStrength(this.pointer, index, strength)
  }
}
