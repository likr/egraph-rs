export class Simulation {
  constructor (Module) {
    this.module = {
      simulationNew: Module.cwrap('simulation_new', 'number', []),
      simulationAddCenterForce: Module.cwrap('simulation_add_center_force', 'void', ['number']),
      simulationAddGroupCenterForce: Module.cwrap('simulation_add_group_center_force', 'void', ['number', 'number', 'number', 'number', 'number']),
      simulationAddGroupLinkForce: Module.cwrap('simulation_add_group_link_force', 'void', ['number', 'number', 'number']),
      simulationAddGroupManyBodyForce: Module.cwrap('simulation_add_group_many_body_force', 'void', ['number', 'number', 'number', 'number', 'number']),
      simulationAddLinkForce: Module.cwrap('simulation_add_link_force', 'void', ['number', 'number']),
      simulationAddManyBodyForce: Module.cwrap('simulation_add_many_body_force', 'void', ['number']),
      simulationStart: Module.cwrap('simulation_start', 'void', ['number', 'number'])
    }
    this.pointer = this.module.simulationNew()
  }

  addCenterForce () {
    this.module.simulationAddCenterForce(this.pointer)
  }

  addGroupCenterForce (groups, numGroups, nodeGroups, numNodes) {
    this.module.simulationAddGroupCenterForce(this.pointer, groups, numGroups, nodeGroups, numNodes)
  }

  addGroupLinkForce (graph, nodeGroups) {
    this.module.simulationAddGroupLinkForce(this.pointer, graph.pointer, nodeGroups)
  }

  addGroupManyBodyForce (groups, numGroups, nodeGroups, numNodes) {
    this.module.simulationAddGroupManyBodyForce(this.pointer, groups, numGroups, nodeGroups, numNodes)
  }

  addLinkForce (graph) {
    this.module.simulationAddLinkForce(this.pointer, graph.pointer)
  }

  addManyBodyForce () {
    this.module.simulationAddManyBodyForce(this.pointer)
  }

  start (graph) {
    const start = Date.now()
    this.module.simulationStart(this.pointer, graph.pointer)
    const stop = Date.now()
    console.log(stop - start)
  }
}
