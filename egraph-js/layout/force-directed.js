export class Simulation {
  constructor (Module) {
    this.module = {
      simulationNew: Module.cwrap('simulation_new', 'number', []),
      simulationAddCenterForce: Module.cwrap('simulation_add_center_force', 'void', ['number']),
      simulationAddGroupForce: Module.cwrap('simulation_add_group_force', 'void', ['number', 'number', 'number', 'number', 'number']),
      simulationAddLinkForce: Module.cwrap('simulation_add_link_force', 'void', ['number', 'number']),
      simulationAddManyBodyForce: Module.cwrap('simulation_add_many_body_force', 'void', ['number']),
      simulationStart: Module.cwrap('simulation_start', 'void', ['number', 'number'])
    }
    this.pointer = this.module.simulationNew()
  }

  addCenterForce () {
    this.module.simulationAddCenterForce(this.pointer)
  }

  addGroupForce (groups, numGroups, nodeGroups, numNodes) {
    this.module.simulationAddGroupForce(this.pointer, groups, numGroups, nodeGroups, numNodes)
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
