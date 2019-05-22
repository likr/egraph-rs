import {getModule} from '..'

export class Simulation {
  constructor (Module) {
    this.functions = getModule().functions
    this.pointer = this.functions.simulationNew()
  }

  addCenterForce () {
    return this.functions.simulationAddCenterForce(this.pointer)
  }

  addGroupCenterForce (groups, numGroups, nodeGroups, numNodes) {
    return this.functions.simulationAddGroupCenterForce(this.pointer, groups, numGroups, nodeGroups, numNodes)
  }

  addGroupLinkForce (graph, nodeGroups, intraGroup = 0.5, interGroup = 0.01) {
    return this.functions.simulationAddGroupLinkForce(this.pointer, graph.pointer, nodeGroups, intraGroup, interGroup)
  }

  addGroupManyBodyForce (groups, numGroups, nodeGroups, numNodes) {
    return this.functions.simulationAddGroupManyBodyForce(this.pointer, groups, numGroups, nodeGroups, numNodes)
  }

  addLinkForce (graph) {
    return this.functions.simulationAddLinkForce(this.pointer, graph.pointer)
  }

  addManyBodyForce () {
    return this.functions.simulationAddManyBodyForce(this.pointer)
  }

  start (graph) {
    const start = Date.now()
    this.functions.simulationStart(this.pointer, graph.pointer)
    const stop = Date.now()
    console.log(stop - start)
  }

  getStrength (index) {
    return this.functions.simulationGetStrength(this.pointer, index)
  }

  setStrength (index, strength) {
    this.functions.simulationSetStrength(this.pointer, index, strength)
  }
}
