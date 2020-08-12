use super::{aggregate_edges, aggregate_nodes, node_group, Group, GroupLink, GroupNode};
use petgraph::graph::{Graph, IndexType, NodeIndex, UnGraph};
use petgraph::EdgeType;
use petgraph_layout_force_simulation::force::{
    CollideForce, LinkForce, ManyBodyForce, PositionForce,
};
use petgraph_layout_force_simulation::{initial_placement, Force, Simulation};
use std::collections::HashMap;

fn create_group_graph(
    nodes: &Vec<GroupNode>,
    links: &Vec<GroupLink>,
) -> UnGraph<GroupNode, GroupLink> {
    let mut graph = Graph::new_undirected();
    let indices = nodes
        .iter()
        .map(|&node| (node.id, graph.add_node(node)))
        .collect::<HashMap<_, _>>();
    for &link in links {
        graph.add_edge(indices[&link.source], indices[&link.target], link);
    }
    graph
}

pub fn force_directed_grouping<
    N,
    E,
    Ty: EdgeType,
    Ix: IndexType,
    F: FnMut(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> usize,
>(
    graph: &Graph<N, E, Ty, Ix>,
    group_accessor: F,
) -> HashMap<usize, Group> {
    let groups = node_group(graph, group_accessor);
    let group_nodes = aggregate_nodes(graph, &groups, |_, _| 1000.);
    let group_links = aggregate_edges(graph, &groups, |_, _, _| 1.);
    let group_graph = create_group_graph(&group_nodes, &group_links);

    let forces: Vec<Box<dyn Force>> = vec![
        Box::new(ManyBodyForce::new(&group_graph)),
        Box::new(LinkForce::new(&group_graph)),
        Box::new(PositionForce::new(
            &group_graph,
            |_, _| 0.1,
            |_, _| Some(0.),
            |_, _| Some(0.),
        )),
        Box::new(CollideForce::new(
            &group_graph,
            |graph, u| graph[u].weight.sqrt(),
            0.7,
            1,
        )),
    ];

    let points = initial_placement(&group_graph);
    let mut simulation = Simulation::new(&group_graph, |_, u| points[&u]);
    let coordinates = simulation.run(&forces.as_slice());

    let mut result = HashMap::new();
    for u in group_graph.node_indices() {
        let size = group_graph[u].weight.sqrt();
        let (x, y) = coordinates[&u];
        result.insert(
            group_graph[u].id,
            Group {
                x,
                y,
                width: size,
                height: size,
            },
        );
    }
    result
}
