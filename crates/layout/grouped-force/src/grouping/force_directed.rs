use super::{aggregate_edges, aggregate_nodes, node_group, Group, GroupLink, GroupNode};
use petgraph::graph::{Graph, IndexType, NodeIndex, UnGraph};
use petgraph::EdgeType;
use petgraph_layout_force::position_force::NodeArgument;
use petgraph_layout_force::{CollideForce, LinkForce, ManyBodyForce, PositionForce};
use petgraph_layout_force_simulation::{apply_forces, initial_placement, Force, Point, Simulation};
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
        Box::new(PositionForce::new(&group_graph, |_, _| NodeArgument {
            strength: None,
            x: Some(0.),
            y: Some(0.),
        })),
        Box::new(CollideForce::new(
            &group_graph,
            |graph, u| graph[u].weight.sqrt(),
            0.7,
            1,
        )),
    ];

    let mut coordinates = initial_placement(&group_graph);
    let mut simulation = Simulation::new();
    simulation.run(&mut |alpha| apply_forces(&mut coordinates.points, &forces, alpha, 0.6));

    let mut result = HashMap::new();
    for (u, p) in coordinates.iter() {
        let size = group_graph[u].weight.sqrt();
        let Point { x, y, .. } = p;
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
