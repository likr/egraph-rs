pub mod force;
pub mod grouping;

use petgraph::graph::{Graph, IndexType, NodeIndex};
use petgraph::EdgeType;
use petgraph_layout_force_simulation::force::CenterForce;
use petgraph_layout_force_simulation::{Force, Point};
use std::collections::HashMap;

pub fn force_grouped<
    N,
    E,
    Ty: EdgeType,
    Ix: IndexType,
    F: FnMut(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> usize,
>(
    graph: &Graph<N, E, Ty, Ix>,
    mut group_accessor: F,
) -> Vec<Box<dyn Force>> {
    let groups = graph
        .node_indices()
        .map(|u| (u, group_accessor(graph, u)))
        .collect::<HashMap<_, _>>();
    let group_pos = grouping::force_directed_grouping(graph, |_, u| groups[&u]);
    vec![
        Box::new(force::GroupManyBodyForce::new(
            &graph,
            |_, _| -30.,
            |_, u| groups[&u],
        )),
        Box::new(force::GroupLinkForce::new(
            &graph,
            0.1,
            0.01,
            |_, u| groups[&u],
            |_, _, _| 30.,
        )),
        Box::new(force::GroupPositionForce::new(
            &graph,
            |_, _| 0.1,
            |_, u| groups[&u],
            |g| group_pos[&g].x,
            |g| group_pos[&g].y,
        )),
        Box::new(force::GroupCenterForce::new(
            &graph,
            |_, u| groups[&u],
            |g| group_pos[&g].x,
            |g| group_pos[&g].y,
        )),
        Box::new(CenterForce::new()),
    ]
}
