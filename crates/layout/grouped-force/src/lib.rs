pub mod force;
pub mod grouping;

use crate::force::group_many_body_force::GroupManyBodyForceArgument;
use petgraph::graph::{Graph, IndexType, NodeIndex};
use petgraph::EdgeType;
use petgraph_layout_force::link_force::LinkArgument;
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
        Box::new(force::GroupManyBodyForce::new(&graph, |_, u| {
            GroupManyBodyForceArgument {
                group: groups[&u],
                strength: Some(-30.),
            }
        })),
        Box::new(force::GroupLinkForce::new(
            &graph,
            |_, u| groups[&u],
            |_, _| LinkArgument {
                distance: Some(30.),
                strength: Some(0.1),
            },
            |_, _| LinkArgument {
                distance: Some(30.),
                strength: Some(0.01),
            },
        )),
        Box::new(force::GroupPositionForce::new(
            &graph,
            |_, u| force::group_position_force::NodeArgument {
                group: groups[&u],
                strength: 0.1,
            },
            |_, g| force::group_position_force::GroupArgument {
                x: group_pos[&g].x,
                y: group_pos[&g].y,
            },
        )),
        // Box::new(force::GroupCenterForce::new(
        //     &graph,
        //     |_, u| groups[&u],
        //     |g| group_pos[&g].x,
        //     |g| group_pos[&g].y,
        // )),
    ]
}
