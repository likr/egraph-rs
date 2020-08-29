use crate::{Force, Point};
use petgraph::graph::{EdgeIndex, Graph, IndexType, NodeIndex};
use petgraph::EdgeType;
use petgraph_layout_force_simulation::force::LinkForce;
use std::collections::HashMap;

pub struct GroupLinkForce {
    link_force: LinkForce,
}

impl GroupLinkForce {
    pub fn new<
        N,
        E,
        Ty: EdgeType,
        Ix: IndexType,
        F1: FnMut(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> usize,
        F2: Fn(&Graph<N, E, Ty, Ix>, EdgeIndex<Ix>) -> f32,
    >(
        graph: &Graph<N, E, Ty, Ix>,
        intra_group: f32,
        inter_group: f32,
        mut group_accessor: F1,
        distance_accessor: F2,
    ) -> GroupLinkForce {
        let groups = graph
            .node_indices()
            .map(|u| (u, group_accessor(graph, u)))
            .collect::<HashMap<_, _>>();
        GroupLinkForce {
            link_force: LinkForce::new_with_accessor(
                graph,
                |_, e| {
                    let (u, v) = graph.edge_endpoints(e).unwrap();
                    if groups[&u] == groups[&v] {
                        intra_group
                    } else {
                        inter_group
                    }
                },
                distance_accessor,
            ),
        }
    }
}

impl Force for GroupLinkForce {
    fn apply(&self, points: &mut Vec<Point>, alpha: f32) {
        self.link_force.apply(points, alpha);
    }
}
