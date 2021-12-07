use crate::{Force, Point};
use petgraph::graph::{EdgeIndex, Graph, IndexType, NodeIndex};
use petgraph::EdgeType;
use petgraph_layout_force::link_force::{LinkArgument, LinkForce};
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
        F2: FnMut(&Graph<N, E, Ty, Ix>, EdgeIndex<Ix>) -> LinkArgument,
        F3: FnMut(&Graph<N, E, Ty, Ix>, EdgeIndex<Ix>) -> LinkArgument,
    >(
        graph: &Graph<N, E, Ty, Ix>,
        mut group_accessor: F1,
        mut inter_link_accessor: F2,
        mut intra_link_accessor: F3,
    ) -> GroupLinkForce {
        let groups = graph
            .node_indices()
            .map(|u| (u, group_accessor(graph, u)))
            .collect::<HashMap<_, _>>();
        GroupLinkForce {
            link_force: LinkForce::new_with_accessor(graph, |graph, e| {
                let (u, v) = graph.edge_endpoints(e).unwrap();
                if groups[&u] == groups[&v] {
                    intra_link_accessor(graph, e)
                } else {
                    inter_link_accessor(graph, e)
                }
            }),
        }
    }
}

impl Force for GroupLinkForce {
    fn apply(&self, points: &mut [Point], alpha: f32) {
        self.link_force.apply(points, alpha);
    }
}
