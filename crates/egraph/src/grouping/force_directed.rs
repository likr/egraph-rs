use super::{aggregate_edges, aggregate_nodes, Group, GroupLink, GroupNode};
use crate::layout::force_directed::initial_placement;
use crate::layout::force_directed::simulation::Simulation;
use crate::Graph;
use std::collections::HashMap;
use std::marker::PhantomData;

pub struct ForceDirectedGrouping<D, G: Graph<D>> {
    pub group: Box<dyn Fn(&G, usize) -> usize>,
    pub node_weight: Box<dyn Fn(&G, usize) -> f32>,
    pub link_weight: Box<dyn Fn(&G, usize, usize) -> f32>,
    phantom: PhantomData<D>,
}

impl<D, G: Graph<D>> ForceDirectedGrouping<D, G> {
    pub fn new() -> ForceDirectedGrouping<D, G> {
        ForceDirectedGrouping {
            group: Box::new(|_, _| 0),
            node_weight: Box::new(|_, _| 1000.),
            link_weight: Box::new(|_, _, _| 1.),
            phantom: PhantomData,
        }
    }

    pub fn call(
        &self,
        graph: &G,
        simulation: &Simulation<D, G>,
        graph_creator: &Box<dyn Fn(&Vec<GroupNode>, &Vec<GroupLink>) -> G>,
    ) -> HashMap<usize, Group> {
        let group_nodes = aggregate_nodes(graph, &self.group, &self.node_weight);
        let group_links = aggregate_edges(graph, &self.group, &self.link_weight);
        let group_graph = graph_creator(&group_nodes, &group_links);

        let mut context = simulation.build(&group_graph);
        let mut points = initial_placement(group_graph.node_count());
        context.start(&mut points);

        let group_weight = group_nodes
            .iter()
            .map(|group| (group.id, group.weight))
            .collect::<HashMap<_, _>>();
        let mut result = HashMap::new();
        for (i, g) in group_graph.nodes().enumerate() {
            let size = group_weight[&g].sqrt();
            result.insert(
                g,
                Group {
                    x: points[i].x,
                    y: points[i].y,
                    width: size,
                    height: size,
                },
            );
        }
        result
    }
}
