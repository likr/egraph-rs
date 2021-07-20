use crate::MIN_DISTANCE;
use petgraph::graph::{EdgeIndex, Graph, IndexType};
use petgraph::visit::EdgeRef;
use petgraph::EdgeType;
use petgraph_layout_force_simulation::{Force, ForceToNode, Point};
use std::collections::HashMap;

#[derive(Copy, Clone)]
pub struct LinkArgument {
    pub distance: Option<f32>,
    pub strength: Option<f32>,
}

impl LinkArgument {
    pub fn new() -> LinkArgument {
        LinkArgument {
            distance: None,
            strength: None,
        }
    }
}

pub struct Link {
    pub source_index: usize,
    pub target_index: usize,
    pub distance: f32,
    pub strength: f32,
    pub bias: f32,
}

impl Link {
    pub fn new(
        source_index: usize,
        target_index: usize,
        distance: f32,
        strength: f32,
        bias: f32,
    ) -> Link {
        Link {
            source_index,
            target_index,
            distance,
            strength,
            bias,
        }
    }
}

#[derive(Force)]
pub struct LinkForce {
    links: Vec<Link>,
    indices: Vec<(usize, usize)>,
}

impl LinkForce {
    pub fn new<N, E, Ty: EdgeType, Ix: IndexType>(graph: &Graph<N, E, Ty, Ix>) -> LinkForce {
        LinkForce::new_with_accessor(graph, |_, _| LinkArgument::new())
    }

    pub fn new_with_accessor<
        N,
        E,
        Ty: EdgeType,
        Ix: IndexType,
        F: FnMut(&Graph<N, E, Ty, Ix>, EdgeIndex<Ix>) -> LinkArgument,
    >(
        graph: &Graph<N, E, Ty, Ix>,
        mut accessor: F,
    ) -> LinkForce {
        let node_indices = graph
            .node_indices()
            .enumerate()
            .map(|(i, u)| (u, i))
            .collect::<HashMap<_, _>>();
        let mut degree = HashMap::new();
        for e in graph.edge_indices() {
            let (source, target) = graph.edge_endpoints(e).unwrap();
            *degree.entry(source).or_insert(0) += 1;
            *degree.entry(target).or_insert(0) += 1;
        }
        let mut indices = Vec::with_capacity(graph.node_count());
        let mut links = Vec::with_capacity(graph.edge_count());
        for u in graph.node_indices() {
            let start = links.len();
            for edge in graph.edges(u) {
                let e = edge.id();
                let v = if u == edge.source() {
                    edge.target()
                } else {
                    edge.source()
                };
                let argument = accessor(graph, e);
                let distance = if let Some(v) = argument.distance {
                    v
                } else {
                    default_distance_accessor(graph, e)
                };
                let strength = if let Some(v) = argument.strength {
                    v
                } else {
                    default_strength_accessor(graph, e)
                };
                let source_degree = degree[&u] as f32;
                let target_degree = degree[&v] as f32;
                let bias = source_degree / (source_degree + target_degree);
                links.push(Link::new(
                    node_indices[&u],
                    node_indices[&v],
                    distance,
                    strength,
                    bias,
                ));
            }
            let stop = links.len();
            indices.push((start, stop));
        }
        LinkForce { links, indices }
    }
}

impl ForceToNode for LinkForce {
    fn apply_to_node(&self, u: usize, points: &mut [Point], alpha: f32) {
        let (start, stop) = self.indices[u];
        for i in start..stop {
            let ref link = self.links[i];
            let source = points[link.source_index];
            let target = points[link.target_index];
            let dx = (target.x + target.vx) - (source.x + source.vx);
            let dy = (target.y + target.vy) - (source.y + source.vy);
            let l = (dx * dx + dy * dy).sqrt().max(MIN_DISTANCE);
            let w = (l - link.distance) / l * alpha * link.strength;
            let ref mut source = points[link.source_index];
            source.vx += dx * w * (1. - link.bias);
            source.vy += dy * w * (1. - link.bias);
        }
    }
}

impl AsRef<dyn Force> for LinkForce {
    fn as_ref(&self) -> &(dyn Force + 'static) {
        self
    }
}

pub fn default_strength_accessor<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
    e: EdgeIndex<Ix>,
) -> f32 {
    let (a, b) = graph.edge_endpoints(e).unwrap();
    let source_degree = graph.neighbors_undirected(a).count();
    let target_degree = graph.neighbors_undirected(b).count();
    1. / (source_degree.min(target_degree)) as f32
}

pub fn default_distance_accessor<N, E, Ty: EdgeType, Ix: IndexType>(
    _graph: &Graph<N, E, Ty, Ix>,
    _e: EdgeIndex<Ix>,
) -> f32 {
    30.
}

// #[test]
// fn test_link() {
//     let mut links = Vec::new();
//     links.push(Link::new(0, 1, 30., 0.5, 0.5));
//     let context = LinkForceContext::new(links);
//     let mut points = Vec::new();
//     points.push(Point::new(-10., 0.));
//     points.push(Point::new(10., 0.));
//     context.apply(&mut points, 1.);
//     assert_eq!(points[0].vx, -2.5);
//     assert_eq!(points[0].vy, 0.);
//     assert_eq!(points[1].vx, 2.5);
//     assert_eq!(points[1].vy, 0.);
// }
