use crate::MIN_DISTANCE;
use petgraph::graph::{EdgeIndex, Graph, IndexType};
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

#[derive(Copy, Clone, Debug)]
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

pub struct LinkForce {
    links: Vec<Link>,
    indices: Vec<Vec<usize>>,
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
        let mut indices = vec![vec![]; graph.node_count()];
        let mut links = Vec::with_capacity(graph.edge_count());
        for (i, e) in graph.edge_indices().enumerate() {
            let (u, v) = graph.edge_endpoints(e).unwrap();
            indices[node_indices[&u]].push(i);
            indices[node_indices[&v]].push(i);
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
        LinkForce { links, indices }
    }
}

impl Force for LinkForce {
    fn apply(&self, points: &mut [Point], alpha: f32) {
        let links = &self.links;
        for link in links {
            let source = points[link.source_index];
            let target = points[link.target_index];
            let dx = (target.x + target.vx) - (source.x + source.vx);
            let dy = (target.y + target.vy) - (source.y + source.vy);
            let l = (dx * dx + dy * dy).sqrt().max(MIN_DISTANCE);
            let w = (l - link.distance) / l * alpha * link.strength;
            {
                let ref mut target = points[link.target_index];
                target.vx -= dx * w * link.bias;
                target.vy -= dy * w * link.bias;
            }
            {
                let ref mut source = points[link.source_index];
                source.vx += dx * w * (1. - link.bias);
                source.vy += dy * w * (1. - link.bias);
            }
        }
    }
}

impl ForceToNode for LinkForce {
    fn apply_to_node(&self, u: usize, points: &mut [Point], alpha: f32) {
        for &i in self.indices[u].iter() {
            let link = self.links[i];
            let source = points[link.source_index];
            let target = points[link.target_index];
            let dx = target.x + target.vx - (source.x + source.vx);
            let dy = target.y + target.vx - (source.y + source.vy);
            let l = (dx * dx + dy * dy).sqrt().max(MIN_DISTANCE);
            let w = (l - link.distance) / l * alpha * link.strength;
            if link.source_index == u {
                points[link.source_index].vx += dx * w * (1. - link.bias);
                points[link.source_index].vy += dy * w * (1. - link.bias);
            } else {
                points[link.target_index].vx -= dx * w * link.bias;
                points[link.target_index].vy -= dy * w * link.bias;
            }
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

#[test]
fn test_link() {
    use petgraph_layout_force_simulation::Coordinates;
    let mut graph = Graph::new_undirected();
    let u = graph.add_node(());
    let v = graph.add_node(());
    let w = graph.add_node(());
    graph.add_edge(u, v, ());
    graph.add_edge(v, w, ());
    let mut coordinates = Coordinates::initial_placement(&graph);
    let force = LinkForce::new(&graph);
    force.apply(&mut coordinates.points, 0.5);
}
