use crate::{Force, Point, MIN_DISTANCE};
use petgraph::graph::{Graph, IndexType, NodeIndex};
use petgraph::EdgeType;
use std::collections::HashMap;

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
}

impl LinkForce {
    pub fn new<N, E, Ty: EdgeType, Ix: IndexType>(graph: &Graph<N, E, Ty, Ix>) -> LinkForce {
        LinkForce::new_with_accessor(
            graph,
            |graph, u, v| {
                let source_degree = graph.neighbors_undirected(u).count();
                let target_degree = graph.neighbors_undirected(v).count();
                1. / (source_degree.min(target_degree)) as f32
            },
            |_, _, _| 30.0,
        )
    }

    pub fn new_with_accessor<
        N,
        E,
        Ty: EdgeType,
        Ix: IndexType,
        F: FnMut(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>, NodeIndex<Ix>) -> f32,
        G: FnMut(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>, NodeIndex<Ix>) -> f32,
    >(
        graph: &Graph<N, E, Ty, Ix>,
        mut strength_accessor: F,
        mut distance_accessor: G,
    ) -> LinkForce {
        let node_indices = graph
            .node_indices()
            .enumerate()
            .map(|(i, u)| (u, i))
            .collect::<HashMap<_, _>>();
        let links = graph
            .edge_indices()
            .map(|e| {
                let (u, v) = graph.edge_endpoints(e).unwrap();
                let distance = distance_accessor(graph, u, v);
                let strength = strength_accessor(graph, u, v);
                let source_degree = graph.neighbors_undirected(u).count() as f32;
                let target_degree = graph.neighbors_undirected(v).count() as f32;
                let bias = source_degree / (source_degree + target_degree);
                Link::new(node_indices[&u], node_indices[&v], distance, strength, bias)
            })
            .collect();
        LinkForce { links }
    }
}

impl Force for LinkForce {
    fn apply(&self, points: &mut Vec<Point>, alpha: f32) {
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

impl AsRef<dyn Force> for LinkForce {
    fn as_ref(&self) -> &(dyn Force + 'static) {
        self
    }
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
