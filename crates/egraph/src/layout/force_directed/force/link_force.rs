use crate::graph::{degree, Graph, NodeIndex};
use crate::layout::force_directed::force::{Force, ForceContext, Point};
use std::collections::HashMap;
use std::marker::PhantomData;

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

pub struct LinkForceContext {
    links: Vec<Link>,
}

impl LinkForceContext {
    pub fn new(links: Vec<Link>) -> LinkForceContext {
        LinkForceContext { links }
    }
}

impl ForceContext for LinkForceContext {
    fn apply(&self, points: &mut Vec<Point>, alpha: f32) {
        let links = &self.links;
        for link in links {
            let source = points[link.source_index];
            let target = points[link.target_index];
            let dx = (target.x + target.vx) - (source.x + source.vx);
            let dy = (target.y + target.vy) - (source.y + source.vy);
            let l = (dx * dx + dy * dy).sqrt().max(1e-6);
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

pub struct LinkForce<D, G: Graph<D>> {
    pub strength: Box<dyn Fn(&G, NodeIndex, NodeIndex) -> f32>,
    pub distance: Box<dyn Fn(&G, NodeIndex, NodeIndex) -> f32>,
    phantom: PhantomData<D>,
}

impl<D, G: Graph<D>> LinkForce<D, G> {
    pub fn new() -> LinkForce<D, G> {
        LinkForce {
            strength: Box::new(|graph, u, v| {
                let source_degree = degree(graph, u);
                let target_degree = degree(graph, v);
                1. / (source_degree.min(target_degree)) as f32
            }),
            distance: Box::new(|_, _, _| 30.0),
            phantom: PhantomData,
        }
    }
}

impl<D, G: Graph<D>> Force<D, G> for LinkForce<D, G> {
    fn build(&self, graph: &G) -> Box<dyn ForceContext> {
        let distance_accessor = &self.distance;
        let strength_accessor = &self.strength;
        let node_indices = graph
            .nodes()
            .enumerate()
            .map(|(i, u)| (u, i))
            .collect::<HashMap<_, _>>();
        let links = graph
            .edges()
            .map(|(u, v)| {
                let distance = distance_accessor(graph, u, v);
                let strength = strength_accessor(graph, u, v);
                let source_degree = degree(graph, u) as f32;
                let target_degree = degree(graph, v) as f32;
                let bias = source_degree / (source_degree + target_degree);
                Link::new(node_indices[&u], node_indices[&v], distance, strength, bias)
            })
            .collect();
        Box::new(LinkForceContext::new(links))
    }
}

#[test]
fn test_link() {
    let mut links = Vec::new();
    links.push(Link::new(0, 1, 30., 0.5, 0.5));
    let context = LinkForceContext::new(links);
    let mut points = Vec::new();
    points.push(Point::new(-10., 0.));
    points.push(Point::new(10., 0.));
    context.apply(&mut points, 1.);
    assert_eq!(points[0].vx, -2.5);
    assert_eq!(points[0].vy, 0.);
    assert_eq!(points[1].vx, 2.5);
    assert_eq!(points[1].vy, 0.);
}
