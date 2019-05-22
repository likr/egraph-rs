use crate::graph::{degree, Graph, NodeIndex};
use crate::layout::force_directed::force::{Force, ForceContext, Point};

pub struct Link {
    pub source: usize,
    pub target: usize,
    pub distance: f32,
    pub strength: f32,
    pub bias: f32,
}

impl Link {
    pub fn new(source: usize, target: usize, distance: f32, strength: f32, bias: f32) -> Link {
        Link {
            source,
            target,
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
            let source = points[link.source];
            let target = points[link.target];
            let dx = (target.x + target.vx) - (source.x + source.vx);
            let dy = (target.y + target.vy) - (source.y + source.vy);
            let l = (dx * dx + dy * dy).sqrt().max(1e-6);
            let w = (l - link.distance) / l * alpha * link.strength;
            {
                let ref mut target = points[link.target];
                target.vx -= dx * w * link.bias;
                target.vy -= dy * w * link.bias;
            }
            {
                let ref mut source = points[link.source];
                source.vx += dx * w * (1. - link.bias);
                source.vy += dy * w * (1. - link.bias);
            }
        }
    }
}

pub struct LinkForce<G> {
    pub strength: Box<Fn(&Graph<G>, NodeIndex, NodeIndex) -> f32>,
    pub distance: Box<Fn(&Graph<G>, NodeIndex, NodeIndex) -> f32>,
}

impl<G> LinkForce<G> {
    pub fn new() -> LinkForce<G> {
        LinkForce {
            strength: Box::new(|graph, u, v| {
                let source_degree = degree(graph, u);
                let target_degree = degree(graph, v);
                1. / (source_degree.min(target_degree)) as f32
            }),
            distance: Box::new(|_, _, _| 30.0),
        }
    }
}

impl<G> Force<G> for LinkForce<G> {
    fn build(&self, graph: &Graph<G>) -> Box<ForceContext> {
        let distance_accessor = &self.distance;
        let strength_accessor = &self.strength;
        let links = graph
            .edges()
            .map(|(u, v)| {
                let distance = distance_accessor(graph, u, v);
                let strength = strength_accessor(graph, u, v);
                let source_degree = degree(graph, u) as f32;
                let target_degree = degree(graph, v) as f32;
                let bias = source_degree / (source_degree + target_degree);
                Link::new(u, v, distance, strength, bias)
            })
            .collect();
        Box::new(LinkForceContext::new(links))
    }
}

#[test]
fn test_link() {
    let mut links = Vec::new();
    links.push(Link::new(0, 1));
    links.push(Link::new(1, 3));
    links.push(Link::new(3, 2));
    links.push(Link::new(2, 0));
    let mut force = LinkForce::new_with_links(links);
    force.strength = 0.0;
    let mut points = Vec::new();
    points.push(Point::new(10., 10.));
    points.push(Point::new(10., -10.));
    points.push(Point::new(-10., 10.));
    points.push(Point::new(-10., -10.));
    force.apply(&mut points, 1.);
    assert_eq!(points[0].vx, 2.5);
    assert_eq!(points[0].vy, 2.5);
    assert_eq!(points[1].vx, 2.5);
    assert_eq!(points[1].vy, -2.5);
    assert_eq!(points[2].vx, -2.5);
    assert_eq!(points[2].vy, 2.5);
    assert_eq!(points[3].vx, -2.5);
    assert_eq!(points[3].vy, -2.5);
}
