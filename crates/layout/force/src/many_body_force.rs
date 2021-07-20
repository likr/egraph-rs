use crate::{Force, ForceToNode, Point, MIN_DISTANCE};
use petgraph::graph::{Graph, IndexType, NodeIndex};
use petgraph::EdgeType;
use quadtree::{Element, NodeId, Quadtree, Rect};

#[derive(Copy, Clone, Debug)]
struct Body {
    x: f32,
    y: f32,
    strength: f32,
}

impl Body {
    fn new(x: f32, y: f32, strength: f32) -> Body {
        Body { x, y, strength }
    }
}

impl Default for Body {
    fn default() -> Body {
        Body::new(0., 0., 0.)
    }
}

fn accumulate(tree: &mut Quadtree<Body>, node_id: NodeId) {
    let mut sum_weight = 0.;
    let mut sum_strength = 0.;
    let mut sum_x = 0.;
    let mut sum_y = 0.;
    for &(ref e, _) in tree.elements(node_id).iter() {
        match **e {
            Element::Leaf { x, y, n, value } => {
                let strength = value * n as f32;
                let weight = strength.abs();
                sum_strength += strength;
                sum_weight += weight;
                sum_x += x * weight;
                sum_y += y * weight;
            }
            Element::Node { node_id } => {
                accumulate(tree, node_id);
                let data = tree.data(node_id);
                let strength = data.strength;
                let weight = strength.abs();
                sum_strength += strength;
                sum_weight += weight;
                sum_x += data.x * weight;
                sum_y += data.y * weight;
            }
            Element::Empty => {}
        }
    }
    let data = tree.data_mut(node_id);
    data.strength = sum_strength;
    data.x = sum_x / sum_weight;
    data.y = sum_y / sum_weight;
}

fn apply_many_body(
    point: &mut Point,
    tree: &Quadtree<Body>,
    node_id: NodeId,
    alpha: f32,
    theta2: f32,
) {
    for &(ref e, _) in tree.elements(node_id).iter() {
        match **e {
            Element::Node { node_id } => {
                let data = tree.data(node_id);
                let rect = tree.rect(node_id);
                let dx = rect.cx - point.x;
                let dy = rect.cy - point.y;
                let l = (dx * dx + dy * dy).max(MIN_DISTANCE);
                if rect.width * rect.height / theta2 < l {
                    point.vx += dx * data.strength * alpha / l;
                    point.vy += dy * data.strength * alpha / l;
                } else {
                    apply_many_body(point, tree, node_id, alpha, theta2);
                }
            }
            Element::Leaf { x, y, n, value } => {
                if x != point.x || y != point.y {
                    let strength = value * n as f32;
                    let dx = x - point.x;
                    let dy = y - point.y;
                    let l = (dx * dx + dy * dy).max(MIN_DISTANCE);
                    point.vx += dx * strength * alpha / l;
                    point.vy += dy * strength * alpha / l;
                }
            }
            Element::Empty => {}
        }
    }
}

pub struct ManyBodyForce {
    strength: Vec<f32>,
}

impl ManyBodyForce {
    pub fn new<N, E, Ty: EdgeType, Ix: IndexType>(graph: &Graph<N, E, Ty, Ix>) -> ManyBodyForce {
        ManyBodyForce::new_with_accessor(graph, |_, _| None)
    }

    pub fn new_with_accessor<
        N,
        E,
        Ty: EdgeType,
        Ix: IndexType,
        F: Fn(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> Option<f32>,
    >(
        graph: &Graph<N, E, Ty, Ix>,
        strength_accessor: F,
    ) -> ManyBodyForce {
        let strength = graph
            .node_indices()
            .map(|u| {
                if let Some(v) = strength_accessor(graph, u) {
                    v
                } else {
                    default_strength_accessor(graph, u)
                }
            })
            .collect();
        ManyBodyForce { strength }
    }
}

impl Force for ManyBodyForce {
    fn apply(&self, points: &mut [Point], alpha: f32) {
        let max_x = points.iter().fold(0.0 / 0.0, |m, v| v.x.max(m));
        let min_x = points.iter().fold(0.0 / 0.0, |m, v| v.x.min(m));
        let max_y = points.iter().fold(0.0 / 0.0, |m, v| v.y.max(m));
        let min_y = points.iter().fold(0.0 / 0.0, |m, v| v.y.min(m));
        let width = max_x - min_x;
        let height = max_y - min_y;
        let size = width.max(height);
        let mut tree: Quadtree<Body> = Quadtree::new(Rect {
            cx: (min_x + max_x) / 2.,
            cy: (min_y + max_y) / 2.,
            width: size,
            height: size,
        });
        let root = tree.root();
        for (point, &strength) in points.iter().zip(&self.strength) {
            tree.insert(root, point.x, point.y, strength);
        }
        accumulate(&mut tree, root);
        for point in points.iter_mut() {
            apply_many_body(point, &tree, root, alpha, 0.81);
        }
    }
}

impl ForceToNode for ManyBodyForce {
    fn apply_to_node(&self, i: usize, points: &mut [Point], alpha: f32) {
        let n = points.len();
        for j in 0..n {
            if i == j {
                continue;
            }
            let Point { x, y, .. } = points[j];
            let ref mut point = points[i];
            let dx = x - point.x;
            let dy = y - point.y;
            let l = (dx * dx + dy * dy).max(MIN_DISTANCE);
            point.vx += dx * self.strength[j] * alpha / l;
            point.vy += dy * self.strength[j] * alpha / l;
        }
    }
}

pub fn default_strength_accessor<N, E, Ty: EdgeType, Ix: IndexType>(
    _graph: &Graph<N, E, Ty, Ix>,
    _u: NodeIndex<Ix>,
) -> f32 {
    -30.
}

// #[test]
// fn test_many_body() {
//     let mut points = Vec::new();
//     points.push(Point::new(10., 10.));
//     points.push(Point::new(10., -10.));
//     points.push(Point::new(-10., 10.));
//     points.push(Point::new(-10., -10.));
//     let context = QuadTreeManyBodyForceContext::new(vec![-30., -30., -30., -30.]);
//     context.apply(&mut points, 1.0);
//     assert!(points[0].vx == 2.25);
//     assert!(points[0].vy == 2.25);
//     assert!(points[1].vx == 2.25);
//     assert!(points[1].vy == -2.25);
//     assert!(points[2].vx == -2.25);
//     assert!(points[2].vy == 2.25);
//     assert!(points[3].vx == -2.25);
//     assert!(points[3].vy == -2.25);
// }
