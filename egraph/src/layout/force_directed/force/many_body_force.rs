use super::force::{Force, Point};
use ::utils::quadtree::{Element, NodeId, Quadtree, Rect};

#[derive(Copy, Clone, Debug)]
struct Body {
    x: f32,
    y: f32,
    strength: f32,
}

impl Body {
    fn new(x: f32, y: f32, strength: f32) -> Body {
        Body {
            x: x,
            y: y,
            strength: strength,
        }
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
            Element::Leaf { x, y, n } => {
                let strength = -30. * n as f32;
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
                let w = rect.width;
                let l = dx * dx + dy * dy;
                if w * w / theta2 < l {
                    point.vx += dx * data.strength * alpha / l;
                    point.vy += dy * data.strength * alpha / l;
                } else {
                    apply_many_body(point, tree, node_id, alpha, theta2);
                }
            }
            Element::Leaf { x, y, n } => {
                if x != point.x || y != point.y {
                    let strength = -30. * n as f32;
                    let dx = x - point.x;
                    let dy = y - point.y;
                    let l = dx * dx + dy * dy;
                    point.vx += dx * strength * alpha / l;
                    point.vy += dy * strength * alpha / l;
                }
            }
            Element::Empty => {}
        }
    }
}

pub struct ManyBodyForce {}

impl ManyBodyForce {
    pub fn new() -> ManyBodyForce {
        ManyBodyForce {}
    }
}

impl Force for ManyBodyForce {
    fn apply(&self, points: &mut Vec<Point>, alpha: f32) {
        let max_x = points.iter().fold(0.0 / 0.0, |m, v| v.x.max(m));
        let min_x = points.iter().fold(0.0 / 0.0, |m, v| v.x.min(m));
        let max_y = points.iter().fold(0.0 / 0.0, |m, v| v.y.max(m));
        let min_y = points.iter().fold(0.0 / 0.0, |m, v| v.y.min(m));
        let width = max_x - min_x;
        let height = max_y - min_y;
        let mut tree = Quadtree::new(Rect {
            cx: min_x + width / 2.,
            cy: min_y + height / 2.,
            width: width,
            height: height,
        });
        let root = tree.root();
        for point in points.iter() {
            tree.insert(root, point.x, point.y);
        }
        accumulate(&mut tree, root);
        for mut point in points.iter_mut() {
            apply_many_body(&mut point, &tree, root, alpha, 0.81);
        }
    }
}

#[test]
fn test_many_body() {
    let force = ManyBodyForce::new();
    let mut points = Vec::new();
    points.push(Point::new(10., 10.));
    points.push(Point::new(10., -10.));
    points.push(Point::new(-10., 10.));
    points.push(Point::new(-10., -10.));
    force.apply(&mut points, 1.);
    assert!(points[0].vx == 2.25);
    assert!(points[0].vy == 2.25);
    assert!(points[1].vx == 2.25);
    assert!(points[1].vy == -2.25);
    assert!(points[2].vx == -2.25);
    assert!(points[2].vy == 2.25);
    assert!(points[3].vx == -2.25);
    assert!(points[3].vy == -2.25);
}
