pub mod map;

use map::Map;
use petgraph::graph::{IndexType, NodeIndex};
use petgraph_layout_force_simulation::{Coordinates, ForceToNode, Point};

pub fn apply_in_non_euclidean_space<M: Map, F: FnMut(usize, &mut [Point])>(
    points: &mut [Point],
    buffer: &mut [Point],
    velocity_decay: f32,
    f: &mut F,
) {
    let n = points.len();
    for u in 0..n {
        let Point { x: cx, y: cy, .. } = points[u];
        for v in 0..n {
            let Point { x, y, .. } = points[v];
            let (zx, zy) = M::to_tangent_space((cx, cy), (x, y));
            buffer[v] = Point::new(zx, zy);
        }
        f(u, buffer);
        points[u].vx = buffer[u].vx;
        points[u].vy = buffer[u].vy;
    }
    for point in points.iter_mut() {
        point.vx *= velocity_decay;
        point.vy *= velocity_decay;
        let (x, y) = M::from_tangent_space((point.x, point.y), (point.vx, point.vy));
        point.x = x;
        point.y = y;
    }
}

pub fn apply_forces_in_non_euclidean_space<M: Map, T: AsRef<dyn ForceToNode>>(
    points: &mut [Point],
    buffer: &mut [Point],
    alpha: f32,
    velocity_decay: f32,
    forces: &[T],
) {
    apply_in_non_euclidean_space::<M, _>(points, buffer, velocity_decay, &mut |u, points| {
        for force in forces {
            force.as_ref().apply_to_node(u, points, alpha);
        }
    });
}

pub fn apply_in_hyperbolic_space<F: FnMut(usize, &mut [Point])>(
    points: &mut [Point],
    buffer: &mut [Point],
    velocity_decay: f32,
    f: &mut F,
) {
    apply_in_non_euclidean_space::<map::HyperbolicSpace, _>(points, buffer, velocity_decay, f);
}

pub fn apply_forces_in_hyperbolic_space<T: AsRef<dyn ForceToNode>>(
    points: &mut [Point],
    buffer: &mut [Point],
    alpha: f32,
    velocity_decay: f32,
    forces: &[T],
) {
    apply_forces_in_non_euclidean_space::<map::HyperbolicSpace, _>(
        points,
        buffer,
        alpha,
        velocity_decay,
        forces,
    );
}

pub fn map_to_tangent_space<M: Map, Ix: IndexType>(
    u: NodeIndex<Ix>,
    source: &Coordinates<Ix>,
    dest: &mut Coordinates<Ix>,
) {
    let x = source.position(u).unwrap();
    for &v in source.indices.iter() {
        let y = source.position(v).unwrap();
        dest.set_position(v, M::to_tangent_space(x, y));
    }
}
