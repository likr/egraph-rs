use crate::Point;

pub trait Force {
    fn apply(&self, points: &mut [Point], alpha: f32);
}

pub trait ForceToNode {
    fn apply_to_node(&self, u: usize, points: &mut [Point], alpha: f32);
}

pub fn update_with<F: FnMut(&mut [Point], f32)>(
    points: &mut [Point],
    alpha: f32,
    velocity_decay: f32,
    f: &mut F,
) {
    f(points, alpha);
    for point in points.iter_mut() {
        point.vx *= velocity_decay;
        point.vy *= velocity_decay;
        point.x += point.vx;
        point.y += point.vy;
    }
}

pub fn apply_forces<T: AsRef<dyn Force>>(
    points: &mut [Point],
    forces: &[T],
    alpha: f32,
    velocity_decay: f32,
) {
    update_with(points, alpha, velocity_decay, &mut |points, alpha| {
        for force in forces {
            force.as_ref().apply(points, alpha);
        }
    });
}

pub fn apply_forces_to_node<T: AsRef<dyn ForceToNode>>(
    points: &mut [Point],
    forces: &[T],
    alpha: f32,
    velocity_decay: f32,
) {
    update_with(points, alpha, velocity_decay, &mut |points, alpha| {
        let n = points.len();
        for force in forces {
            for u in 0..n {
                force.as_ref().apply_to_node(u, points, alpha);
            }
        }
    });
}
