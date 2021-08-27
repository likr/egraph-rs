use crate::Point;

pub trait Force {
    fn apply(&self, points: &mut [Point], alpha: f32);
}

pub trait ForceToNode {
    fn apply_to_node(&self, u: usize, points: &mut [Point], alpha: f32);
}

pub fn center(points: &mut [Point]) {
    let cx = points.iter().map(|p| p.x).sum::<f32>() / points.len() as f32;
    let cy = points.iter().map(|p| p.y).sum::<f32>() / points.len() as f32;
    for point in points.iter_mut() {
        point.x -= cx;
        point.y -= cy;
    }
}

pub fn update_position(points: &mut [Point], velocity_decay: f32) {
    for point in points.iter_mut() {
        point.vx *= velocity_decay;
        point.vy *= velocity_decay;
        point.x += point.vx;
        point.y += point.vy;
    }
}

pub fn update_with<F: FnMut(&mut [Point], f32)>(
    points: &mut [Point],
    alpha: f32,
    velocity_decay: f32,
    f: &mut F,
) {
    f(points, alpha);
    update_position(points, velocity_decay);
}

pub fn clamp_region(points: &mut [Point], x0: f32, y0: f32, x1: f32, y1: f32) {
    for i in 0..points.len() {
        points[i].x = points[i].x.clamp(x0, x1);
        points[i].y = points[i].y.clamp(y0, y1);
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
