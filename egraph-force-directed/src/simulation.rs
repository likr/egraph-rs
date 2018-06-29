use force::{Point, Force};

pub fn start_simulation(points: &mut Vec<Point>, forces: &Vec<Box<Force>>) {
    let mut alpha = 1.;
    let alpha_min = 0.001;
    let alpha_decay = 1. - (alpha_min as f32).powf(1. / 300.);
    let alpha_target = 0.;
    let velocity_decay = 0.6;
    loop {
        alpha += (alpha_target - alpha) * alpha_decay;
        for force in forces.iter() {
            force.apply(points, alpha);
        }
        for point in points.iter_mut() {
            point.vx *= velocity_decay;
            point.x += point.vx;
            point.vy *= velocity_decay;
            point.y += point.vy;
        }
        if alpha < alpha_min {
            break;
        }
    }
}
