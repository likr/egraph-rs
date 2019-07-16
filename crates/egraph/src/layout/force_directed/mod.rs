pub mod force;
pub mod simulation;

pub use self::force::{Force, Point};
pub use self::simulation::{Simulation, SimulationBuilder};
use std::f32;

pub fn initial_placement(n: usize) -> Vec<Point> {
    (0..n)
        .map(|i| {
            let r = 10. * (i as usize as f32).sqrt();
            let theta = f32::consts::PI * (3. - (5. as f32).sqrt()) * (i as usize as f32);
            let x = r * theta.cos();
            let y = r * theta.sin();
            Point::new(x, y)
        })
        .collect::<Vec<_>>()
}
