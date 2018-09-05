use std::f64;
use super::Group;

pub struct RadialLayout {
}

impl RadialLayout {
    pub fn new() -> RadialLayout {
        RadialLayout {
        }
    }

    pub fn call(&self, width: f64, height: f64, values: &Vec<f64>) -> Vec<Group> {
        radial_layout(width, height, values)
    }
}

pub fn radial_layout(width: f64, height: f64, values: &Vec<f64>) -> Vec<Group> {
    let mut offset : f64 = 0.;
    let d_theta = f64::consts::PI * 2. / values.len() as f64;
    let r = width.min(height) / 2.;
    let max_r = r * (f64::consts::PI / values.len() as f64).sin();
    let max_value = values.iter().fold(0. / 0., |m, v| v.max(m));
    values.iter()
        .map(|value| {
            let x = r * offset.cos();
            let y = r * offset.sin();
            let size = 2. * max_r * value / max_value;
            offset += d_theta;
            Group::new(x, y, size, size)
        })
        .collect()
}
