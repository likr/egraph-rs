use std::f64::consts::PI;
use super::{Group, Grouping};

pub struct RadialGrouping {
}

impl RadialGrouping {
    pub fn new() -> RadialGrouping {
        RadialGrouping {
        }
    }
}

impl Grouping for RadialGrouping {
    fn call(&self, width: f64, height: f64, values: &Vec<f64>) -> Vec<Group> {
        radial_grouping(width, height, values)
    }
}

pub fn radial_grouping(width: f64, height: f64, values: &Vec<f64>) -> Vec<Group> {
    let mut offset = 0. as f64;
    let n = values.len() as f64;
    let d_theta = PI * 2. / n;
    let total_value = values.iter().fold(0., |s, v| s + v);
    let max_value = values.iter().fold(0. / 0., |m, v| v.max(m));
    let large_r = (width * height * max_value / total_value / PI).sqrt() / (PI / n).sin();
    values.iter()
        .map(|value| {
            let r = (width * height * value / total_value / PI).sqrt();
            let x = large_r * offset.cos();
            let y = large_r * offset.sin();
            offset += d_theta;
            Group::new(x, y, 2. * r, 2. * r)
        })
        .collect()
}
