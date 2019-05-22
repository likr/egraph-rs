use super::{Group, Grouping};
use petgraph::graph::IndexType;
use petgraph::prelude::*;
use petgraph::EdgeType;
use std::collections::HashMap;
use std::f32::consts::PI;

pub struct RadialGrouping<N, E, Ty: EdgeType, Ix: IndexType> {
    pub group: Box<Fn(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> usize>,
    pub size: Box<Fn(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> f32>,
}

impl<N, E, Ty: EdgeType, Ix: IndexType> RadialGrouping<N, E, Ty, Ix> {
    pub fn new() -> RadialGrouping<N, E, Ty, Ix> {
        RadialGrouping {
            group: Box::new(|_, _| 0),
            size: Box::new(|_, _| 1.),
        }
    }
}

impl<N, E, Ty: EdgeType, Ix: IndexType> Grouping<N, E, Ty, Ix> for RadialGrouping<N, E, Ty, Ix> {
    fn call(&self, graph: &Graph<N, E, Ty, Ix>, width: f32, height: f32) -> HashMap<usize, Group> {
        let values = self.group_size(graph, &self.group, &self.size);
        radial_grouping(width, height, &values)
    }
}

pub fn radial_grouping(
    width: f32,
    height: f32,
    values: &HashMap<usize, f32>,
) -> HashMap<usize, Group> {
    let mut offset = 0. as f32;
    let n = values.len() as f32;
    let d_theta = PI * 2. / n;
    let total_value = values.values().fold(0., |s, v| s + v);
    let max_value = values.values().fold(0. / 0., |m, v| v.max(m));
    let large_r = (width * height * max_value / total_value / PI).sqrt() / (PI / n).sin();
    let mut result = HashMap::new();
    for (&g, value) in values {
        let r = (width * height * value / total_value / PI).sqrt();
        let x = large_r * offset.cos();
        let y = large_r * offset.sin();
        offset += d_theta;
        result.insert(g, Group::new(x, y, 2. * r, 2. * r));
    }
    result
}
