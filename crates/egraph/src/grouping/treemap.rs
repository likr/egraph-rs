use super::{group_size, Group};
use crate::utils::treemap::{normalize, squarify};
use crate::Graph;
use std::cmp::Ordering;
use std::collections::HashMap;

pub struct TreemapGrouping<G> {
    pub group: Box<Fn(&Graph<G>, usize) -> usize>,
    pub size: Box<Fn(&Graph<G>, usize) -> f32>,
}

impl<G> TreemapGrouping<G> {
    pub fn new() -> TreemapGrouping<G> {
        TreemapGrouping {
            group: Box::new(|_, _| 0),
            size: Box::new(|_, _| 1.),
        }
    }

    pub fn call(&self, graph: &Graph<G>, width: f32, height: f32) -> HashMap<usize, Group> {
        let values_map = group_size(graph, &self.group, &self.size);
        let mut items = values_map.iter().collect::<Vec<_>>();
        items.sort_by(|item1, item2| {
            if item1.1 == item2.1 {
                Ordering::Equal
            } else if item1.1 < item2.1 {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        });
        let mut values = items.iter().map(|item| *item.1 as f64).collect::<Vec<_>>();
        normalize(&mut values, (width * height) as f64);

        let mut result = HashMap::new();
        for (tile, item) in squarify(width as f64, height as f64, &values)
            .iter()
            .zip(items)
        {
            let g = *item.0;
            result.insert(
                g,
                Group::new(
                    (tile.x + tile.dx / 2.) as f32 - width / 2.,
                    (tile.y + tile.dy / 2.) as f32 - height / 2.,
                    tile.dx as f32,
                    tile.dy as f32,
                ),
            );
        }
        result
    }
}
