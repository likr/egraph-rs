use super::{aggregate_nodes, Group};
use crate::utils::treemap::{normalize, squarify};
use crate::Graph;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::marker::PhantomData;

pub struct TreemapGrouping<D, G: Graph<D>> {
    pub group: Box<dyn Fn(&G, usize) -> usize>,
    pub weight: Box<dyn Fn(&G, usize) -> f32>,
    phantom: PhantomData<D>,
}

impl<D, G: Graph<D>> TreemapGrouping<D, G> {
    pub fn new() -> TreemapGrouping<D, G> {
        TreemapGrouping {
            group: Box::new(|_, _| 0),
            weight: Box::new(|_, _| 1.),
            phantom: PhantomData,
        }
    }

    pub fn call(&self, graph: &G, width: f32, height: f32) -> HashMap<usize, Group> {
        let mut items = aggregate_nodes(graph, &self.group, &self.weight);
        items.sort_by(|item1, item2| {
            if item1.weight == item2.weight {
                Ordering::Equal
            } else if item1.weight < item2.weight {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        });
        let mut values = items
            .iter()
            .map(|item| item.weight as f64)
            .collect::<Vec<_>>();
        normalize(&mut values, (width * height) as f64);

        let mut result = HashMap::new();
        for (tile, item) in squarify(width as f64, height as f64, &values)
            .iter()
            .zip(items)
        {
            let g = item.id;
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
