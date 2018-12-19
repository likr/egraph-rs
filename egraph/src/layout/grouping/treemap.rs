use super::{Group, Grouping};
use petgraph::graph::IndexType;
use petgraph::prelude::*;
use petgraph::EdgeType;
use std::cmp::Ordering;
use std::collections::HashMap;
use utils::treemap::squarify;

pub struct TreemapGrouping<N, E, Ty: EdgeType, Ix: IndexType> {
    pub group: Box<Fn(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> usize>,
    pub size: Box<Fn(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> f32>,
}

impl<N, E, Ty: EdgeType, Ix: IndexType> TreemapGrouping<N, E, Ty, Ix> {
    pub fn new() -> TreemapGrouping<N, E, Ty, Ix> {
        TreemapGrouping {
            group: Box::new(|_, _| 0),
            size: Box::new(|_, _| 1.),
        }
    }
}

impl<N, E, Ty: EdgeType, Ix: IndexType> Grouping<N, E, Ty, Ix> for TreemapGrouping<N, E, Ty, Ix> {
    fn call(&self, graph: &Graph<N, E, Ty, Ix>, width: f32, height: f32) -> HashMap<usize, Group> {
        let values_map = self.group_size(graph, &self.group, &self.size);
        let mut items = values_map.iter().collect::<Vec<_>>();
        items.sort_by(|item1, item2| {
            if item1.1 == item2.1 {
                Ordering::Equal
            } else if item1.1 > item2.1 {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        });
        let values = items.iter().map(|item| *item.1 as f64).collect::<Vec<_>>();

        let mut result = HashMap::new();
        for (tile, item) in squarify(width as f64, height as f64, &values)
            .iter()
            .zip(items)
        {
            let g = *item.0;
            result.insert(
                g,
                Group::new(
                    (tile.x + tile.dx / 2.) as f32,
                    (tile.y + tile.dy / 2.) as f32,
                    tile.dx as f32,
                    tile.dy as f32,
                ),
            );
        }
        result
    }
}
