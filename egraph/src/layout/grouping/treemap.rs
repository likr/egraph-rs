use super::{Group, Grouping};
use utils::treemap::squarify;

pub struct TreemapGrouping {}

impl TreemapGrouping {
    pub fn new() -> TreemapGrouping {
        TreemapGrouping {}
    }
}

impl Grouping for TreemapGrouping {
    fn call(&self, width: f64, height: f64, values: &Vec<f64>) -> Vec<Group> {
        squarify(width, height, values)
            .iter()
            .map(|tile| {
                return Group::new(
                    tile.x + tile.dx / 2.,
                    tile.y + tile.dy / 2.,
                    tile.dx,
                    tile.dy,
                );
            })
            .collect()
    }
}
